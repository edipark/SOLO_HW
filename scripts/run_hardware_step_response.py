#!/usr/bin/env python3
"""Run the simulation step-response schedule on one physical AX-18A.

Only the selected servo ID is read or written. Other joints are untouched.
The robot must be suspended so the tested leg can move freely.

Schedule (same defaults as run_sim_step_response.py):
    0 -> +step -> 0 -> -step -> 0

The script saves the original RAM settings of the selected servo and restores
them after the experiment. Present Load is logged only as a qualitative
internal estimate; it is not a measured torque.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import sys
import time
from datetime import datetime

import numpy as np
import yaml
from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLO_HW_ROOT = os.path.dirname(SCRIPT_DIR)

# AX-18A Protocol 1.0 RAM addresses
ADDR_TORQUE_ENABLE = 24
ADDR_CW_COMPLIANCE_MARGIN = 26
ADDR_CCW_COMPLIANCE_MARGIN = 27
ADDR_CW_COMPLIANCE_SLOPE = 28
ADDR_CCW_COMPLIANCE_SLOPE = 29
ADDR_GOAL_POSITION = 30
ADDR_TORQUE_LIMIT = 34
ADDR_PRESENT_POSITION = 36
ADDR_PRESENT_LOAD = 40
ADDR_PUNCH = 48

MAX_POSITION_RAW = 1023
RAD_PER_UNIT = math.radians(300.0) / MAX_POSITION_RAW


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-joint AX-18A hardware step response")
    parser.add_argument("--config", default=os.path.join(SOLO_HW_ROOT, "config.yaml"))
    parser.add_argument("--joint-name", default="L_Thigh_Joint")
    parser.add_argument("--step-deg", type=float, default=5.0)
    parser.add_argument("--sample-hz", type=float, default=120.0)
    parser.add_argument("--settle-s", type=float, default=0.75)
    parser.add_argument("--step-hold-s", type=float, default=1.5)
    parser.add_argument("--center-hold-s", type=float, default=0.8)
    parser.add_argument("--final-hold-s", type=float, default=0.8)
    parser.add_argument(
        "--apply-config", action="store_true",
        help="Temporarily apply YAML compliance margin/slope/punch to this servo only",
    )
    parser.add_argument(
        "--torque-limit-ratio", type=float, default=None,
        help="Temporarily override this servo's Torque Limit; e.g. 0.3 to match the sim run",
    )
    parser.add_argument("--abort-tracking-error-deg", type=float, default=20.0)
    parser.add_argument("--max-consecutive-read-failures", type=int, default=5)
    parser.add_argument(
        "--output-root",
        default=os.path.join(SOLO_HW_ROOT, "logs", "ax18a_sysid", "hardware"),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.step_deg <= 15.0:
        raise ValueError("--step-deg must be in (0, 15]")
    if not 0.0 < args.sample_hz <= 250.0:
        raise ValueError("--sample-hz must be in (0, 250]")
    if args.torque_limit_ratio is not None and not 0.0 < args.torque_limit_ratio <= 1.0:
        raise ValueError("--torque-limit-ratio must be in (0, 1]")
    if args.abort_tracking_error_deg <= args.step_deg:
        raise ValueError("--abort-tracking-error-deg must exceed --step-deg")
    if args.max_consecutive_read_failures < 1:
        raise ValueError("--max-consecutive-read-failures must be at least 1")
    for name in ("settle_s", "step_hold_s", "center_hold_s", "final_hold_s"):
        if getattr(args, name) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def load_config(path: str) -> dict:
    with open(path) as file:
        return yaml.safe_load(file)


def create_output_dir(root: str, joint_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.abspath(os.path.join(root, f"{joint_name}_{timestamp}"))
    os.makedirs(path, exist_ok=False)
    return path


def build_schedule(args: argparse.Namespace) -> tuple[list[dict], float]:
    segments = [
        {"phase": "settle", "duration": args.settle_s, "offset_rad": 0.0},
        {"phase": "positive", "duration": args.step_hold_s, "offset_rad": math.radians(args.step_deg)},
        {"phase": "center", "duration": args.center_hold_s, "offset_rad": 0.0},
        {"phase": "negative", "duration": args.step_hold_s, "offset_rad": -math.radians(args.step_deg)},
        {"phase": "final", "duration": args.final_hold_s, "offset_rad": 0.0},
    ]
    cursor = 0.0
    for segment in segments:
        segment["start_s"] = cursor
        cursor += segment["duration"]
        segment["end_s"] = cursor
    return segments, cursor


def command_at_time(elapsed_s: float, segments: list[dict]) -> tuple[str, float]:
    for segment in segments:
        if elapsed_s < segment["end_s"]:
            return segment["phase"], segment["offset_rad"]
    return segments[-1]["phase"], segments[-1]["offset_rad"]


class SingleAX18A:
    """Minimal Protocol 1.0 connection that addresses exactly one servo."""

    def __init__(self, port: str, baudrate: int, servo_id: int, offset_raw: int, lower_rad: float, upper_rad: float):
        self.servo_id = int(servo_id)
        self.offset_raw = int(offset_raw)
        self.lower_rad = float(lower_rad)
        self.upper_rad = float(upper_rad)
        self.port = PortHandler(port)
        self.packet = PacketHandler(1.0)
        self.baudrate = int(baudrate)
        self.opened = False

    def connect(self) -> None:
        if not self.port.openPort():
            raise RuntimeError("failed to open Dynamixel port")
        self.opened = True
        if not self.port.setBaudRate(self.baudrate):
            raise RuntimeError(f"failed to set baudrate {self.baudrate}")
        _, result, error = self.packet.ping(self.port, self.servo_id)
        self._check("ping", result, error)

    def close(self) -> None:
        if self.opened:
            self.port.closePort()
            self.opened = False

    def _check(self, operation: str, result: int, error: int) -> None:
        if result != COMM_SUCCESS:
            raise RuntimeError(f"{operation}: {self.packet.getTxRxResult(result)}")
        if error:
            raise RuntimeError(f"{operation}: {self.packet.getRxPacketError(error)}")

    def read1(self, address: int) -> int:
        value, result, error = self.packet.read1ByteTxRx(self.port, self.servo_id, address)
        self._check(f"read address {address}", result, error)
        return int(value)

    def read2(self, address: int) -> int:
        value, result, error = self.packet.read2ByteTxRx(self.port, self.servo_id, address)
        self._check(f"read address {address}", result, error)
        return int(value)

    def try_read2(self, address: int) -> int | None:
        value, result, error = self.packet.read2ByteTxRx(self.port, self.servo_id, address)
        if result != COMM_SUCCESS or error:
            return None
        return int(value)

    def write1(self, address: int, value: int) -> None:
        result, error = self.packet.write1ByteTxRx(self.port, self.servo_id, address, int(value))
        self._check(f"write address {address}", result, error)

    def write2(self, address: int, value: int) -> None:
        result, error = self.packet.write2ByteTxRx(self.port, self.servo_id, address, int(value))
        self._check(f"write address {address}", result, error)

    def raw_to_rad(self, raw: int) -> float:
        return (raw - self.offset_raw) * RAD_PER_UNIT

    def rad_to_raw(self, radians: float) -> int:
        clipped = float(np.clip(radians, self.lower_rad, self.upper_rad))
        return int(np.clip(round(clipped / RAD_PER_UNIT + self.offset_raw), 0, MAX_POSITION_RAW))

    def read_position(self) -> float | None:
        raw = self.try_read2(ADDR_PRESENT_POSITION)
        return None if raw is None else self.raw_to_rad(raw)

    def write_position(self, radians: float) -> None:
        self.write2(ADDR_GOAL_POSITION, self.rad_to_raw(radians))

    def read_present_load(self) -> tuple[int, float]:
        raw = self.try_read2(ADDR_PRESENT_LOAD)
        if raw is None:
            return -1, float("nan")
        magnitude = raw & 0x3FF
        direction = -1.0 if raw & 0x400 else 1.0
        return raw, direction * magnitude / 1023.0

    def snapshot_ram(self) -> dict[str, int]:
        return {
            "torque_enable": self.read1(ADDR_TORQUE_ENABLE),
            "cw_margin": self.read1(ADDR_CW_COMPLIANCE_MARGIN),
            "ccw_margin": self.read1(ADDR_CCW_COMPLIANCE_MARGIN),
            "cw_slope": self.read1(ADDR_CW_COMPLIANCE_SLOPE),
            "ccw_slope": self.read1(ADDR_CCW_COMPLIANCE_SLOPE),
            "torque_limit": self.read2(ADDR_TORQUE_LIMIT),
            "punch": self.read2(ADDR_PUNCH),
        }

    def restore_ram(self, snapshot: dict[str, int]) -> None:
        self.write1(ADDR_CW_COMPLIANCE_MARGIN, snapshot["cw_margin"])
        self.write1(ADDR_CCW_COMPLIANCE_MARGIN, snapshot["ccw_margin"])
        self.write1(ADDR_CW_COMPLIANCE_SLOPE, snapshot["cw_slope"])
        self.write1(ADDR_CCW_COMPLIANCE_SLOPE, snapshot["ccw_slope"])
        self.write2(ADDR_TORQUE_LIMIT, snapshot["torque_limit"])
        self.write2(ADDR_PUNCH, snapshot["punch"])
        self.write1(ADDR_TORQUE_ENABLE, snapshot["torque_enable"])


def read_baseline(servo: SingleAX18A, attempts: int = 5) -> float:
    for _ in range(attempts):
        position = servo.read_position()
        if position is not None:
            return position
        time.sleep(0.02)
    raise RuntimeError(f"failed to read servo {servo.servo_id} position {attempts} times")


def save_outputs(output_dir: str, records: list[dict], metadata: dict, segments: list[dict]) -> None:
    fields = [
        "time_s", "phase", "target_rad", "position_rad", "velocity_rad_s",
        "position_error_rad", "present_load_raw", "present_load_signed_ratio",
        "sample_duration_ms", "deadline_missed",
    ]
    with open(os.path.join(output_dir, "responses.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    valid = [row for row in records if np.isfinite(row["position_rad"])]
    velocities = np.asarray([row["velocity_rad_s"] for row in records if np.isfinite(row["velocity_rad_s"])])
    summary = {
        "result": metadata["result"],
        "sample_count": len(records),
        "valid_position_count": len(valid),
        "deadline_miss_count": int(sum(row["deadline_missed"] for row in records)),
        "mean_sample_duration_ms": float(np.mean([row["sample_duration_ms"] for row in records])),
        "max_sample_duration_ms": float(np.max([row["sample_duration_ms"] for row in records])),
        "position_rmse_rad": float(np.sqrt(np.mean([
            (row["target_rad"] - row["position_rad"]) ** 2 for row in valid
        ]))) if valid else None,
        "max_abs_velocity_rad_s": float(np.max(np.abs(velocities))) if velocities.size else None,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as file:
        json.dump(summary, file, indent=2)
    with open(os.path.join(output_dir, "config.json"), "w") as file:
        json.dump(metadata, file, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        print(f"[hw-sysid] Plot skipped: {error}")
        return

    t = np.asarray([row["time_s"] for row in records])
    target = np.asarray([row["target_rad"] for row in records])
    position = np.asarray([row["position_rad"] for row in records])
    velocity = np.asarray([row["velocity_rad_s"] for row in records])
    load = np.asarray([row["present_load_signed_ratio"] for row in records])
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True, constrained_layout=True)
    axes[0].step(t, target, where="post", color="black", linestyle="--", linewidth=1.8)
    axes[0].plot(t, position, color="tab:blue", linewidth=1.3)
    axes[1].plot(t, velocity, color="tab:orange", linewidth=1.1)
    axes[2].plot(t, load, color="tab:red", linewidth=1.0)
    axes[0].set_title("Position: command (black dashed), response (blue)")
    axes[1].set_title("Finite-difference joint velocity")
    axes[2].set_title("Present Load signed ratio (internal estimate, not measured torque)")
    axes[0].set_ylabel("position [rad]")
    axes[1].set_ylabel("velocity [rad/s]")
    axes[2].set_ylabel("load ratio")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylim(-1.05, 1.05)
    for axis in axes:
        for segment in segments[1:]:
            axis.axvline(segment["start_s"], color="gray", linewidth=0.8, alpha=0.45)
        axis.grid(True, alpha=0.25)
    fig.suptitle(
        f"Hardware AX-18A step response — {metadata['joint_name']}, "
        f"±{metadata['step_deg']:g}°, torque ratio={metadata['used_ram']['torque_limit'] / 1023:.3f}",
        fontsize=14,
    )
    fig.savefig(os.path.join(output_dir, "step_response.png"), dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_args(args)
    cfg = load_config(args.config)
    joints = cfg["joints"]
    matches = [(index, joint) for index, joint in enumerate(joints) if joint["name"] == args.joint_name]
    if len(matches) != 1:
        raise ValueError(f"joint {args.joint_name!r} not found exactly once")
    joint_index, joint = matches[0]
    dxl_cfg = cfg["dynamixel"]
    servo = SingleAX18A(
        port=dxl_cfg["port"],
        baudrate=dxl_cfg["baudrate"],
        servo_id=joint["servo_id"],
        offset_raw=joint["offset_raw"],
        lower_rad=joint["lower_rad"],
        upper_rad=joint["upper_rad"],
    )
    segments, duration_s = build_schedule(args)
    output_dir = create_output_dir(args.output_root, args.joint_name)
    step_rad = math.radians(args.step_deg)
    period_s = 1.0 / args.sample_hz
    tracking_limit_rad = math.radians(args.abort_tracking_error_deg)

    running = True
    result = "completed"
    original_ram = None
    used_ram = None
    baseline = None
    records: list[dict] = []

    def request_stop(_signum, _frame) -> None:
        nonlocal running, result
        result = "user_interrupt"
        running = False

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        servo.connect()
        original_ram = servo.snapshot_ram()
        baseline = read_baseline(servo)
        if baseline - step_rad < servo.lower_rad or baseline + step_rad > servo.upper_rad:
            raise RuntimeError("step would cross the configured joint limit")

        requested_torque_limit = (
            round(1023 * args.torque_limit_ratio)
            if args.torque_limit_ratio is not None
            else original_ram["torque_limit"]
        )
        print("\n[hw-sysid] SINGLE-SERVO suspended step response")
        print(f"  Joint:       {args.joint_name} (config index={joint_index})")
        print(f"  Servo ID:    {joint['servo_id']} — no other servo ID will be addressed")
        print(f"  Baseline:    {baseline:+.5f} rad ({math.degrees(baseline):+.2f} deg)")
        print(f"  Command:     0 -> +{args.step_deg:g}° -> 0 -> -{args.step_deg:g}° -> 0")
        print(f"  Sample rate: {args.sample_hz:g} Hz")
        print(f"  Torque limit:{requested_torque_limit}/1023 ({requested_torque_limit/1023:.3f})")
        print(f"  Output:      {output_dir}")
        print("\n  Robot must be SUSPENDED. Keep clear of the tested leg and hold the power switch.")
        if input("  Type RUN to start: ").strip() != "RUN":
            result = "confirmation_cancelled"
            return

        # Set the current measured pose as goal before enabling torque to avoid a startup jump.
        servo.write_position(baseline)
        if args.apply_config:
            servo.write1(ADDR_CW_COMPLIANCE_MARGIN, dxl_cfg.get("compliance_margin", 1))
            servo.write1(ADDR_CCW_COMPLIANCE_MARGIN, dxl_cfg.get("compliance_margin", 1))
            servo.write1(ADDR_CW_COMPLIANCE_SLOPE, dxl_cfg.get("compliance_slope", 64))
            servo.write1(ADDR_CCW_COMPLIANCE_SLOPE, dxl_cfg.get("compliance_slope", 64))
            servo.write2(ADDR_PUNCH, dxl_cfg.get("punch", 32))
        if args.torque_limit_ratio is not None:
            servo.write2(ADDR_TORQUE_LIMIT, round(1023 * args.torque_limit_ratio))
        servo.write1(ADDR_TORQUE_ENABLE, 1)
        used_ram = servo.snapshot_ram()

        start = time.monotonic()
        next_deadline = start
        previous_phase = None
        previous_position = baseline
        previous_time = start
        consecutive_failures = 0

        while running:
            loop_start = time.monotonic()
            elapsed = loop_start - start
            if elapsed >= duration_s:
                break
            phase, offset = command_at_time(elapsed, segments)
            target = baseline + offset
            if phase != previous_phase:
                servo.write_position(target)
                previous_phase = phase
                print(f"[hw-sysid] t={elapsed:6.3f}s phase={phase:8s} target={target:+.5f} rad")

            position = servo.read_position()
            load_raw, load_ratio = servo.read_present_load()
            sample_time = time.monotonic()
            if position is None:
                position_value = float("nan")
                velocity = float("nan")
                consecutive_failures += 1
            else:
                position_value = position
                dt = sample_time - previous_time
                velocity = (position - previous_position) / dt if dt > 0 else float("nan")
                previous_position = position
                previous_time = sample_time
                consecutive_failures = 0
                if abs(target - position) > tracking_limit_rad:
                    result = "tracking_error_limit"
                    running = False
            if consecutive_failures >= args.max_consecutive_read_failures:
                result = "position_read_failures"
                running = False

            next_deadline += period_s
            sleep_s = next_deadline - time.monotonic()
            records.append({
                "time_s": sample_time - start,
                "phase": phase,
                "target_rad": target,
                "position_rad": position_value,
                "velocity_rad_s": velocity,
                "position_error_rad": target - position_value,
                "present_load_raw": load_raw,
                "present_load_signed_ratio": load_ratio,
                "sample_duration_ms": (time.monotonic() - loop_start) * 1000.0,
                "deadline_missed": int(sleep_s <= 0.0),
            })
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_deadline = time.monotonic()

    finally:
        if servo.opened:
            try:
                if baseline is not None:
                    servo.write_position(baseline)
                    time.sleep(0.6)
                if original_ram is not None:
                    servo.restore_ram(original_ram)
            finally:
                servo.close()

        if baseline is not None and used_ram is not None and records:
            metadata = {
                "joint_name": args.joint_name,
                "joint_index": joint_index,
                "servo_id": joint["servo_id"],
                "baseline_rad": baseline,
                "step_deg": args.step_deg,
                "requested_sample_hz": args.sample_hz,
                "segments": segments,
                "apply_config": args.apply_config,
                "requested_torque_limit_ratio": args.torque_limit_ratio,
                "original_ram": original_ram,
                "used_ram": used_ram,
                "result": result,
                "present_load_is_inferred_not_measured_torque": True,
            }
            save_outputs(output_dir, records, metadata, segments)
            print(f"[hw-sysid] Result:  {result}")
            print(f"[hw-sysid] CSV:     {os.path.join(output_dir, 'responses.csv')}")
            print(f"[hw-sysid] Plot:    {os.path.join(output_dir, 'step_response.png')}")
            print(f"[hw-sysid] Summary: {os.path.join(output_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
