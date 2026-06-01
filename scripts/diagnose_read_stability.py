#!/usr/bin/env python3
"""Diagnose Dynamixel read stability for selected servo IDs.

This script isolates read-path communication behavior from deploy inference.
It continuously reads present position from selected servos and reports
per-servo failure statistics and loop timing.

Examples:
  python scripts/diagnose_read_stability.py --config config.yaml
  python scripts/diagnose_read_stability.py --config config.yaml --servo-id 4 5 6
  python scripts/diagnose_read_stability.py --config config.yaml --servo-id 4,5,6 --duration 20 --hz 30
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import yaml
from dynamixel_sdk import COMM_SUCCESS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hardware.dynamixel_interface import (  # noqa: E402
    ADDR_PRESENT_POSITION,
    DynamixelInterface,
)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_servo_ids(values: list[str] | None, default_ids: list[int]) -> list[int]:
    if not values:
        return default_ids

    parsed: list[int] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                parsed.append(int(token))

    # Preserve input order while removing duplicates.
    seen = set()
    unique = []
    for sid in parsed:
        if sid not in seen:
            unique.append(sid)
            seen.add(sid)
    return unique


def build_hardware_for_selected(cfg: dict, selected_ids: list[int]) -> DynamixelInterface:
    joint_by_id = {j["servo_id"]: j for j in cfg["joints"]}
    missing = [sid for sid in selected_ids if sid not in joint_by_id]
    if missing:
        raise ValueError(f"Servo ID(s) not found in config: {missing}")

    selected_joints = [joint_by_id[sid] for sid in selected_ids]
    dxl_cfg = cfg["dynamixel"]
    return DynamixelInterface(
        port=dxl_cfg["port"],
        baudrate=dxl_cfg["baudrate"],
        servo_ids=[j["servo_id"] for j in selected_joints],
        offsets_raw=[j["offset_raw"] for j in selected_joints],
        lower_rads=[j["lower_rad"] for j in selected_joints],
        upper_rads=[j["upper_rad"] for j in selected_joints],
        apply_hardware_config=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Diagnose DXL read stability")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--servo-id",
        nargs="+",
        default=None,
        help="Target servo IDs (space or comma separated). Default: 4 5 6",
    )
    parser.add_argument("--duration", type=float, default=20.0,
                        help="Test duration in seconds (default: 20)")
    parser.add_argument("--hz", type=float, default=30.0,
                        help="Read loop frequency (default: 30)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    target_ids = parse_servo_ids(args.servo_id, default_ids=[4, 5, 6])
    dxl = build_hardware_for_selected(cfg, target_ids)

    ok_count = defaultdict(int)
    comm_fail_count = defaultdict(int)
    status_error_count = defaultdict(int)
    exception_count = defaultdict(int)

    loop_count = 0
    loop_warn_count = 0
    max_loop_ms = 0.0
    target_dt = 1.0 / max(args.hz, 1.0)

    print(f"[diag] Port={cfg['dynamixel']['port']} baud={cfg['dynamixel']['baudrate']}")
    print(f"[diag] Target servo IDs: {target_ids}")
    print(f"[diag] Running {args.duration:.1f}s @ {args.hz:.1f}Hz")

    t_start = time.monotonic()

    try:
        dxl.connect()

        while True:
            now = time.monotonic()
            if now - t_start >= args.duration:
                break

            t0 = time.monotonic()
            for sid in target_ids:
                try:
                    raw, result, error = dxl.packet_handler.read2ByteTxRx(
                        dxl.port_handler, sid, ADDR_PRESENT_POSITION
                    )
                    if result != COMM_SUCCESS:
                        comm_fail_count[sid] += 1
                        continue
                    if error != 0:
                        status_error_count[sid] += 1
                        continue
                    # Touch value so type errors are visible during diagnostics.
                    _ = int(raw)
                    ok_count[sid] += 1
                except Exception as exc:  # noqa: BLE001
                    exception_count[sid] += 1
                    print(f"[diag] EXCEPTION sid={sid}: {type(exc).__name__}: {exc}")

            loop_count += 1
            loop_ms = (time.monotonic() - t0) * 1000.0
            if loop_ms > max_loop_ms:
                max_loop_ms = loop_ms
            if loop_ms > (1000.0 / max(args.hz, 1.0)):
                loop_warn_count += 1

            sleep_s = target_dt - (time.monotonic() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)

    finally:
        dxl.disconnect()

    print("\n[diag] === Summary ===")
    print(f"[diag] Loops: {loop_count}, max loop: {max_loop_ms:.2f}ms, overruns: {loop_warn_count}")
    for sid in target_ids:
        total = ok_count[sid] + comm_fail_count[sid] + status_error_count[sid] + exception_count[sid]
        fail = comm_fail_count[sid] + status_error_count[sid] + exception_count[sid]
        fail_pct = (100.0 * fail / total) if total else 0.0
        print(
            f"[diag] SID {sid:2d} | total={total:5d} ok={ok_count[sid]:5d} "
            f"comm_fail={comm_fail_count[sid]:5d} status_err={status_error_count[sid]:5d} "
            f"exceptions={exception_count[sid]:5d} fail_rate={fail_pct:6.2f}%"
        )


if __name__ == "__main__":
    main()
