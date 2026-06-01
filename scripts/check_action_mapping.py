#!/usr/bin/env python3
"""Check policy-action to hardware-joint mapping one joint at a time.

This script sends a small positive and negative action-equivalent position
command to each configured joint. Use it during bring-up to verify:

1. policy action index -> config joint name -> servo ID mapping,
2. positive action direction vs the URDF positive joint axis,
3. readback sign and approximate target tracking.

Examples:

    # Safest: use the current robot pose as neutral and test every joint
    python scripts/check_action_mapping.py --config config.yaml

    # Test the exact deploy action equation around action=0 / joint target=0
    python scripts/check_action_mapping.py --config config.yaml --neutral home

    # Test one joint only, with manual yes/no notes after each + pulse
    python scripts/check_action_mapping.py --config config.yaml --joint L_Thigh_Joint --interactive
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

SOLO_WS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOLO_WS_DIR))

from utils.action_transform import (
    action_signs_from_config,
    actions_to_joint_targets,
    joint_limits_from_config,
    joint_targets_to_actions,
)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_hardware(cfg: dict) -> DynamixelInterface:
    from hardware.dynamixel_interface import DynamixelInterface

    joints = cfg["joints"]
    dxl_cfg = cfg["dynamixel"]
    return DynamixelInterface(
        port=dxl_cfg["port"],
        baudrate=dxl_cfg["baudrate"],
        servo_ids=[j["servo_id"] for j in joints],
        offsets_raw=[j["offset_raw"] for j in joints],
        lower_rads=[j["lower_rad"] for j in joints],
        upper_rads=[j["upper_rad"] for j in joints],
        apply_hardware_config=dxl_cfg.get("apply_hardware_config", True),
        compliance_margin=dxl_cfg.get("compliance_margin", 1),
        compliance_slope=dxl_cfg.get("compliance_slope", 64),
        punch=dxl_cfg.get("punch", 32),
        torque_limit_ratio=dxl_cfg.get("torque_limit_ratio", 0.96),
    )


def default_urdf_path() -> Path:
    return SOLO_WS_DIR / "assets/Dextra_lowerbody.urdf"


def load_joint_axes(urdf_path: Path) -> dict[str, str]:
    if not urdf_path.exists():
        return {}

    root = ET.parse(urdf_path).getroot()
    axes = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        axis = joint.find("axis")
        if name and axis is not None:
            axes[name] = axis.attrib.get("xyz", "unknown")
    return axes


def resolve_joint_indices(tokens: list[str], joints: list[dict]) -> list[int]:
    if not tokens:
        return list(range(len(joints)))

    by_name = {j["name"]: i for i, j in enumerate(joints)}
    by_servo_id = {str(j["servo_id"]): i for i, j in enumerate(joints)}
    indices = []

    for token in tokens:
        if token in by_name:
            indices.append(by_name[token])
        elif token.startswith("idx:") and token[4:].isdigit():
            idx = int(token[4:])
            if idx < 0 or idx >= len(joints):
                raise ValueError(f"Joint index out of range: {idx}")
            indices.append(idx)
        elif token.startswith("id:") and token[3:] in by_servo_id:
            indices.append(by_servo_id[token[3:]])
        elif token in by_servo_id:
            indices.append(by_servo_id[token])
        else:
            names = ", ".join(j["name"] for j in joints)
            raise ValueError(f"Unknown joint selector '{token}'. Known joints: {names}")

    return sorted(set(indices))


def ramp_policy_targets(dxl: DynamixelInterface, start: np.ndarray, end: np.ndarray,
                        action_scale: float, action_offset: float,
                        action_signs: np.ndarray,
                        joint_lower: np.ndarray, joint_upper: np.ndarray,
                        duration: float, frequency: float):
    start_action = joint_targets_to_actions(start, action_scale, action_offset, action_signs)
    end_action = joint_targets_to_actions(end, action_scale, action_offset, action_signs)
    steps = max(1, int(duration * frequency))
    for step in range(1, steps + 1):
        alpha = step / steps
        action = (1.0 - alpha) * start_action + alpha * end_action
        target = actions_to_joint_targets(
            action,
            action_scale,
            action_offset,
            joint_lower,
            joint_upper,
            action_signs,
        )
        dxl.write_position_targets(target.astype(np.float32))
        time.sleep(1.0 / frequency)


def print_mapping_table(
    joints: list[dict], axes: dict[str, str], action_scale: float, step_rad: float
):
    print("\n=== Config action index -> joint -> servo mapping ===")
    print(f"{'idx':>3s}  {'joint':<24s}  {'id':>3s}  {'URDF +axis':>10s}  {'+step action':>12s}")
    print("-" * 62)
    action_for_step = step_rad / action_scale
    for idx, joint in enumerate(joints):
        name = joint["name"]
        print(f"{idx:3d}  {name:<24s}  {joint['servo_id']:3d}  "
              f"{axes.get(name, 'unknown'):>10s}  {action_for_step:12.5f}")


def policy_target_from_requested(
    cfg: dict,
    requested_target: np.ndarray,
    action_signs: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    action_scale = float(cfg["control"]["action_scale"])
    action_offset = float(cfg["control"].get("action_offset", 0.0))
    action = joint_targets_to_actions(requested_target, action_scale, action_offset, action_signs)
    target = actions_to_joint_targets(
        action,
        action_scale,
        action_offset,
        joint_lower,
        joint_upper,
        action_signs,
    )
    return action, target


def run_joint_test(dxl: DynamixelInterface, cfg: dict, indices: list[int], axes: dict[str, str],
                   neutral: np.ndarray, step_rad: float, hold_sec: float,
                   settle_sec: float, ramp_sec: float, frequency: float,
                   interactive: bool):
    joints = cfg["joints"]
    action_scale = float(cfg["control"]["action_scale"])
    action_offset = float(cfg["control"].get("action_offset", 0.0))
    action_signs = action_signs_from_config(cfg)
    joint_lower, joint_upper = joint_limits_from_config(joints)
    equivalent_action = step_rad / action_scale
    summaries = []
    min_effective_ratio = 0.30
    min_response_rad = max(0.01, 0.20 * step_rad)

    print("\n=== One-joint action pulse test ===")
    print(f"step={step_rad:.4f} rad ({math.degrees(step_rad):.2f} deg), "
          f"equivalent normalized action={equivalent_action:.5f}")
    print("Watch the robot: the named joint should move in the URDF +axis direction on the + pulse.")

    neutral_action, neutral_target = policy_target_from_requested(
        cfg,
        neutral,
        action_signs,
        joint_lower,
        joint_upper,
    )

    current = dxl.read_positions()
    ramp_policy_targets(dxl, current, neutral_target, action_scale, action_offset,
                        action_signs,
                        joint_lower, joint_upper, ramp_sec, frequency)
    time.sleep(settle_sec)

    for idx in indices:
        joint = joints[idx]
        name = joint["name"]
        servo_id = joint["servo_id"]
        axis = axes.get(name, "unknown")

        print(f"\n--- idx={idx} joint={name} servo_id={servo_id} URDF +axis={axis} ---")
        print(f"Neutral command: action[{idx}]={neutral_action[idx]:+.5f}, "
              f"target[{idx}]={neutral_target[idx]:+.4f} rad")

        base = dxl.read_positions()
        ramp_policy_targets(dxl, base, neutral_target, action_scale, action_offset,
                    action_signs,
                            joint_lower, joint_upper, ramp_sec, frequency)
        time.sleep(settle_sec)
        before = dxl.read_positions()

        positive_requested = neutral_target.copy()
        positive_requested[idx] = neutral_target[idx] + step_rad
        positive_action, positive_target = policy_target_from_requested(
            cfg, positive_requested, action_signs, joint_lower, joint_upper)
        positive_command_delta = positive_target[idx] - neutral_target[idx]
        print(f"Sending + pulse through deploy transform: action[{idx}]={positive_action[idx]:+.5f}, "
              f"target delta={positive_command_delta:+.4f} rad")
        if not math.isclose(float(positive_command_delta), step_rad, rel_tol=0.0, abs_tol=1e-4):
            print(
                f"  WARNING: requested +{step_rad:.4f} rad was clipped to "
                f"{positive_command_delta:+.4f} rad"
            )
        ramp_policy_targets(dxl, neutral_target, positive_target, action_scale, action_offset,
                    action_signs,
                            joint_lower, joint_upper, ramp_sec, frequency)
        time.sleep(hold_sec)
        after_positive = dxl.read_positions()
        positive_delta = after_positive - before
        dominant_positive = int(np.argmax(np.abs(positive_delta)))

        print(f"Readback after + pulse: delta[{idx}]={positive_delta[idx]:+.4f} rad, "
              f"dominant idx={dominant_positive} "
              f"({joints[dominant_positive]['name']}, {positive_delta[dominant_positive]:+.4f} rad)")

        note = ""
        if interactive:
            note = input("Did the printed joint move in URDF + direction? [y/n/q] ").strip().lower()
            if note == "q":
                raise KeyboardInterrupt

        print(f"Returning through deploy transform: action[{idx}]={neutral_action[idx]:+.5f}")
        ramp_policy_targets(dxl, positive_target, neutral_target, action_scale, action_offset,
                    action_signs,
                            joint_lower, joint_upper, ramp_sec, frequency)
        time.sleep(settle_sec)
        before_negative = dxl.read_positions()

        negative_requested = neutral_target.copy()
        negative_requested[idx] = neutral_target[idx] - step_rad
        negative_action, negative_target = policy_target_from_requested(
            cfg, negative_requested, action_signs, joint_lower, joint_upper)
        negative_command_delta = negative_target[idx] - neutral_target[idx]
        print(f"Sending - pulse through deploy transform: action[{idx}]={negative_action[idx]:+.5f}, "
              f"target delta={negative_command_delta:+.4f} rad")
        if not math.isclose(float(negative_command_delta), -step_rad, rel_tol=0.0, abs_tol=1e-4):
            print(
                f"  WARNING: requested -{step_rad:.4f} rad was clipped to "
                f"{negative_command_delta:+.4f} rad"
            )
        ramp_policy_targets(dxl, neutral_target, negative_target, action_scale, action_offset,
                    action_signs,
                            joint_lower, joint_upper, ramp_sec, frequency)
        time.sleep(hold_sec)
        after_negative = dxl.read_positions()
        negative_delta = after_negative - before_negative
        dominant_negative = int(np.argmax(np.abs(negative_delta)))

        print(f"Readback after - pulse: delta[{idx}]={negative_delta[idx]:+.4f} rad, "
              f"dominant idx={dominant_negative} "
              f"({joints[dominant_negative]['name']}, {negative_delta[dominant_negative]:+.4f} rad)")

        ramp_policy_targets(dxl, negative_target, neutral_target, action_scale, action_offset,
                    action_signs,
                            joint_lower, joint_upper, ramp_sec, frequency)
        time.sleep(settle_sec)

        response_gain = 0.5 * (positive_delta[idx] - negative_delta[idx])
        dominant_ok = (dominant_positive == idx) and (dominant_negative == idx)
        effective = abs(response_gain) >= min_response_rad

        if not effective:
            verdict = "too-small"
        elif response_gain > 0 and dominant_ok:
            verdict = "direction-ok"
        elif response_gain < 0 and dominant_ok:
            verdict = "direction-flipped"
        elif response_gain > 0:
            verdict = "cross-coupled+"
        else:
            verdict = "cross-coupled-"

        ratio = abs(response_gain) / max(step_rad, 1e-6)
        if effective and ratio < min_effective_ratio:
            verdict = f"weak-response({ratio:.2f})"

        print(
            f"Auto verdict: {verdict}, response_gain={response_gain:+.4f} rad "
            f"(expected sign: + for URDF +axis)"
        )

        summaries.append(
            (idx, name, servo_id, axis, positive_delta[idx], negative_delta[idx], response_gain, verdict, note)
        )

    print("\n=== Summary ===")
    print(
        f"{'idx':>3s}  {'joint':<24s}  {'id':>3s}  {'+delta':>9s}  {'-delta':>9s}  "
        f"{'gain':>9s}  {'verdict':>18s}  {'manual':>8s}"
    )
    print("-" * 108)
    for idx, name, servo_id, axis, positive_delta, negative_delta, response_gain, verdict, note in summaries:
        manual = note if note else "n/a"
        print(
            f"{idx:3d}  {name:<24s}  {servo_id:3d}  {positive_delta:+9.4f}  {negative_delta:+9.4f}  "
            f"{response_gain:+9.4f}  {verdict:>18s}  {manual:>8s}"
        )

    if interactive:
        flipped = [name for _, name, _, _, _, _, _, _, note in summaries if note == "n"]
        if flipped:
            print("\nManual notes marked these joints as sign-flipped relative to URDF +axis:")
            for name in flipped:
                print(f"  - {name}")
            print("Add a per-joint sign correction for these before deployment.")

    auto_flipped = [name for _, name, _, _, _, _, _, verdict, _ in summaries if verdict == "direction-flipped"]
    if auto_flipped:
        print("\nAuto-check marked these joints as sign-flipped from encoder response:")
        for name in auto_flipped:
            print(f"  - {name}")
        print("These joints likely need sign inversion in deploy/action mapping.")


def main():
    parser = argparse.ArgumentParser(description="Check policy action to hardware joint mapping.")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--joint", action="append", default=[],
                        help="Joint selector: exact name, servo ID, id:<ID>, or idx:<index>. Repeatable.")
    parser.add_argument("--urdf", type=str, default=str(default_urdf_path()),
                        help="URDF path used to print positive joint axes")
    parser.add_argument("--neutral", choices=("current", "home"), default="current",
                        help="Use current pose or action-zero home pose as neutral")
    parser.add_argument("--step-rad", type=float, default=0.1,
                        help="Small test pulse amplitude in radians")
    parser.add_argument("--step-deg", type=float, default=None,
                        help="Override test pulse amplitude in degrees")
    parser.add_argument("--hold-sec", type=float, default=0.5,
                        help="Hold each pulse before readback")
    parser.add_argument("--settle-sec", type=float, default=0.25,
                        help="Settling time after returning to neutral")
    parser.add_argument("--ramp-sec", type=float, default=0.6,
                        help="Ramp duration for each small move")
    parser.add_argument("--frequency", type=float, default=30.0,
                        help="Write frequency during ramps")
    parser.add_argument("--interactive", action="store_true",
                        help="Ask whether each + pulse matches URDF + direction")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the start confirmation prompt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    joints = cfg["joints"]
    axes = load_joint_axes(Path(args.urdf))
    indices = resolve_joint_indices(args.joint, joints)
    step_rad = math.radians(args.step_deg) if args.step_deg is not None else args.step_rad
    if step_rad <= 0.0:
        raise ValueError("Test pulse amplitude must be positive")

    print_mapping_table(joints, axes, float(cfg["control"]["action_scale"]), step_rad)
    print("\nThis test will enable torque and move one joint at a time with small ramps.")
    print("Keep the robot supported, clear the legs, and be ready to cut power.")
    if args.neutral == "home":
        print("Neutral mode: home. The test uses the exact deploy action-zero target: all joints = 0 rad.")
    else:
        print(
            "Neutral mode: current. "
            "This is safer for bench checks but tests relative direction around the current pose."
        )

    if not args.yes:
        answer = input("Continue? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    dxl = build_hardware(cfg)
    running = True

    def sigint_handler(sig, frame):
        nonlocal running
        running = False
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, sigint_handler)

    try:
        dxl.connect()
        current = dxl.read_positions()
        action_offset = float(cfg["control"].get("action_offset", 0.0))
        neutral = (
            np.full(len(joints), action_offset, dtype=np.float32)
            if args.neutral == "home"
            else current.copy()
        )
        if not running:
            return
        run_joint_test(
            dxl=dxl,
            cfg=cfg,
            indices=indices,
            axes=axes,
            neutral=neutral,
            step_rad=step_rad,
            hold_sec=args.hold_sec,
            settle_sec=args.settle_sec,
            ramp_sec=args.ramp_sec,
            frequency=args.frequency,
            interactive=args.interactive,
        )
    except KeyboardInterrupt:
        print("\nInterrupted. Returning to neutral before disconnect...")
        try:
            current = dxl.read_positions()
            action_offset = float(cfg["control"].get("action_offset", 0.0))
            action_signs = action_signs_from_config(cfg)
            neutral = (
                np.full(len(joints), action_offset, dtype=np.float32)
                if args.neutral == "home"
                else current.copy()
            )
            joint_lower, joint_upper = joint_limits_from_config(joints)
            _, neutral_target = policy_target_from_requested(
                cfg,
                neutral,
                action_signs,
                joint_lower,
                joint_upper,
            )
            ramp_policy_targets(
                dxl,
                current,
                neutral_target,
                float(cfg["control"]["action_scale"]),
                action_offset,
                action_signs,
                joint_lower,
                joint_upper,
                args.ramp_sec,
                args.frequency,
            )
        except Exception as exc:
            print(f"Return-to-neutral skipped after error: {exc}")
    finally:
        dxl.disconnect()


if __name__ == "__main__":
    main()