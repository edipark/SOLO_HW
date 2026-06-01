#!/usr/bin/env python3
"""Open-loop hardware replay from a sim action log.

Loads the ``.npz`` action log produced by ``play_teacher_with_estimator.py``
(via ``--action-log-output``) and sends the normalized policy actions to the
12 Dynamixel AX-18A servos at the configured control frequency.

The actions go through the **exact same** clip → scale → offset →
joint-limit transform as ``deploy.py`` (``actions_to_joint_targets``).

Usage::

    # Replay on hardware
    python replay_actions.py --actions logs/rollout/actions_run01.npz

    # Dry-run (parse, print info, no servo writes)
    python replay_actions.py --actions logs/rollout/actions_run01.npz --dry-run

    # Override control frequency (default: from config.yaml)
    python replay_actions.py --actions logs/rollout/actions_run01.npz --freq 30

    # Loop replay N times (useful for extended testing)
    python replay_actions.py --actions logs/rollout/actions_run01.npz --loops 3
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np
import yaml

SOLO_WS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOLO_WS_DIR))

from hardware.dynamixel_interface import DynamixelInterface
from utils.action_transform import (
    action_signs_from_config,
    actions_to_joint_targets,
    joint_limits_from_config,
)
from utils.timing import RateController

NUM_JOINTS = 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_hardware(cfg: dict) -> DynamixelInterface:
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


def load_action_log(npz_path: str) -> tuple[np.ndarray, dict]:
    """Load .npz action log from play_teacher_with_estimator.py.

    Returns:
        actions : (T, action_dim) float32 normalized policy actions
        meta    : dict of metadata stored in the npz
    """
    data = np.load(npz_path, allow_pickle=False)
    if "actions" not in data:
        raise ValueError(
            f"'actions' array not found in {npz_path}. "
            "Make sure the file was created with --action-log-output."
        )
    actions = data["actions"].astype(np.float32)
    if actions.ndim != 2:
        raise ValueError(
            f"Expected actions shape [T, action_dim], got {actions.shape}"
        )
    if actions.shape[1] != NUM_JOINTS:
        raise ValueError(
            f"Action dim mismatch: file has {actions.shape[1]}, expected {NUM_JOINTS}"
        )

    meta = {k: data[k].item() if data[k].ndim == 0 else data[k]
            for k in data.files if k != "actions"}
    return actions, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open-loop replay of sim action log on DEXTRA hardware."
    )
    parser.add_argument("--actions", type=str, required=True,
                        help=".npz action log from play_teacher_with_estimator.py")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--freq", type=float, default=None,
                        help="Override control frequency [Hz] (default: from config)")
    parser.add_argument("--loops", type=int, default=1,
                        help="Number of times to replay the log (default: 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and time replay without writing to servos")
    parser.add_argument("--no-home", action="store_true",
                        help="Skip return-to-home after replay finishes")
    args = parser.parse_args()

    # --- Load config ---
    cfg = load_config(args.config)
    freq = args.freq if args.freq is not None else cfg["control"]["frequency_hz"]
    action_scale = float(cfg["control"]["action_scale"])
    action_offset = float(cfg["control"].get("action_offset", 0.0))
    action_signs = action_signs_from_config(cfg)
    joint_lower, joint_upper = joint_limits_from_config(cfg["joints"])
    safety = cfg.get("safety", {})
    startup_hold_sec = float(safety.get("startup_hold_sec", 1.0))
    watchdog_warn_ms = float(safety.get("watchdog_warn_ms", 50.0))

    # --- Load action log ---
    npz_path = str(Path(args.actions).resolve())
    print(f"[replay] Loading action log: {npz_path}")
    actions, meta = load_action_log(npz_path)
    total_frames = actions.shape[0]

    # Log metadata from the npz
    sim_dt = float(meta.get("step_dt", 1.0 / freq))
    teacher_ckpt = str(meta.get("teacher_checkpoint", "unknown"))
    estimator_ckpt = str(meta.get("estimator_checkpoint", "unknown"))
    log_env_id = int(meta.get("env_id", 0))
    print(f"[replay] Frames: {total_frames}  ({total_frames / freq:.1f}s @ {freq}Hz)")
    print(f"[replay] Sim step_dt: {sim_dt*1000:.2f}ms  (recorded at ~{1.0/sim_dt:.1f}Hz)")
    print(f"[replay] Teacher:    {teacher_ckpt}")
    print(f"[replay] Estimator:  {estimator_ckpt}")
    print(f"[replay] Env id:     {log_env_id}")
    print(f"[replay] action_scale={action_scale}, action_offset={action_offset}")
    print(f"[replay] loops={args.loops}")
    if args.dry_run:
        print("[replay] DRY-RUN — no servo commands will be sent")

    # Pre-compute all joint targets so there is no per-step Python overhead
    # inside the tight loop.
    targets_all = np.stack(
        [
            actions_to_joint_targets(actions[i], action_scale, action_offset,
                                     joint_lower, joint_upper,
                                     action_signs=action_signs)
            for i in range(total_frames)
        ],
        axis=0,
    )  # (T, 12) float32

    # --- Initialize hardware ---
    dxl = None
    if not args.dry_run:
        dxl = build_hardware(cfg)
        dxl.connect()
        prev_pos = dxl.read_positions()
        print(f"[replay] Holding start position for {startup_hold_sec:.1f}s...")
        dxl.write_position_targets(prev_pos)
        time.sleep(startup_hold_sec)

    # --- Signal handler ---
    running = True

    def sigint_handler(sig, frame):
        nonlocal running
        print("\n[replay] Interrupt received, stopping after current step...")
        running = False

    signal.signal(signal.SIGINT, sigint_handler)

    # --- Replay loop ---
    rate = RateController(freq)
    total_steps = 0
    overrun_count = 0

    for loop_idx in range(args.loops):
        if not running:
            break
        if args.loops > 1:
            print(f"[replay] Loop {loop_idx + 1}/{args.loops}  ({total_frames} steps)")

        rate.reset()
        t_start = time.monotonic()

        for step in range(total_frames):
            if not running:
                break

            target = targets_all[step]

            if dxl:
                dxl.write_position_targets(target)

            overrun = rate.sleep()
            if overrun:
                overrun_count += 1
                dt_ms = rate.last_dt * 1000.0
                if dt_ms > watchdog_warn_ms:
                    print(
                        f"[replay] WARNING: step {step} loop took {dt_ms:.1f}ms "
                        f"(target: {1000.0/freq:.1f}ms)"
                    )

            total_steps += 1

            if (step + 1) % 100 == 0 or step == total_frames - 1:
                elapsed = time.monotonic() - t_start
                print(
                    f"[replay] loop={loop_idx+1}  step {step+1:4d}/{total_frames}  "
                    f"elapsed={elapsed:.1f}s"
                )

    elapsed_total = time.monotonic() - t_start if 't_start' in dir() else 0.0
    print(
        f"[replay] Finished {total_steps} steps total  "
        f"(overruns: {overrun_count})"
    )

    # --- Shutdown ---
    if dxl:
        if not args.no_home:
            print("[replay] Returning to home position...")
            dxl.go_to_home()
            time.sleep(1.0)
        dxl.disconnect()

    print("[replay] Done.")


if __name__ == "__main__":
    main()
