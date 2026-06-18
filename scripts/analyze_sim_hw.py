#!/usr/bin/env python3
"""Analyze sim-to-real discrepancies and visualize joint trajectories.

Generates three plot types:

  1. **action_comparison.png**  — sim actions (npz) vs hardware policy actions
     (deploy CSV).  Requires ``--sim_npz`` + ``--deploy_csv``.

  2. **joint_positions.png**  — measured joint positions during open-loop replay
     vs during policy inference.  Requires ``--replay_csv`` + ``--deploy_csv``.

  3. **tracking_error.png**  — commanded targets vs measured positions during
     replay, showing how well each servo tracks.  Requires ``--replay_csv``.

Usage::

    # 1. Compare sim actions vs hw policy actions
    python scripts/analyze_sim_hw.py \\
        --sim_npz  logs/rollout/actions_run03.npz \\
        --deploy_csv logs/deploy_YYYYMMDD_HHMMSS.csv

    # 2. Compare joint angles: replay vs policy inference
    python scripts/analyze_sim_hw.py \\
        --replay_csv logs/replay_YYYYMMDD_HHMMSS.csv \\
        --deploy_csv logs/deploy_YYYYMMDD_HHMMSS.csv

    # 3. Full analysis (all three plots)
    python scripts/analyze_sim_hw.py \\
        --sim_npz  logs/rollout/actions_run03.npz \\
        --replay_csv logs/replay_YYYYMMDD_HHMMSS.csv \\
        --deploy_csv logs/deploy_YYYYMMDD_HHMMSS.csv \\
        --out_dir logs/analysis
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

NUM_JOINTS = 12

# Per-joint CRB inertia [kg·m²] extracted from URDF (see logs/sysid/recommendations.txt).
# Mirror joints (R side) share the primary joint's measured value (bilateral symmetry).
SYSID_INERTIA_KGM2 = np.array([
    2.86e-4, 2.86e-4,   # L/R HipYaw
    5.13e-3, 5.13e-3,   # L/R HipRoll
    5.17e-3, 5.17e-3,   # L/R Thigh
    1.19e-3, 1.19e-3,   # L/R Calf
    1.98e-4, 1.98e-4,   # L/R AnklePitch
    2.67e-5, 2.67e-5,   # L/R AnkleRoll
], dtype=np.float64)

# AX-18A position resolution: 1 step = 0.29° in radians (from ax18a.py)
_AX18A_RAD_PER_STEP: float = 0.29 * np.pi / 180.0  # ≈0.005061 rad/step


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_csv(path: str) -> dict[str, np.ndarray]:
    """Load a CSV file with a header row into a dict of column→ndarray."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    result: dict[str, np.ndarray] = {}
    for key in rows[0]:
        try:
            result[key] = np.array([float(r[key]) for r in rows], dtype=np.float64)
        except ValueError:
            result[key] = np.array([r[key] for r in rows])
    return result


def extract_cols(data: dict, prefix: str, n: int = NUM_JOINTS) -> np.ndarray:
    """Stack data['{prefix}_0'] … data['{prefix}_{n-1}'] → (T, n) float32."""
    return np.stack([data[f"{prefix}_{i}"] for i in range(n)], axis=1).astype(np.float32)


def load_npz_actions(path: str) -> np.ndarray:
    """Load actions from a sim .npz log → (T, 12) float32."""
    d = np.load(path, allow_pickle=False)
    if "actions" not in d:
        raise ValueError(f"'actions' key not found in {path}")
    return d["actions"].astype(np.float32)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _make_fig(title: str) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(4, 3, figsize=(16, 12))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    return fig, axes


def _finish_fig(fig: plt.Figure, path: str) -> None:
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] Saved: {path}")


def _rmse_label(ax: plt.Axes, rmse: float, unit: str = "") -> None:
    label = f"RMSE={rmse:.4f}{(' ' + unit) if unit else ''}"
    ax.text(0.02, 0.03, label, transform=ax.transAxes,
            fontsize=6, color="firebrick",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))


# ---------------------------------------------------------------------------
# Plot 1 — action comparison: sim npz  vs  hw deploy CSV
# ---------------------------------------------------------------------------

def plot_action_comparison(
    sim_actions: np.ndarray,
    hw_actions: np.ndarray,
    freq: float,
    joint_names: list[str],
    out_path: str,
) -> None:
    T = min(len(sim_actions), len(hw_actions))
    sim_a = sim_actions[:T]
    hw_a  = hw_actions[:T]
    t = np.arange(T) / freq

    rmse = np.sqrt(np.mean((sim_a - hw_a) ** 2, axis=0))

    print("\n[analyze] Action RMSE  (sim vs hw policy):")
    for name, r in zip(joint_names, rmse):
        print(f"  {name:<24}  {r:.4f}")
    print(f"  {'MEAN':<24}  {rmse.mean():.4f}")

    fig, axes = _make_fig("Action Comparison: Sim (blue) vs Hardware Policy (orange)")
    for j in range(NUM_JOINTS):
        r, c = divmod(j, 3)
        ax = axes[r][c]
        ax.plot(t, sim_a[:, j], color="tab:blue",   lw=1.0, alpha=0.85, label="sim")
        ax.plot(t, hw_a[:,  j], color="tab:orange", lw=1.0, alpha=0.85, label="hw policy")
        ax.set_title(joint_names[j], fontsize=8, pad=2)
        ax.set_xlabel("time (s)", fontsize=7)
        ax.set_ylabel("action [-1, 1]", fontsize=7)
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(0, color="black", lw=0.4, ls=":")
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right")
        _rmse_label(ax, rmse[j])

    _finish_fig(fig, out_path)


# ---------------------------------------------------------------------------
# Plot 2 — joint positions: replay measured  vs  deploy measured
# ---------------------------------------------------------------------------

def plot_joint_positions(
    replay_tgt:  np.ndarray,   # (T_r, 12)  commanded target during replay
    replay_pos:  np.ndarray,   # (T_r, 12)  measured position during replay
    deploy_pos:  np.ndarray,   # (T_d, 12)  measured position during policy deploy
    freq: float,
    joint_names: list[str],
    out_path: str,
) -> None:
    T = min(len(replay_pos), len(deploy_pos))
    t = np.arange(T) / freq

    rmse = np.sqrt(np.mean((replay_pos[:T] - deploy_pos[:T]) ** 2, axis=0))

    print("\n[analyze] Joint position RMSE  (replay measured vs deploy measured):")
    for name, r in zip(joint_names, rmse):
        print(f"  {name:<24}  {r:.4f} rad")
    print(f"  {'MEAN':<24}  {rmse.mean():.4f} rad")

    fig, axes = _make_fig(
        "Joint Positions: Replay cmd (gray dashed) · Replay measured (blue) · Policy measured (orange)"
    )
    for j in range(NUM_JOINTS):
        r, c = divmod(j, 3)
        ax = axes[r][c]
        ax.plot(t, replay_tgt[:T, j], color="gray",       lw=0.8, ls="--", alpha=0.6, label="replay cmd")
        ax.plot(t, replay_pos[:T, j], color="tab:blue",   lw=1.2, alpha=0.85, label="replay meas")
        ax.plot(t, deploy_pos[:T, j], color="tab:orange", lw=1.2, alpha=0.85, label="policy meas")
        ax.set_title(joint_names[j], fontsize=8, pad=2)
        ax.set_xlabel("time (s)", fontsize=7)
        ax.set_ylabel("position (rad)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right")
        _rmse_label(ax, rmse[j], "rad")

    _finish_fig(fig, out_path)


# ---------------------------------------------------------------------------
# Plot 3 — servo tracking error during replay
# ---------------------------------------------------------------------------

def plot_tracking_error(
    replay_tgt: np.ndarray,   # (T, 12)
    replay_pos: np.ndarray,   # (T, 12)
    freq: float,
    joint_names: list[str],
    out_path: str,
) -> None:
    T = len(replay_pos)
    t = np.arange(T) / freq
    err = replay_tgt - replay_pos

    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    max_err = np.abs(err).max(axis=0)

    print("\n[analyze] Tracking error  (replay target − measured):")
    print(f"  {'joint':<24}  {'RMSE (rad)':>10}  {'max |err| (rad)':>16}")
    for name, r, m in zip(joint_names, rmse, max_err):
        print(f"  {name:<24}  {r:>10.4f}  {m:>16.4f}")
    print(f"  {'MEAN':<24}  {rmse.mean():>10.4f}  {max_err.mean():>16.4f}")

    fig, axes = _make_fig(
        "Servo Tracking: Target (gray dashed) · Measured (blue) · Error = target−meas (red)"
    )
    for j in range(NUM_JOINTS):
        r, c = divmod(j, 3)
        ax = axes[r][c]
        ax.plot(t, replay_tgt[:, j], color="gray",      lw=0.8, ls="--", alpha=0.6, label="target")
        ax.plot(t, replay_pos[:, j], color="tab:blue",  lw=1.0, alpha=0.85, label="measured")
        ax.plot(t, err[:, j],        color="tab:red",   lw=0.8, alpha=0.75, label="error")
        ax.axhline(0, color="black", lw=0.4, ls=":")
        ax.set_title(joint_names[j], fontsize=8, pad=2)
        ax.set_xlabel("time (s)", fontsize=7)
        ax.set_ylabel("position (rad)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right")
        _rmse_label(ax, rmse[j], "rad")

    _finish_fig(fig, out_path)


# ---------------------------------------------------------------------------
# Sim forward model — AX18AActuator (compliance model, from ax18a.py)
# ---------------------------------------------------------------------------

def simulate_ax18a_actuator(
    targets: np.ndarray,          # (T, 12) commanded target positions [rad]
    q0:      np.ndarray,          # (12,)   initial joint positions    [rad]
    v0:      np.ndarray,          # (12,)   initial joint velocities   [rad/s]
    inertia: np.ndarray,          # (12,)   per-joint CRB inertia (no armature) [kg·m²]
    effort_limit:          float = 1.8,
    stall_torque:          float = 1.8,
    velocity_limit:        float = 10.16,
    damping:               float = 0.035,          # from dextra_amp_env_cfg.py
    armature:              float = 0.00054,
    coulomb:               float = 0.04,
    viscous:               float = 0.0,
    compliance_margin_rad: float = 1.0  * _AX18A_RAD_PER_STEP,   # 1 step ≈0.00506 rad
    compliance_slope_rad:  float = 64.0 * _AX18A_RAD_PER_STEP,   # 64 steps≈0.3239 rad
    punch:                 float = 32.0,
    vel_eps:               float = 0.01,
    ctrl_dt:               float = 1.0 / 30.0,
    substeps:              int   = 40,     # 40 sub-steps/ctrl-step keeps Δv < v_limit per step
) -> np.ndarray:                  # (T, 12) simulated joint positions [rad]
    """Forward-simulate AX18AActuator dynamics (compliance model).

    Implements the same torque pipeline as ``AX18AActuator.compute()``
    in ``ax18a.py``, then integrates J·q̈ = τ_out via semi-implicit Euler.

    Pipeline (per sub-step)::

        1. Compliance torque: dead-zone → 0; slope zone → punch + k_eff·(|e|−margin)·sign(e);
                              saturated  → effort_limit·sign(e)
        2. Back-EMF damping:  τ -= damping × v
        3. Directional torque-speed sat. (4-quadrant DC motor curve):
              driving torque  (v in τ direction) is limited by the motor curve;
              braking torque  (v opposite τ)     is always available ≤ stall_torque.
        4. Coulomb friction:  reduces |τ| only, never flips direction

    Default values match ``dextra_amp_env_cfg.py`` AX18AActuatorCfg:
      compliance_slope=64 steps (≈0.324 rad, k_eff≈5.4 N·m/rad),
      damping=0.035, effort_limit=1.8, armature=5.4e-4, coulomb=0.04.
    """
    J_total      = np.asarray(inertia, dtype=np.float64) + armature
    punch_torque = (punch / 1023.0) * effort_limit
    dt = ctrl_dt / substeps
    T  = targets.shape[0]
    q  = q0.astype(np.float64).copy()
    v  = v0.astype(np.float64).copy()
    out = np.empty((T, NUM_JOINTS), dtype=np.float32)

    for step in range(T):
        q_tgt = targets[step].astype(np.float64)
        for _ in range(substeps):
            pos_error = q_tgt - q
            abs_error = np.abs(pos_error)

            # 1. Compliance torque
            in_dead_zone = abs_error < compliance_margin_rad
            torque_ratio = np.clip(
                (abs_error - compliance_margin_rad) / compliance_slope_rad,
                0.0, 1.0,
            )
            tau_mag = np.where(
                in_dead_zone, 0.0,
                punch_torque + (effort_limit - punch_torque) * torque_ratio,
            )
            tau_p = tau_mag * np.sign(pos_error)

            # 2. Back-EMF damping
            tau_d = tau_p - damping * v

            # 3. Directional torque-speed saturation (4-quadrant DC motor curve).
            #
            #    Bug in naive symmetric clip: when |v| >= velocity_limit,
            #    tau_max = 0 and np.clip(tau, -0, 0) = 0 for BOTH driving AND
            #    braking torques, causing the joint to coast at constant
            #    velocity indefinitely.
            #
            #    Fix: only restrict the DRIVING direction (torque that accelerates
            #    the joint further).  Braking torque (opposing motion) is always
            #    available up to stall_torque, providing the necessary deceleration.
            sign_tau      = np.sign(tau_d)
            v_in_dir      = sign_tau * v              # + = driving, − = braking
            vel_frac_dir  = np.clip(v_in_dir / velocity_limit, 0.0, 1.0)
            tau_drive_max = stall_torque * (1.0 - vel_frac_dir)   # ≥ 0 always
            tau_d = sign_tau * np.minimum(np.abs(tau_d), np.maximum(0.0, tau_drive_max))
            tau_d = np.clip(tau_d, -stall_torque, stall_torque)   # cap braking too

            # 4. Coulomb + viscous friction (magnitude reduction only)
            friction = coulomb * np.tanh(v / vel_eps) + viscous * v
            tau_sign = np.sign(tau_d)
            tau_out  = tau_sign * np.maximum(0.0, np.abs(tau_d) - np.abs(friction))

            # 5. Semi-implicit Euler
            acc = tau_out / J_total
            v  += acc * dt
            q  += v   * dt

        out[step] = q

    return out


# ---------------------------------------------------------------------------
# Plot 4 — sim expected positions vs hardware measured positions
# ---------------------------------------------------------------------------

def plot_sim_vs_hw(
    hw_pos:   np.ndarray,          # (T, 12) hardware measured positions [rad]
    sim_pos:  np.ndarray,          # (T, 12) simulated positions         [rad]
    targets:  np.ndarray,          # (T, 12) commanded targets           [rad]
    freq:     float,
    joint_names:            list[str],
    compliance_slope_steps: float,  # AX-18A register steps (e.g. 64)
    damping:                float,
    out_path:               str,
) -> None:
    T = len(hw_pos)
    t = np.arange(T) / freq

    rmse = np.sqrt(np.mean((hw_pos - sim_pos) ** 2, axis=0))
    bias = (hw_pos - sim_pos).mean(axis=0)

    slope_rad = compliance_slope_steps * _AX18A_RAD_PER_STEP
    punch_torque = (32.0 / 1023.0) * 1.8
    k_eff = (1.8 - punch_torque) / slope_rad if slope_rad > 0 else float("inf")

    print(f"\n[analyze] Sim-expected (AX18A) vs hardware  "
          f"(slope={compliance_slope_steps:.0f} steps / {slope_rad:.4f} rad, "
          f"k_eff≈{k_eff:.2f} N·m/rad, d={damping}):")
    print(f"  {'joint':<24}  {'RMSE (rad)':>10}  {'mean bias (rad)':>16}")
    for name, r, b in zip(joint_names, rmse, bias):
        print(f"  {name:<24}  {r:>10.4f}  {b:>16.4f}")
    print(f"  {'MEAN':<24}  {rmse.mean():>10.4f}  {bias.mean():>16.4f}")

    fig, axes = _make_fig(
        f"Sim Expected / AX18A (orange dashed) vs Hardware Measured (blue)  "
        f"[slope={compliance_slope_steps:.0f} steps, k_eff≈{k_eff:.1f} N·m/rad, d={damping}]"
    )
    for j in range(NUM_JOINTS):
        r, c = divmod(j, 3)
        ax = axes[r][c]
        ax.plot(t, targets[:, j], color="gray",       lw=0.7, ls="--", alpha=0.5, label="target")
        ax.plot(t, hw_pos[:,  j], color="tab:blue",   lw=1.2, alpha=0.85, label="hw measured")
        ax.plot(t, sim_pos[:, j], color="tab:orange", lw=1.2, ls="--", alpha=0.85, label="sim expected")
        ax.set_title(joint_names[j], fontsize=8, pad=2)
        ax.set_xlabel("time (s)", fontsize=7)
        ax.set_ylabel("position (rad)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right")
        _rmse_label(ax, rmse[j], "rad")

    _finish_fig(fig, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sim-to-real discrepancy analysis and joint position visualization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sim_npz",     type=str, default=None,
                        help="Sim action log .npz (from play_teacher_with_estimator.py)")
    parser.add_argument("--deploy_csv",  type=str, default=None,
                        help="Hardware deploy log CSV (from deploy.py --log)")
    parser.add_argument("--replay_csv",  type=str, default=None,
                        help="Hardware replay log CSV (from replay_actions.py --log)")
    parser.add_argument("--config",      type=str, default="config.yaml",
                        help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--out_dir",     type=str, default="logs/analysis",
                        help="Output directory for plots (default: logs/analysis)")
    parser.add_argument("--damping",          type=float, default=0.035,
                        help="AX18A back-EMF damping [N·m·s/rad] (default: 0.035 — from env cfg)")
    parser.add_argument("--compliance_slope", type=float, default=64.0,
                        help="AX18A compliance slope [register steps] (default: 64.0 — from env cfg)")
    parser.add_argument("--effort_limit",     type=float, default=1.8,
                        help="AX18A Torque Limit register [N·m] (default: 1.8)")
    parser.add_argument("--substeps",         type=int, default=40,
                        help="Physics sub-steps per control step (default: 40 — "
                             "keeps Δv < velocity_limit per sub-step for all joints)")
    args = parser.parse_args()

    if not any([args.sim_npz, args.deploy_csv, args.replay_csv]):
        parser.error("Provide at least one of --sim_npz, --deploy_csv, --replay_csv.")

    # --- Load config ---
    cfg = yaml.safe_load(open(args.config))
    freq = float(cfg["control"]["frequency_hz"])
    joint_names = [j["name"].replace("_Joint", "") for j in cfg["joints"]]

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load data ---
    sim_actions  = None
    deploy_data  = None
    replay_data  = None

    if args.sim_npz:
        sim_actions = load_npz_actions(args.sim_npz)
        T = sim_actions.shape[0]
        print(f"[analyze] Sim npz:    {T} steps  ({T/freq:.1f}s @ {freq:.0f}Hz)  "
              f"action range [{sim_actions.min():.3f}, {sim_actions.max():.3f}]")

    if args.deploy_csv:
        deploy_data = load_csv(args.deploy_csv)
        T = len(deploy_data["step"])
        print(f"[analyze] Deploy CSV: {T} steps  ({T/freq:.1f}s)")

    if args.replay_csv:
        replay_data = load_csv(args.replay_csv)
        T = len(replay_data["step"])
        print(f"[analyze] Replay CSV: {T} steps  ({T/freq:.1f}s)")

    # ------------------------------------------------------------------
    # Plot 1 — action comparison
    # ------------------------------------------------------------------
    if sim_actions is not None and deploy_data is not None:
        hw_actions = extract_cols(deploy_data, "action")
        plot_action_comparison(
            sim_actions, hw_actions, freq, joint_names,
            os.path.join(args.out_dir, "action_comparison.png"),
        )
    elif sim_actions is not None and deploy_data is None:
        print("[analyze] Skipping action comparison: --deploy_csv not provided.")
    elif sim_actions is None and deploy_data is not None:
        print("[analyze] Skipping action comparison: --sim_npz not provided.")

    # ------------------------------------------------------------------
    # Plot 2 — joint positions: replay vs deploy
    # ------------------------------------------------------------------
    if replay_data is not None and deploy_data is not None:
        replay_tgt = extract_cols(replay_data,  "target")
        replay_pos = extract_cols(replay_data,  "pos")
        deploy_pos = extract_cols(deploy_data,  "pos")
        plot_joint_positions(
            replay_tgt, replay_pos, deploy_pos, freq, joint_names,
            os.path.join(args.out_dir, "joint_positions.png"),
        )

    # ------------------------------------------------------------------
    # Plot 3 — servo tracking error (replay only)
    # ------------------------------------------------------------------
    if replay_data is not None:
        replay_tgt = extract_cols(replay_data, "target")
        replay_pos = extract_cols(replay_data, "pos")
        plot_tracking_error(
            replay_tgt, replay_pos, freq, joint_names,
            os.path.join(args.out_dir, "tracking_error.png"),
        )

    # ------------------------------------------------------------------
    # Plot 4 — sim-expected positions vs hardware measured positions
    # ------------------------------------------------------------------
    if deploy_data is not None:
        hw_pos  = extract_cols(deploy_data, "pos")
        hw_vel  = extract_cols(deploy_data, "vel")
        targets = extract_cols(deploy_data, "target")
        slope_rad = args.compliance_slope * _AX18A_RAD_PER_STEP
        punch_torque = (32.0 / 1023.0) * args.effort_limit
        k_eff = (args.effort_limit - punch_torque) / slope_rad if slope_rad > 0 else float("inf")
        print(f"\n[analyze] Forward simulating AX18AActuator "
              f"(slope={args.compliance_slope:.0f} steps / {slope_rad:.4f} rad, "
              f"k_eff≈{k_eff:.2f} N·m/rad, d={args.damping}, substeps={args.substeps})...")
        sim_pos = simulate_ax18a_actuator(
            targets=targets,
            q0=hw_pos[0],
            v0=hw_vel[0],
            inertia=SYSID_INERTIA_KGM2,
            effort_limit=args.effort_limit,
            damping=args.damping,
            compliance_slope_rad=slope_rad,
            ctrl_dt=1.0 / freq,
            substeps=args.substeps,
        )
        plot_sim_vs_hw(
            hw_pos=hw_pos,
            sim_pos=sim_pos,
            targets=targets,
            freq=freq,
            joint_names=joint_names,
            compliance_slope_steps=args.compliance_slope,
            damping=args.damping,
            out_path=os.path.join(args.out_dir, "sim_expected_vs_hw.png"),
        )

    print(f"\n[analyze] Done. Plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
