"""Policy action to joint target conversion shared by deploy tools."""

from __future__ import annotations

import numpy as np


def joint_limits_from_config(joints: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return lower/upper joint limits in config action order."""
    joint_lower = np.array([joint["lower_rad"] for joint in joints], dtype=np.float32)
    joint_upper = np.array([joint["upper_rad"] for joint in joints], dtype=np.float32)
    return joint_lower, joint_upper


def action_signs_from_config(cfg: dict) -> np.ndarray:
        """Return per-joint action signs from config in joint order.

        Supported config key:
            control.action_sign_by_joint_name: {joint_name: +/-1}

        Missing key defaults to all +1 (no sign inversion).
        """
        joints = cfg["joints"]
        sign_map = cfg.get("control", {}).get("action_sign_by_joint_name", {})
        signs = []
        for joint in joints:
                value = sign_map.get(joint["name"], 1)
                signs.append(-1.0 if float(value) < 0.0 else 1.0)
        return np.array(signs, dtype=np.float32)


def actions_to_joint_targets(
    actions: np.ndarray,
    action_scale: float,
    action_offset: float,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    action_signs: np.ndarray | None = None,
) -> np.ndarray:
    """Match deploy.py action clipping, scaling, offset, and joint-limit clipping."""
    action_array = np.asarray(actions, dtype=np.float32)
    action_clipped = np.clip(action_array, -1.0, 1.0)
    if action_signs is not None:
        signs = np.asarray(action_signs, dtype=np.float32)
        if signs.shape != action_clipped.shape:
            raise ValueError("action_signs shape must match actions shape")
        action_clipped = action_clipped * signs
    targets = action_offset + action_scale * action_clipped
    return np.clip(targets, joint_lower, joint_upper).astype(np.float32)


def joint_targets_to_actions(
    targets: np.ndarray,
    action_scale: float,
    action_offset: float,
    action_signs: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the policy action values that request absolute joint targets."""
    if action_scale == 0.0:
        raise ValueError("action_scale must be non-zero")
    target_array = np.asarray(targets, dtype=np.float32)
    actions = ((target_array - action_offset) / action_scale).astype(np.float32)
    if action_signs is not None:
        signs = np.asarray(action_signs, dtype=np.float32)
        if signs.shape != actions.shape:
            raise ValueError("action_signs shape must match targets shape")
        actions = actions * signs
    return actions
