"""ONNX Runtime wrapper for the LSTM State Estimator with history buffer.

The ONNX model already has normalization baked in:
  input(1, window, 24) → InputNorm → LSTM → FC → OutputDenorm → output(1, 19)

This wrapper manages the sliding-window history buffer in numpy,
matching the play_teacher_with_estimator.py pattern (torch.roll → FIFO).
"""

import os

import numpy as np
import onnxruntime as ort


class LSTMEstimatorONNX:
    """LSTM state estimator with sliding-window history buffer."""

    def __init__(self, onnx_path: str, window: int = 50, encoder_dim: int = 24):
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        self.window = window
        self.encoder_dim = encoder_dim

        # History buffer: (1, window, encoder_dim) — zero-initialized
        self.history = np.zeros((1, window, encoder_dim), dtype=np.float32)
        self.valid_count = 0

        # Auto-load init history from models/lstm_init_history.npy (generated
        # by export_to_onnx.py).  If found, the LSTM starts with synthetic
        # zero-pose rollout context instead of a cold-start all-zero buffer.
        self._init_history: np.ndarray | None = None
        init_path = os.path.join(
            os.path.dirname(os.path.abspath(onnx_path)), "lstm_init_history.npy"
        )
        if os.path.exists(init_path):
            init_h = np.load(init_path).astype(np.float32)
            if init_h.shape == (window, encoder_dim):
                self._init_history = init_h
                self.history[0] = init_h
                self.valid_count = window
                print(f"[estimator] LSTM init history loaded: {init_path}")
            else:
                print(
                    f"[estimator] WARNING: lstm_init_history.npy shape "
                    f"{init_h.shape} != ({window}, {encoder_dim}), ignored"
                )

    def update_and_predict(self, encoder_obs: np.ndarray) -> np.ndarray:
        """Update history buffer and run LSTM estimator.

        Args:
            encoder_obs: Encoder observation of shape (24,) — [joint_pos, joint_vel].

        Returns:
            Privileged state estimate of shape (19,).
        """
        # FIFO shift left, append new observation at the end
        self.history = np.roll(self.history, -1, axis=1)
        self.history[0, -1, :] = encoder_obs.astype(np.float32)
        self.valid_count += 1

        # Run ONNX inference
        result = self.session.run(None, {self.input_name: self.history})
        return result[0].flatten()

    def reset(self):
        """Reset history buffer to init state (call on episode start/robot restart)."""
        if self._init_history is not None:
            self.history[0] = self._init_history
            self.valid_count = self.window
        else:
            self.history[:] = 0.0
            self.valid_count = 0
