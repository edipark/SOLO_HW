"""CSV data logger for deployment diagnostics."""

import csv
import os
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Replay logger
# ---------------------------------------------------------------------------

class ReplayLogger:
    """Logs per-step replay data (commanded targets + measured positions) to CSV.

    Schema: step, timestamp, loop_dt_ms, target_0..N-1, pos_0..N-1
    """

    def __init__(self, log_dir: str = "logs", num_joints: int = 12):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(log_dir, f"replay_{timestamp}.csv")
        self.num_joints = num_joints

        header = ["step", "timestamp", "loop_dt_ms"]
        header += [f"target_{i}" for i in range(num_joints)]
        header += [f"pos_{i}" for i in range(num_joints)]

        self._file = open(self.filepath, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(header)
        self._step = 0

    def log(self, loop_dt_ms: float, targets, positions):
        """Write one row: targets and measured positions for this step."""
        row = [self._step, time.time(), f"{loop_dt_ms:.2f}"]
        row += [f"{v:.6f}" for v in targets]
        row += [f"{v:.6f}" for v in positions]
        self._writer.writerow(row)
        self._step += 1

    def close(self):
        self._file.close()
        print(f"[log] Saved {self._step} steps to {self.filepath}")


# ---------------------------------------------------------------------------
# Deploy logger
# ---------------------------------------------------------------------------



class CSVLogger:
    """Logs per-step deployment data to a CSV file.

    Each row contains: timestamp, loop time, joint positions, joint velocities,
    privileged estimate, raw actions, and position targets.
    """

    def __init__(self, log_dir: str = "logs", num_joints: int = 12, priv_dim: int = 19):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(log_dir, f"deploy_{timestamp}.csv")
        self.num_joints = num_joints
        self.priv_dim = priv_dim

        # Build header
        header = ["step", "timestamp", "loop_dt_ms"]
        header += [f"pos_{i}" for i in range(num_joints)]
        header += [f"vel_{i}" for i in range(num_joints)]
        header += [f"priv_{i}" for i in range(priv_dim)]
        header += [f"action_{i}" for i in range(num_joints)]
        header += [f"target_{i}" for i in range(num_joints)]
        self.header = header

        self._file = open(self.filepath, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(header)
        self._step = 0

    def log(self, loop_dt_ms: float, positions, velocities,
            priv_est, actions, targets):
        """Write one row of deployment data."""
        row = [self._step, time.time(), f"{loop_dt_ms:.2f}"]
        row += [f"{v:.6f}" for v in positions]
        row += [f"{v:.6f}" for v in velocities]
        row += [f"{v:.6f}" for v in priv_est]
        row += [f"{v:.6f}" for v in actions]
        row += [f"{v:.6f}" for v in targets]
        self._writer.writerow(row)
        self._step += 1

    def close(self):
        """Flush and close the log file."""
        self._file.close()
        print(f"[log] Saved {self._step} steps to {self.filepath}")
