"""


Provides logging utilities for training RL agents. Logs scalar metrics
(e.g., reward, loss) to both CSV and TensorBoard. Buffers values and
writes averaged scalars every `flush_interval` steps for cleaner logs.
"""

import csv
import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, agent_name, log_dir="logs", tb_dir="tb", flush_interval=1000):
        """
        Initializes a new logging session.

        Args:
            agent_name (str): Name of the agent (used for folder structure).
            log_dir (str): Directory to save CSV logs.
            tb_dir (str): Directory to save TensorBoard logs.
            flush_interval (int): Number of steps between averaging and writing scalars.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, agent_name, f"run_{timestamp}")
        self.tb_path = os.path.join(tb_dir, agent_name, f"run_{timestamp}")

        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.tb_path, exist_ok=True)

        self.scalar_file = open(
            os.path.join(self.log_path, "scalars.csv"), "w", newline=""
        )
        self.scalar_writer = csv.writer(self.scalar_file)
        self.writer = SummaryWriter(self.tb_path)
        self.scalar_buffer = {}  # {tag: [(step, value)]}
        self.flush_interval = flush_interval

    def log_scalar(self, tag, value, step):
        """
        Buffers scalar values and logs the average every flush_interval steps.

        Args:
            tag (str): Metric name (e.g., 'loss').
            value (float): Metric value.
            step (int): Current global training step.
        """
        if tag not in self.scalar_buffer:
            self.scalar_buffer[tag] = []
        self.scalar_buffer[tag].append((step, value))

        if len(self.scalar_buffer[tag]) >= self.flush_interval:
            avg = sum(v for _, v in self.scalar_buffer[tag]) / len(
                self.scalar_buffer[tag]
            )
            last_step = self.scalar_buffer[tag][-1][0]
            self.scalar_writer.writerow([last_step, tag, avg])
            self.writer.add_scalar(tag, avg, last_step)
            self.scalar_buffer[tag].clear()

    def log_config(self, config_dict):
        """
        Logs a dictionary of training config parameters as hyperparameters in TensorBoard.

        Args:
            config_dict (dict): Dictionary of hyperparameters.
        """
        self.writer.add_hparams(config_dict, {})

    def close(self):
        """
        Flushes remaining scalar values and closes file handles.
        """
        for tag, values in self.scalar_buffer.items():
            if values:
                avg = sum(v for _, v in values) / len(values)
                last_step = values[-1][0]
                self.scalar_writer.writerow([last_step, tag, avg])
                self.writer.add_scalar(tag, avg, last_step)
        self.scalar_file.close()
        self.writer.close()

    def flush(self):
        """
        Force-flushes current scalar buffer to disk without closing.
        """
        for tag, values in self.scalar_buffer.items():
            if values:
                avg = sum(v for _, v in values) / len(values)
                last_step = values[-1][0]
                self.scalar_writer.writerow([last_step, tag, avg])
                self.writer.add_scalar(tag, avg, last_step)
                self.scalar_buffer[tag].clear()
        self.scalar_file.flush()
