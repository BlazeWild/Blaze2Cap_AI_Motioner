import functools
import logging
import time
from collections import defaultdict, deque
import numpy as np
import torch

# Optional: pip install tabulate
try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

logger = logging.getLogger(__name__)

def format_time(s: float) -> str:
    """Return a nice string representation of `s` seconds."""
    m = int(s / 60)
    s -= m * 60
    h = int(m / 60)
    m -= h * 60
    ms = int((s - int(s)) * 1000)
    s = int(s)
    
    parts = []
    if h > 0: parts.append(f"{h}h")
    if m > 0: parts.append(f"{m}m")
    if s > 0: parts.append(f"{s}s")
    if ms > 0 and s == 0: parts.append(f"{ms}ms")
    return "".join(parts) if parts else "0s"

class Timer(object):
    """
    A simple timer to profile code execution blocks.
    Usage:
        timer = Timer()
        timer("data_loading")
        # ... code ...
        timer("model_forward")
        print(timer.print())
    """
    def __init__(self, msg="", synchronize=False, history_size=1000, precision=3):
        self.msg = msg
        self.synchronize = synchronize
        self.precision = precision
        self.time_history = defaultdict(functools.partial(deque, maxlen=history_size))
        
        self.start = self.get_time()
        self.last_checkpoint = self.start

    def get_time(self):
        if self.synchronize and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()

    def reset(self):
        self.last_checkpoint = self.get_time()

    def __call__(self, stage_name: str):
        """Records time elapsed since last call."""
        current_time = self.get_time()
        duration = current_time - self.last_checkpoint
        self.last_checkpoint = current_time
        
        self.time_history[stage_name].append(duration)
        return duration

    def print(self):
        if tabulate:
            # Prepare data for table
            data = []
            for k, v in self.time_history.items():
                mean_time = np.mean(v)
                total_time = np.sum(v)
                data.append([k, format_time(mean_time), f"{mean_time*1000:.2f} ms"])
            return tabulate(data, headers=["Stage", "Avg Time", "Raw (ms)"], tablefmt="simple")
        else:
            # Fallback if tabulate is not installed
            lines = ["Profile Results:"]
            for k, v in self.time_history.items():
                lines.append(f"  {k}: {np.mean(v)*1000:.2f} ms")
            return "\n".join(lines)