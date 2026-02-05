# -*- coding: utf-8 -*-
# @Time    : 2/5/26
# @Project : Real-Time Motion Prediction
# @File    : train_utils.py

import random
import os
import time
import datetime
import logging
from collections import defaultdict
import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ==============================================================================
#  0. TIMER CLASS (Profiling)
# ==============================================================================
class Timer:
    """
    Simple timer for profiling training steps.
    Usage:
        timer = Timer()
        timer.tick("data_load")
        # ... do data loading ...
        timer.tick("forward")
        # ... do forward pass ...
        print(timer.report())
    """
    def __init__(self):
        self.times = defaultdict(float)
        self.counts = defaultdict(int)
        self._last_tick = time.perf_counter()
        self._last_name = None
    
    def tick(self, name: str):
        """Record time since last tick and assign to given name."""
        now = time.perf_counter()
        if self._last_name is not None:
            elapsed = now - self._last_tick
            self.times[self._last_name] += elapsed
            self.counts[self._last_name] += 1
        self._last_tick = now
        self._last_name = name
    
    def reset(self):
        """Reset all recorded times."""
        self.times.clear()
        self.counts.clear()
        self._last_tick = time.perf_counter()
        self._last_name = None
    
    def report(self) -> str:
        """Return formatted report of all recorded times."""
        lines = []
        total = sum(self.times.values())
        for name, t in self.times.items():
            count = self.counts[name]
            avg = t / count if count > 0 else 0
            pct = (t / total * 100) if total > 0 else 0
            lines.append(f"  {name}: {t:.3f}s total ({count}x, {avg*1000:.1f}ms avg, {pct:.1f}%)")
        return "\n".join(lines) if lines else "  (no timing data)"

# ==============================================================================
#  1. DATA PREFETCHER (Accelerates Training)
# ==============================================================================
class CudaPreFetcher:
    """
    Accelerates data loading by moving the next batch to GPU 
    in a background stream while the current batch is processing.
    """
    def __init__(self, data_loader, device):
        self._loader_len = len(data_loader)  # Store length before iter()
        self.loader = iter(data_loader)
        self.stream = torch.cuda.Stream()
        self.device = device
        self.batch = None
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
            
        with torch.cuda.stream(self.stream):
            self.batch = self._cuda(self.batch)

    def _cuda(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device, non_blocking=True)
        elif isinstance(x, dict):
            return {k: self._cuda(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [self._cuda(i) for i in x]
        return x

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

    def __iter__(self):
        return self
    
    def __len__(self):
        # Return the original DataLoader's length
        # We store a reference to get the correct batch count
        return getattr(self, '_loader_len', 0)


# ==============================================================================
#  2. REPRODUCIBILITY
# ==============================================================================
def set_random_seed(seed: int, deterministic: bool = True):
    """
    Sets seeds for all random number generators to ensure reproducible results.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.debug(f"Manual seed set to {seed} (Deterministic).")
        else:
            logger.debug(f"Manual seed set to {seed} (Benchmark Mode).")
    else:
        logger.warning("No manual seed set. Training will be non-deterministic.")


# ==============================================================================
#  3. DISTRIBUTED HELPERS (DDP)
# ==============================================================================
def get_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

def init_distributed_mode(args):
    """
    Initialize DDP (Distributed Data Parallel) if available.
    Expects 'args' object with 'rank', 'world_size', 'gpu'.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif hasattr(args, 'rank'):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    
    torch.distributed.init_process_group(
        backend=args.dist_backend, 
        init_method=args.dist_url,
        world_size=args.world_size, 
        rank=args.rank
    )
    torch.distributed.barrier()