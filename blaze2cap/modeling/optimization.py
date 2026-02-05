# -*- coding: utf-8 -*-
# @Time    : 2/5/26
# @Project : Real-Time Motion Prediction
# @File    : optimization.py

"""
PyTorch optimization configuration.
Replaces legacy BertAdam with standard AdamW and Decoupled Schedulers.
"""

__all__ = [
    "optimizer_cfg",
    "scheduler_cfg"
]

import logging
import torch.optim as optim
from hydra_zen import builds

logger = logging.getLogger(__name__)

# --- 1. Optimizer Configuration ---
# Use AdamW (Adam with decoupled Weight Decay).
# This is the standard optimizer for Transformers (BERT, GPT, ViT, etc.)
# It fixes the L2 regularization issues present in the original Adam.
optimizer_cfg = builds(
    optim.AdamW, 
    populate_full_signature=True
)

# --- 2. Scheduler Configuration ---
# Use OneCycleLR.
# This scheduler warms up the learning rate for the first few steps (warmup)
# and then anneals it, which is crucial for preventing early divergence
# in Motion Transformers.
scheduler_cfg = builds(
    optim.lr_scheduler.OneCycleLR, 
    populate_full_signature=True
)


# USE IN TRAINING CODE 

# # 1. Initialize Optimizer
# # (The 'builds' config from optimization.py creates the class, you just pass params)
# optimizer = optim.AdamW(
#     model.parameters(), 
#     lr=1e-4, 
#     weight_decay=0.01,
#     betas=(0.9, 0.999)
# )

# # 2. Initialize Scheduler
# # You must pass the optimizer to the scheduler
# scheduler = optim.lr_scheduler.OneCycleLR(
#     optimizer,
#     max_lr=1e-4,
#     total_steps=num_epochs * len(train_loader), # Total training steps
#     pct_start=0.1 # 10% Warmup (Matches your old 'warmup_linear' logic)
# )

# # 3. Step in loop
# # loss.backward()
# # optimizer.step()
# # scheduler.step() # Step scheduler every batch