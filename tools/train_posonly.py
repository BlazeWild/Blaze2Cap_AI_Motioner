"""
Blaze2Cap Training Script (Canonical Position Version)
======================================================
Train the MotionTransformer model to predict 3D skeletal motion from BlazePose landmarks.
Now uses a direct Position-to-Position architecture with Canonical Alignment.

Usage:
    python -m tools.train
"""

import os
import sys
import functools
import logging
from collections import defaultdict

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torchinfo import summary
from tqdm import tqdm

# --- Blaze2Cap Modules ---
from blaze2cap.modules.models_posonly import MotionTransformer
from blaze2cap.modules.data_loader_posonly import PoseSequenceDataset
from blaze2cap.modules.loss_posonly import MotionLoss
from blaze2cap.utils.logging_ import setup_logging
from blaze2cap.utils.train_utils import set_random_seed, Timer

# --- OPTIMIZATION (RTX 4090) ---
torch.set_float32_matmul_precision('high') 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True 

# --- CONFIGURATION ---
CONFIG = {
    "experiment_name": "canonical_motion_transformer",
    "data_root": "./blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset",
    "save_dir": "./checkpoints",
    "log_dir": "./logs",
    
    # Model Hyperparameters
    "num_joints_in": 19,    # BlazePose Reduced
    "input_feats": 8,       # [wx,wy,wz, vx,vy,vz, vis, anc]
    "num_joints_out": 20,   # TotalCapture Body (2-21)
    
    "d_model": 512,
    "num_layers": 6,
    "n_head": 8,
    "d_ff": 1024,
    "dropout": 0.1,
    "max_len": 512,
    
    # Training
    "batch_size": 32,
    "num_workers": 6,
    "max_windows_train": 64,  # Subsample for speed
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 150,
    "window_size": 64,
    "warmup_pct": 0.1,
    
    # Loss Weights
    "lambda_pos": 100.0,      # Main MPJPE driver
    "lambda_vel": 10.0,       # Smoothness
    "lambda_smooth": 1.0,     # Jitter reduction
    "lambda_contact": 0.5,    # Anti-slide
    
    # System
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_amp": True,
    "gradient_clip": 1.0,
}

# Fix fragmentation issues
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler, device, epoch):
    """Train for one epoch with mixed precision support."""
    model.train()
    timer = Timer()
    stats = defaultdict(float)
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [TRAIN]")
    
    for batch_idx, batch in enumerate(pbar):
        timer.tick("data_load")
        
        # 1. Move to Device
        src = batch["source"].to(device, non_blocking=True) # (B, 64, 19, 8)
        tgt_rot = batch["target"].to(device, non_blocking=True) # (B, 64, 22, 6)
        
        # 2. Forward Pass
        optimizer.zero_grad(set_to_none=True)
        
        with autocast('cuda', enabled=CONFIG["use_amp"]):
            # Pred: (B, 64, 20, 3) -> Positions
            preds = model(src)
            timer.tick("forward")
            
            # 3. Compute Loss
            # Loss function handles GT Rotation -> GT Position conversion internally
            loss, loss_logs = criterion(preds, tgt_rot)
            
            if not torch.isfinite(loss):
                print(f"⚠️ NaN/Inf Loss at Step {batch_idx}. Skipping.")
                continue

        # 4. Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        timer.tick("backward")
        
        # 5. Logging
        stats["loss"] += loss.item()
        for k, v in loss_logs.items():
            stats[k] += v
            
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}", 
            "MPJPE": f"{loss_logs['l_mpjpe']:.4f}"
        })

    # Averages
    N = len(loader)
    avg_stats = {k: v / N for k, v in stats.items()}
    return avg_stats

def validate(model, loader, criterion, device, epoch):
    """Validate MPJPE in Millimeters."""
    model.eval()
    pbar = tqdm(loader, desc=f"Epoch {epoch} [VAL]")
    
    mpjpe_sum = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in pbar:
            src = batch["source"].to(device, non_blocking=True)
            tgt_rot = batch["target"].to(device, non_blocking=True)
            
            # Forward
            preds = model(src) # (B, S, 20, 3)
            
            # Get Ground Truth Positions using the Loss function's helper
            # This ensures we compare apples to apples (Canonical Pos vs Canonical Pos)
            tgt_pos = criterion._compute_target_positions(tgt_rot)
            
            # Compute MPJPE (Euclidean Dist)
            diff = preds - tgt_pos
            dist = torch.norm(diff, dim=-1) # (B, S, 20)
            
            # Mean over batch/seq/joints
            batch_mpjpe = dist.mean().item()
            
            B = src.shape[0]
            mpjpe_sum += batch_mpjpe * B
            count += B
            
    avg_mpjpe_m = mpjpe_sum / max(count, 1)
    avg_mpjpe_mm = avg_mpjpe_m * 1000.0
    
    return avg_mpjpe_mm

def collate_flatten(batch):
    """Custom collate to handle variable length or just stack."""
    # Since we window in Dataset, everything should be fixed size (64)
    # Just standard stacking
    return {
        "source": torch.stack([b["source"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch])
    }

def main():
    # 1. Setup
    set_random_seed(CONFIG["seed"])
    logger = setup_logging(CONFIG["log_dir"], log_file="train.log")
    device = CONFIG["device"]
    
    logger.info(f"🚀 Starting Canonical Training on {device}")
    
    # 2. Data
    logger.info("Initializing Datasets...")
    # Train: Subsample for speed
    train_dataset = PoseSequenceDataset(
        CONFIG["data_root"], 
        CONFIG["window_size"], 
        split="train", 
        max_windows=CONFIG["max_windows_train"]
    )
    # Val: Use ALL windows for accurate metrics
    val_dataset = PoseSequenceDataset(
        CONFIG["data_root"], 
        CONFIG["window_size"], 
        split="val", 
        max_windows=None 
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        collate_fn=collate_flatten,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG["batch_size"], # Larger batch size for val is fine
        shuffle=False, 
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        collate_fn=collate_flatten
    )
    
    logger.info(f"Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")
    
    # 3. Model
    model = MotionTransformer(
        num_joints=CONFIG["num_joints_in"],
        input_feats=CONFIG["input_feats"],
        output_joints=CONFIG["num_joints_out"],
        d_model=CONFIG["d_model"],
        num_layers=CONFIG["num_layers"],
        n_head=CONFIG["n_head"],
        d_ff=CONFIG["d_ff"],
        dropout=CONFIG["dropout"]
    ).to(device)
    
    # Log Model Summary
    try:
        input_shape = (CONFIG['batch_size'], CONFIG['window_size'], CONFIG['num_joints_in'], CONFIG['input_feats'])
        logger.info(f"Model Summary:\n{summary(model, input_size=input_shape, device=device)}")
    except Exception as e:
        logger.warning(f"Could not print summary: {e}")

    # 4. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG["lr"] * 10,
        steps_per_epoch=len(train_loader),
        epochs=CONFIG["epochs"],
        pct_start=CONFIG["warmup_pct"]
    )
    
    # 5. Loss
    criterion = MotionLoss(
        lambda_pos=CONFIG["lambda_pos"],
        lambda_vel=CONFIG["lambda_vel"],
        lambda_smooth=CONFIG["lambda_smooth"],
        lambda_contact=CONFIG["lambda_contact"]
    ).to(device)
    
    scaler = GradScaler('cuda', enabled=CONFIG["use_amp"])
    
    # 6. Training Loop
    best_mpjpe = float("inf")
    
    for epoch in range(1, CONFIG["epochs"] + 1):
        logger.info(f"--- Epoch {epoch}/{CONFIG['epochs']} ---")
        
        # Train
        train_stats = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, device, epoch)
        
        logger.info(f"[TRAIN] Loss: {train_stats['loss']:.5f} | MPJPE_Proxy: {train_stats['l_mpjpe']:.5f} | Smooth: {train_stats['l_smooth']:.5f}")
        
        # Validate
        val_mpjpe_mm = validate(model, val_loader, criterion, device, epoch)
        logger.info(f"[VAL] MPJPE: {val_mpjpe_mm:.2f} mm")
        
        # Save Best
        if val_mpjpe_mm < best_mpjpe:
            best_mpjpe = val_mpjpe_mm
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mpjpe': best_mpjpe,
            }, os.path.join(CONFIG["save_dir"], "best_model.pth"))
            logger.info(f"⭐ New Best Model Saved! ({val_mpjpe_mm:.2f} mm)")
            
        # Periodic Save
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, os.path.join(CONFIG["save_dir"], f"checkpoint_epoch{epoch}.pth"))
            
    logger.info(f"Training Complete. Best MPJPE: {best_mpjpe:.2f} mm")

if __name__ == "__main__":
    # Create dirs
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)
    main()