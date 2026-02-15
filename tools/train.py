"""
Blaze2Cap Training Script (Canonical Rotation - 27 Joints)
==========================================================
State-of-the-Art Configuration for 6D Rotation Learning.
Includes advanced Physics-based constraints (Contact, Floor, Tilt).
"""

import os
import sys
import logging
import glob
import re
from collections import defaultdict
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torchinfo import summary
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# --- IMPORTS ---
from blaze2cap.modules.models import MotionTransformer
from blaze2cap.modules.data_loader import PoseSequenceDataset
from blaze2cap.modeling.loss import MotionLoss
from blaze2cap.utils.logging_ import setup_logging
from blaze2cap.utils.train_utils import set_random_seed

# NEW: Import the robust evaluator we just created
from blaze2cap.modeling.eval_motion import evaluate_motion

# --- OPTIMIZATION (RTX 4090) ---
torch.set_float32_matmul_precision('high') 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True 

# --- CONFIGURATION ---
CONFIG = {
    "experiment_name": "canonical_rotation_28joints_deep",
    "data_root": os.path.join(PROJECT_ROOT, "blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset"),
    "save_dir": os.path.join(PROJECT_ROOT, "checkpoints"),
    "log_dir": os.path.join(PROJECT_ROOT, "logs"),
    
    # Model Hyperparameters
    "num_joints_in": 28,    # 27 Input Joints
    "input_feats": 14,      # 19 Features (Pos, Vel, Par, Chi, Vis, Anc, Align, SVel, Scale)
    "num_joints_out": 21,   # 21 Output Joints (Root + Body)
    
    "d_model": 512,
    "num_layers": 6,        # Deep Model
    "n_head": 8,
    "d_ff": 1024,
    "dropout": 0.1,
    "max_len": 512,
    
    # Training
    "batch_size": 4,       # Reverted to 32 for stability (8 is too noisy)
    "accumulation_steps": 4,  # NEW: Effective Batch Size = 8 * 4 = 32
    "num_workers": 6,
    "max_windows_train": 64, 
    "lr": 1e-4,             
    "weight_decay": 1e-4,
    "epochs": 150,
    "window_size": 64,
    "warmup_pct": 0.1,
    
    # Loss Weights
    "lambda_root": 5.0,       # High priority on Trajectory
    "lambda_rot": 1.0,        # Pose Structure
    "lambda_pos": 2.0,        # FK Validity
    "lambda_vel": 2.5,        # Dynamics
    "lambda_smooth": 5,     # Anti-Jitter
    "lambda_contact": 2.0,    # Anti-Slide
    "lambda_floor": 2.0,      # Anti-Clipping
    "lambda_tilt": 1.0,       # Spine Stability
    
    # System
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_amp": True,
    "gradient_clip": 1.0,
}

# Note: expandable_segments not supported on Windows
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler, device, epoch):
    model.train()
    stats = defaultdict(float)
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [TRAIN]")
    
    # Reset gradients at start of epoch
    optimizer.zero_grad(set_to_none=True)
    
    for i, batch in enumerate(pbar):
        # Inputs
        src = batch["source"].to(device, non_blocking=True)
        tgt_rot = batch["target"].to(device, non_blocking=True)
        
        # Forward & Loss
        with autocast('cuda', enabled=CONFIG["use_amp"]):
            preds = model(src)
            loss, loss_logs = criterion(preds, tgt_rot)
            
            # Divide loss by accumulation steps
            loss = loss / CONFIG["accumulation_steps"]
            
            if not torch.isfinite(loss):
                continue

        # Backward
        scaler.scale(loss).backward()
        
        # Step Optimizer (Only every N batches)
        if (i + 1) % CONFIG["accumulation_steps"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"])
            
            # Check scale before step to detect skips
            scale_before = scaler.get_scale()
            
            scaler.step(optimizer)
            scaler.update()
            
            # FIX: Only step scheduler if scaler didn't skip the optimizer step
            # (If scale decreased, it means Infs were found and step was skipped)
            scale_after = scaler.get_scale()
            if scale_after >= scale_before:
                scheduler.step()
            
            # Reset gradients
            optimizer.zero_grad(set_to_none=True)
        
        # Logging
        current_loss = loss.item() * CONFIG["accumulation_steps"]
        stats["loss"] += current_loss
        for k, v in loss_logs.items():
            stats[k] += v
            
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({
            "Loss": f"{current_loss:.4f}", 
            "Root": f"{loss_logs.get('l_root', 0):.4f}",
            "RAcc": f"{loss_logs.get('l_root_acc', 0):.4f}", 
            "Pos": f"{loss_logs.get('l_mpjpe', 0):.4f}",
            "Slide": f"{loss_logs.get('l_contact', 0):.4f}",
            "LR": f"{current_lr:.2e}" 
        })

    N = len(loader)
    return {k: v / N for k, v in stats.items()}


def validate(model, loader, device, epoch):
    """
    Uses the external evaluate_motion function to compute:
    MPJPE (mm), Root Rot (deg), Slide (mm)
    """
    
    # Free memory before validation
    torch.cuda.empty_cache()
    
    model.eval()
    pbar = tqdm(loader, desc=f"Epoch {epoch} [VAL]")
    
    metrics_sum = defaultdict(float)
    count = 0
    
    with torch.no_grad():
        for batch in pbar:
            src = batch["source"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)
            
            # Forward
            preds = model(src)
            
            # Use the new robust evaluator
            batch_metrics = evaluate_motion(preds, tgt)
            
            B = src.shape[0]
            for k, v in batch_metrics.items():
                metrics_sum[k] += v * B
            count += B
            
            # Clean postfix (Removed RootPos)
            pbar.set_postfix({
                "MPJPE": f"{batch_metrics['MPJPE']:.1f}",
                "RotDeg": f"{batch_metrics.get('Root_Rot_Deg', 0):.1f}",
                "Slide": f"{batch_metrics.get('Foot_Slide', 0):.1f}"
            })
            
    avg_metrics = {k: v / max(count, 1) for k, v in metrics_sum.items()}
    return avg_metrics

def collate_flatten(batch):
    return {
        "source": torch.cat([b["source"] for b in batch], dim=0),
        "target": torch.cat([b["target"] for b in batch], dim=0)
    }

def main():
    set_random_seed(CONFIG["seed"])
    logger = setup_logging(CONFIG["log_dir"], log_file="train.log")
    device = CONFIG["device"]
    
    logger.info(f"[START] Deep Rotation Training (L={CONFIG['num_layers']}) on {device}")
    
    # 1. Data
    train_dataset = PoseSequenceDataset(CONFIG["data_root"], CONFIG["window_size"], "train", CONFIG["max_windows_train"])
    val_dataset = PoseSequenceDataset(CONFIG["data_root"], CONFIG["window_size"], "val", None)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, 
                              num_workers=CONFIG["num_workers"], pin_memory=True, collate_fn=collate_flatten, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, 
                            num_workers=CONFIG["num_workers"], pin_memory=True, collate_fn=collate_flatten)
    
    # 2. Model (Dimensions match new config)
    model = MotionTransformer(
        num_joints=CONFIG["num_joints_in"], 
        input_feats=CONFIG["input_feats"], 
        num_joints_out=CONFIG["num_joints_out"],
        d_model=CONFIG["d_model"], 
        num_layers=CONFIG["num_layers"], 
        n_head=CONFIG["n_head"], 
        d_ff=CONFIG["d_ff"], 
        dropout=CONFIG["dropout"]
    ).to(device)
    
    # Display Model Summary
    try:
        summary(model, input_size=(CONFIG["batch_size"], CONFIG["window_size"], CONFIG["input_feats"]))
    except Exception as e:
        logger.warning(f"[WARN] Could not print model summary: {e}")
    
    # 3. Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    
    criterion = MotionLoss(
        lambda_root=CONFIG["lambda_root"],
        lambda_rot=CONFIG["lambda_rot"], 
        lambda_pos=CONFIG["lambda_pos"],
        lambda_vel=CONFIG["lambda_vel"], 
        lambda_smooth=CONFIG["lambda_smooth"], 
        lambda_contact=CONFIG["lambda_contact"],
        lambda_floor=CONFIG["lambda_floor"],
        lambda_tilt=CONFIG["lambda_tilt"]
    ).to(device)
    
    scaler = GradScaler('cuda', enabled=CONFIG["use_amp"])
    
    # 4. Resume Logic
    best_mpjpe = float("inf")
    start_epoch = 1
    checkpoint_load_path = os.path.join(CONFIG["save_dir"], "latest_checkpoint.pth")
    
    if os.path.exists(checkpoint_load_path):
        checkpoint = torch.load(checkpoint_load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_mpjpe = checkpoint.get('best_mpjpe', float('inf'))
        logger.info(f"[RESUME] Resuming from epoch {start_epoch} (Best MPJPE: {best_mpjpe:.2f})")

    # 5. Scheduler
    total_steps = len(train_loader) * CONFIG["epochs"]
    last_step_index = (start_epoch - 1) * len(train_loader) - 1
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CONFIG["lr"]*10, total_steps=total_steps, pct_start=CONFIG["warmup_pct"], last_epoch=last_step_index)
    
    # Load scheduler state if available (Critical for OneCycleLR resume)
    if os.path.exists(checkpoint_load_path) and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("[OK] Scheduler state loaded.")
        except Exception as e:
            logger.warning(f"[WARN] Could not load scheduler state: {e}")

    # 6. Loop
    for epoch in range(start_epoch, CONFIG["epochs"] + 1):
        logger.info(f"--- Epoch {epoch}/{CONFIG['epochs']} ---")
        
        # Train
        train_stats = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, device, epoch)
        logger.info(f"[TRAIN] Loss: {train_stats['loss']:.5f} | Root: {train_stats.get('l_root',0):.4f} | RAcc: {train_stats.get('l_root_acc', 0):.4f} | Pos: {train_stats.get('l_mpjpe', 0):.4f}")
        
        # Validate
        
        # Free memory from training loop
        del train_stats
        torch.cuda.empty_cache()
        
        metrics = validate(model, val_loader, device, epoch)
        # UPDATED LOGGING: Removed RootPos
        logger.info(f"[VAL] MPJPE: {metrics['MPJPE']:.2f}mm | RotDeg: {metrics.get('Root_Rot_Deg', 0):.2f}° | Slide: {metrics.get('Foot_Slide', 0):.2f}mm")
        
        # --- ROBUST SAVING LOGIC ---
        
        # 1. Create Checkpoint (Include Scheduler!)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 
            'best_mpjpe': best_mpjpe
        }
        
        # 2. Save "Latest" (Every Epoch)
        latest_path = os.path.join(CONFIG["save_dir"], "latest_checkpoint.pth")
        torch.save(checkpoint, latest_path)
        logger.info(f"[SAVE] Epoch {epoch} checkpoint saved.")

        # 3. Save Every 10 Epochs
        if epoch % 10 == 0:
            periodic_path = os.path.join(CONFIG["save_dir"], f"checkpoint_epoch_{epoch}.pth")
            torch.save(checkpoint, periodic_path)
            logger.info(f"[SAVE] Periodic checkpoint saved: {periodic_path}")
        
        # 3. Save "Best" (Conditional)
        if metrics['MPJPE'] < best_mpjpe:
            best_mpjpe = metrics['MPJPE']
            best_path = os.path.join(CONFIG["save_dir"], "best_model.pth")
            torch.save(checkpoint, best_path)
            logger.info(f"[BEST] New Best Model Saved! ({best_mpjpe:.2f}mm)")

if __name__ == "__main__":
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)
    main()