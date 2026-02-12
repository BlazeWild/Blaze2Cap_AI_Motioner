"""
Blaze2Cap Training Script (Canonical Rotation - 27 Joints)
==========================================================
State-of-the-Art Configuration for 6D Rotation Learning.
Includes advanced Physics-based constraints (Contact, Floor, Tilt).
"""

import os
import sys
import functools
import logging
import glob
import re
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
# CRITICAL: Using the correct 27-joint/Rotation modules
from blaze2cap.modules.models import MotionTransformer
from blaze2cap.modules.data_loader import PoseSequenceDataset
from blaze2cap.modeling.loss import MotionLoss
from blaze2cap.utils.logging_ import setup_logging
from blaze2cap.utils.train_utils import set_random_seed, Timer

# --- OPTIMIZATION (RTX 4090) ---
torch.set_float32_matmul_precision('high') 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True 

# --- CONFIGURATION ---
CONFIG = {
    "experiment_name": "canonical_rotation_27joints_deep",
    "data_root": os.path.join(PROJECT_ROOT, "blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset"),
    "save_dir": os.path.join(PROJECT_ROOT, "checkpoints"),
    "log_dir": os.path.join(PROJECT_ROOT, "logs"),
    
    # Model Hyperparameters
    "num_joints_in": 27,    # <--- CHANGED: Using full BlazePose (27 joints)
    "input_feats": 19,      # <--- CHANGED: 19 Features [Pos(3), Vel(3), Par(3), Chi(3), Vis(1), Anc(1), Align(2), SVel(2), Scale(1)]
    "num_joints_out": 22,   # 22 Output Body Joints (Rotations)
    
    "d_model": 512,
    "num_layers": 6,        # <--- DEEPER MODEL (Better for Rotation)
    "n_head": 8,
    "d_ff": 1024,
    "dropout": 0.1,
    "max_len": 512,
    
    # Training
    "batch_size": 32,       # Keep 32 for stability
    "num_workers": 6,
    "max_windows_train": 64, 
    "lr": 1e-4,             
    "weight_decay": 1e-4,
    "epochs": 150,
    "window_size": 64,
    "warmup_pct": 0.1,
    
    # Loss Weights (Advanced Physics Constraints)
    "lambda_rot": 3.0,        # <--- INCREASED: Prioritize Structure
    "lambda_pos": 2.0,        # <--- BALANCED: Ensure FK validity
    "lambda_vel": 1.0,        # Dynamics
    "lambda_smooth": 0.5,     # Anti-Jitter (Increased from 0.1)
    "lambda_contact": 2.0,    # Anti-Slide (Increased from 0.5)
    "lambda_floor": 2.0,      # Anti-Clipping (NEW)
    "lambda_tilt": 1.0,       # Spine Stability (NEW)
    
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
    stats = defaultdict(float)
    
    # Progress Bar with LR Logging
    pbar = tqdm(loader, desc=f"Epoch {epoch} [TRAIN]")
    
    for batch_idx, batch in enumerate(pbar):
        # 1. Inputs
        src = batch["source"].to(device, non_blocking=True) # (B, 64, 27, 19)
        tgt_rot = batch["target"].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # 2. Forward & Loss
        with autocast('cuda', enabled=CONFIG["use_amp"]):
            preds = model(src)
            loss, loss_logs = criterion(preds, tgt_rot)
            
            if not torch.isfinite(loss):
                continue

        # 3. Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step() # Step every batch for OneCycleLR
        
        # 4. Logging
        stats["loss"] += loss.item()
        for k, v in loss_logs.items():
            stats[k] += v
            
        # Get Current LR
        current_lr = scheduler.get_last_lr()[0]
            
        # Updated Postfix to show key metrics and LR
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}", 
            "Rot": f"{loss_logs.get('l_rot', 0):.4f}",
            "Pos": f"{loss_logs.get('l_mpjpe', 0):.4f}",
            "Slide": f"{loss_logs.get('l_contact', 0):.4f}",
            "LR": f"{current_lr:.2e}" 
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
            
            # Forward (Output is Rotations)
            preds_rot = model(src)
            
            # Convert both to positions for metric using Loss FK Helper
            pred_pos = criterion._run_canonical_fk(preds_rot)
            
            gt_body_rot = tgt_rot[:, :, 2:, :] 
            tgt_pos = criterion._run_canonical_fk(gt_body_rot)
            
            # Calculate MPJPE
            diff = pred_pos - tgt_pos
            dist = torch.norm(diff, dim=-1)
            batch_mpjpe = dist.mean().item()
            
            B = src.shape[0]
            mpjpe_sum += batch_mpjpe * B
            count += B
            
    avg_mpjpe_m = mpjpe_sum / max(count, 1)
    avg_mpjpe_mm = avg_mpjpe_m * 1000.0
    
    return avg_mpjpe_mm

def collate_flatten(batch):
    return {
        "source": torch.cat([b["source"] for b in batch], dim=0),
        "target": torch.cat([b["target"] for b in batch], dim=0)
    }

def main():
    set_random_seed(CONFIG["seed"])
    logger = setup_logging(CONFIG["log_dir"], log_file="train.log")
    device = CONFIG["device"]
    
    logger.info(f"🚀 Starting Deep Rotation Training (L={CONFIG['num_layers']}) on {device}")
    
    # 1. Data
    train_dataset = PoseSequenceDataset(CONFIG["data_root"], CONFIG["window_size"], "train", CONFIG["max_windows_train"])
    val_dataset = PoseSequenceDataset(CONFIG["data_root"], CONFIG["window_size"], "val", None)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, 
                              num_workers=CONFIG["num_workers"], pin_memory=True, collate_fn=collate_flatten, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, 
                            num_workers=CONFIG["num_workers"], pin_memory=True, collate_fn=collate_flatten)
    
    logger.info(f"Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")
    
    # 2. Model
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
    
    try:
        input_shape = (CONFIG['batch_size'], CONFIG['window_size'], CONFIG['num_joints_in'], CONFIG['input_feats'])
        logger.info(f"Model Summary:\n{summary(model, input_size=input_shape, device=device)}")
    except Exception as e:
        logger.warning(f"Could not print summary: {e}")

    # 3. Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG["lr"], 
        weight_decay=CONFIG["weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 4. Loss
    criterion = MotionLoss(
        lambda_rot=CONFIG["lambda_rot"], 
        lambda_pos=CONFIG["lambda_pos"],
        lambda_vel=CONFIG["lambda_vel"], 
        lambda_smooth=CONFIG["lambda_smooth"], 
        lambda_contact=CONFIG["lambda_contact"],
        lambda_floor=CONFIG["lambda_floor"],
        lambda_tilt=CONFIG["lambda_tilt"]
    ).to(device)
    
    scaler = GradScaler('cuda', enabled=CONFIG["use_amp"])
    
    # 5. Checkpoint & Resume Logic
    best_mpjpe = float("inf")
    start_epoch = 1
    
    checkpoint_load_path = os.path.join(CONFIG["save_dir"], "latest_checkpoint.pth")
    if not os.path.exists(checkpoint_load_path):
        available_checkpoints = glob.glob(os.path.join(CONFIG["save_dir"], "checkpoint_epoch*.pth"))
        if available_checkpoints:
            available_checkpoints.sort(key=lambda f: int(re.findall(r'\d+', os.path.basename(f))[0]), reverse=True)
            checkpoint_load_path = available_checkpoints[0]

    if os.path.exists(checkpoint_load_path):
        logger.info(f"🔄 Loading checkpoint from {checkpoint_load_path}")
        try:
            checkpoint = torch.load(checkpoint_load_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger.info("✅ Optimizer state loaded successfully")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load optimizer state: {e}. Using fresh optimizer.")
            
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_mpjpe = checkpoint.get('best_mpjpe', float('inf'))
            logger.info(f"🚀 Resuming from epoch {start_epoch} (Best MPJPE: {best_mpjpe:.2f})")
        except Exception as e:
            logger.error(f"❌ Error loading checkpoint: {e}. Starting from scratch.")
            start_epoch = 1
            best_mpjpe = float("inf")
    else:
        logger.info("🆕 No checkpoint found. Starting training from scratch.")

    # 6. Scheduler (OneCycleLR with Resume Support)
    # Total steps for the entire training run
    total_steps = len(train_loader) * CONFIG["epochs"]
    
    # Calculate where we are if resuming (last completed step index)
    # If starting fresh (epoch 1), last_epoch should be -1
    last_step_index = (start_epoch - 1) * len(train_loader) - 1
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG["lr"] * 10, # Peak LR (usually 10x base)
        total_steps=total_steps,
        pct_start=CONFIG["warmup_pct"],
        last_epoch=last_step_index # Resume logic
    )
    
    # Attempt to load scheduler state if available (overrides calculation if valid)
    if start_epoch > 1 and os.path.exists(checkpoint_load_path):
        try:
            checkpoint = torch.load(checkpoint_load_path, map_location=device)
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("✅ Scheduler state loaded successfully")
        except:
            pass

    # 7. Training Loop
    latest_path = os.path.join(CONFIG["save_dir"], "latest_checkpoint.pth")
    
    for epoch in range(start_epoch, CONFIG["epochs"] + 1):
        logger.info(f"--- Epoch {epoch}/{CONFIG['epochs']} ---")
        
        # Train
        train_stats = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, device, epoch)
        
        logger.info(f"[TRAIN] Loss: {train_stats['loss']:.5f} | Rot: {train_stats.get('l_rot', 0):.5f} | Pos: {train_stats.get('l_mpjpe', 0):.5f}")
        
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
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mpjpe': best_mpjpe,
            }, os.path.join(CONFIG["save_dir"], "best_model.pth"))
            logger.info(f"⭐ New Best Model Saved! ({val_mpjpe_mm:.2f} mm)")
            
        # Save Regular Checkpoint & Latest
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_mpjpe': best_mpjpe,
        }
        torch.save(checkpoint_data, latest_path)
        
        if epoch % 10 == 0:
            torch.save(checkpoint_data, os.path.join(CONFIG["save_dir"], f"checkpoint_epoch{epoch}.pth"))

    logger.info(f"Training Complete. Best MPJPE: {best_mpjpe:.2f} mm")

if __name__ == "__main__":
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)
    main()