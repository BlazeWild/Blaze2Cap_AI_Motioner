"""
Blaze2Cap Training Script
========================
Train the MotionTransformer model to predict 3D skeletal motion from BlazePose landmarks.

Usage:
    cd Blaze2Cap_full
    python -m tools.train
"""

import os
import sys

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

# --- Blaze2Cap Modules (via __init__.py exports) ---
from blaze2cap.modules.models import MotionTransformer
from blaze2cap.data.data_loader import PoseSequenceDataset
from blaze2cap.modeling.loss import MotionCorrectionLoss
from blaze2cap.modeling.eval_motion import evaluate_motion, MotionEvaluator
from blaze2cap.utils.logging import setup_logging
from blaze2cap.utils.checkpoint import save_checkpoint, load_checkpoint, auto_resume
from blaze2cap.utils.train_utils import CudaPreFetcher, set_random_seed, Timer

# --- CONFIGURATION (Optimized for L4 GPU - 24GB VRAM) ---
CONFIG = {
    "experiment_name": "motion_transformer_L4_smooth",
    "data_root": "./blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset",
    "save_dir": "./checkpoints",
    "log_dir": "./logs",
    
    # Model Hyperparameters
    "num_joints": 25,
    "input_feats": 18,
    "d_model": 256,
    "num_layers": 4,
    "n_head": 4,
    "d_ff": 512,
    "dropout": 0.1,
    "max_len": 512,
    
    # Training Hyperparameters (L4 GPU Optimized)
    "batch_size": 4,         # Smaller batch since each sample yields many windows
    "num_workers": 0,        # Use single-process loading for stability
    "max_windows_per_sample": 256,  # Limit windows per file to keep batches manageable
    "lr": 1e-4,
    "weight_decay": 0.01,
    "epochs": 100,           # More epochs for better convergence
    "window_size": 64,       # Larger window = better temporal context = smoother motion
    "warmup_pct": 0.1,
    
    # Loss Weights (Motion Smoothing Focused)
    "lambda_rot": 1.0,       # Keep geometry grounded
    "lambda_smooth": 5.0,    # High smoothness weight for velocity consistency
    
    # System
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_amp": False,        # Disabled to avoid NaN/overflow issues
    "resume_checkpoint": None,  # Set to "auto" for auto-resume or path to .pth
    "gradient_clip": 1.0,
}


def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler, device, epoch, config):
    """Train for one epoch with mixed precision support."""
    model.train()
    timer = Timer()
    
    running_loss = 0.0
    running_rot = 0.0
    running_smooth = 0.0
    
    # Use CudaPreFetcher for speed (only if CUDA available)
    use_prefetcher = device == "cuda" and config.get("num_workers", 0) > 0
    if use_prefetcher:
        prefetcher = CudaPreFetcher(loader, device)
        batch = next(prefetcher)
        total_batches = len(loader)
    else:
        batch_iter = iter(loader)
        batch = next(batch_iter, None)
        total_batches = len(loader)
    
    pbar = tqdm(total=total_batches, desc=f"Epoch {epoch} [TRAIN]")
    batch_idx = 0
    
    while batch is not None:
        timer.tick("data_load")
        
        # 1. Unpack Batch and move to device (if not using prefetcher)
        if not use_prefetcher:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
        
        src = batch["source"]   # [B, S, 25, 18] or [B, S, 450]
        mask = batch["mask"]    # [B, S]
        tgt = batch["target"]   # [B, S, 132]

        # Sanitize any remaining NaNs/Infs
        if not torch.isfinite(src).all():
            src = torch.nan_to_num(src, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(tgt).all():
            tgt = torch.nan_to_num(tgt, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2. Forward Pass with Mixed Precision
        optimizer.zero_grad()
        
        with autocast(enabled=config["use_amp"] and device == "cuda"):
            preds = model(src, key_padding_mask=None)
            timer.tick("forward")
            
            # 3. Compute Loss
            loss_dict = criterion(preds, tgt, mask)
            loss = loss_dict["loss"]
        
            # Skip step on non-finite loss to avoid NaN propagation
            if not torch.isfinite(loss):
                if device == "cuda":
                    torch.cuda.synchronize()
                pbar.update(1)
                pbar.set_postfix({"loss": "nan", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
                if use_prefetcher:
                    batch = next(prefetcher, None)
                else:
                    batch = next(batch_iter, None)
                batch_idx += 1
                continue
        
        # 4. Backward Pass with Gradient Scaling
        if config["use_amp"] and device == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["gradient_clip"])
            optimizer.step()
        
        # 5. Scheduler Step (per batch for OneCycleLR)
        scheduler.step()
        timer.tick("backward")
        
        # 6. Stats
        running_loss += loss.item()
        running_rot += loss_dict["l_rot"].item()
        running_smooth += loss_dict["l_smooth"].item()
        
        # Next batch
        if use_prefetcher:
            batch = next(prefetcher, None)
        else:
            batch = next(batch_iter, None)
        
        batch_idx += 1
        pbar.update(1)
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })

    pbar.close()
    
    # Return average metrics
    N = max(batch_idx, 1)
    return {
        "loss": running_loss / N,
        "rot": running_rot / N,
        "smooth": running_smooth / N,
        "time_stats": timer.report()
    }


def collate_flatten_windows(batch, max_windows_per_sample=None):
    """Concatenate variable-length window stacks across samples.

    Each dataset item returns tensors shaped (F, window, ...). We concatenate along F
    to create a single large batch with consistent window length.
    """
    if len(batch) == 0:
        return {"source": torch.empty(0), "mask": torch.empty(0), "target": torch.empty(0)}

    sources = []
    masks = []
    targets = []

    for item in batch:
        src = item["source"]
        msk = item["mask"]
        tgt = item["target"]

        if max_windows_per_sample is not None and src.shape[0] > max_windows_per_sample:
            idx = torch.randperm(src.shape[0])[:max_windows_per_sample]
            src = src[idx]
            msk = msk[idx]
            tgt = tgt[idx]

        sources.append(src)
        masks.append(msk)
        targets.append(tgt)

    return {
        "source": torch.cat(sources, dim=0),
        "mask": torch.cat(masks, dim=0),
        "target": torch.cat(targets, dim=0)
    }


def validate(model, loader, criterion, device, epoch, skeleton_config=None):
    """Validate the model and compute MPJPE/MARE metrics without storing all batches."""
    model.eval()

    # Default skeleton config (should be replaced with real values)
    if skeleton_config is None:
        skeleton_config = {
            'parents': [0, 0, 1, 2, 3, 4, 5, 6, 5, 8, 9, 10, 5, 12, 13, 14, 0, 16, 17, 0, 19, 20],
            'offsets': torch.ones(22, 3) * 0.1
        }

    # Use simple iteration for validation
    pbar = tqdm(total=len(loader), desc=f"Epoch {epoch} [VAL]")

    mare_sum = 0.0
    mare_count = 0
    mpjpe_sum = 0.0
    mpjpe_count = 0

    evaluator = MotionEvaluator(skeleton_config['parents'], skeleton_config['offsets'])

    with torch.no_grad():
        for batch in loader:
            src = batch["source"].to(device)
            mask = batch["mask"].to(device)
            tgt = batch["target"].to(device)  # [B, S, 132]

            pred_combined = model.forward_combined(src, key_padding_mask=None)

            if tgt.dim() == 3 and tgt.shape[-1] == 132:
                tgt = tgt.view(tgt.shape[0], tgt.shape[1], 22, 6)

            # MARE accumulation (mean of squared diff over all elements)
            diff = (pred_combined - tgt) ** 2
            mare_sum += diff.sum().item()
            mare_count += diff.numel()

            # MPJPE accumulation (compute FK per batch)
            pred_xyz = evaluator._forward_kinematics(pred_combined)
            gt_xyz = evaluator._forward_kinematics(tgt)
            dist = torch.norm(pred_xyz - gt_xyz, dim=-1)
            mpjpe_sum += dist.sum().item()
            mpjpe_count += dist.numel()

            pbar.update(1)

    pbar.close()

    metrics = {
        "MPJPE": mpjpe_sum / max(mpjpe_count, 1),
        "MARE": mare_sum / max(mare_count, 1)
    }
    return metrics


def main():
    # 1. Setup
    set_random_seed(CONFIG["seed"])
    logger = setup_logging(CONFIG["log_dir"], log_file="train.log")
    device = CONFIG["device"]
    logger.info(f"Starting experiment: {CONFIG['experiment_name']} on {device}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 2. Data
    logger.info("Initializing Datasets...")
    train_dataset = PoseSequenceDataset(CONFIG["data_root"], CONFIG["window_size"], split="train")
    val_dataset = PoseSequenceDataset(CONFIG["data_root"], CONFIG["window_size"], split="test")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        drop_last=True,  # Avoid batch size issues
        collate_fn=lambda b: collate_flatten_windows(b, CONFIG.get("max_windows_per_sample"))
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=False, 
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        collate_fn=lambda b: collate_flatten_windows(b, None)
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # 3. Model
    model = MotionTransformer(
        num_joints=CONFIG["num_joints"],
        input_feats=CONFIG["input_feats"],
        d_model=CONFIG["d_model"],
        num_layers=CONFIG["num_layers"],
        n_head=CONFIG["n_head"],
        d_ff=CONFIG["d_ff"],
        dropout=CONFIG["dropout"],
        max_len=CONFIG["max_len"]
    ).to(device)
    
    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params:,} total params, {trainable_params:,} trainable")
    
    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG["lr"], 
        weight_decay=CONFIG["weight_decay"]
    )
    
    # OneCycleLR: scheduler.step() called per BATCH
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG["lr"],
        steps_per_epoch=len(train_loader),
        epochs=CONFIG["epochs"],
        pct_start=CONFIG["warmup_pct"]
    )
    
    # 5. Loss
    criterion = MotionCorrectionLoss(
        lambda_rot=CONFIG["lambda_rot"], 
        lambda_smooth=CONFIG["lambda_smooth"]
    ).to(device)
    
    # 6. Mixed Precision Scaler
    scaler = GradScaler(enabled=CONFIG["use_amp"] and device == "cuda")
    
    # 7. Resume (Optional)
    start_epoch = 0
    if CONFIG["resume_checkpoint"] == "auto":
        resume_path = auto_resume(CONFIG["save_dir"])
        if resume_path:
            start_epoch = load_checkpoint(resume_path, model, optimizer, scheduler)
            # Adjust scheduler for resumed training
            logger.info(f"Resumed from epoch {start_epoch}")
    elif CONFIG["resume_checkpoint"]:
        start_epoch = load_checkpoint(
            CONFIG["resume_checkpoint"], model, optimizer, scheduler
        )
        logger.info(f"Resumed from checkpoint: {CONFIG['resume_checkpoint']}")

    # 8. Training Loop
    best_mpjpe = float("inf")
    
    for epoch in range(start_epoch + 1, CONFIG["epochs"] + 1):
        logger.info(f"--- Epoch {epoch}/{CONFIG['epochs']} ---")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, 
            criterion, scaler, device, epoch, CONFIG
        )
        logger.info(f"Train Loss: {train_metrics['loss']:.5f} "
                   f"(Rot: {train_metrics['rot']:.5f}, Smooth: {train_metrics['smooth']:.5f})")
        logger.debug(f"Timing:\n{train_metrics['time_stats']}")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        mpjpe = val_metrics["MPJPE"]
        mare = val_metrics["MARE"]
        
        logger.info(f"Val MPJPE: {mpjpe:.5f} | Val MARE: {mare:.5f}")
        
        # Save Checkpoint
        is_best = mpjpe < best_mpjpe
        if is_best:
            best_mpjpe = mpjpe
            logger.info(f"New best MPJPE: {best_mpjpe:.5f}")
            
        save_checkpoint(
            CONFIG["save_dir"], 
            epoch, 
            model, 
            optimizer, 
            scheduler,
            config=CONFIG,
            metrics=val_metrics, 
            is_best=is_best
        )

        # Save milestone checkpoints every 25 epochs
        if epoch % 25 == 0:
            save_checkpoint(
                CONFIG["save_dir"],
                epoch,
                model,
                optimizer,
                scheduler,
                config=CONFIG,
                metrics=val_metrics,
                is_best=False,
                prefix="milestone"
            )
    
    logger.info(f"Training complete! Best MPJPE: {best_mpjpe:.5f}")


if __name__ == "__main__":
    main()