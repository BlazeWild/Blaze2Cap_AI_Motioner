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
import functools

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
# from torch.cuda.amp import GradScaler, autocast # Deprecated in 2.4+
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# --- Blaze2Cap Modules (via __init__.py exports) ---
from blaze2cap.modules.models import MotionTransformer
from blaze2cap.modules.data_loader import PoseSequenceDataset
from blaze2cap.modeling.loss import MotionCorrectionLoss
from blaze2cap.modeling.eval_motion import evaluate_motion, MotionEvaluator
from blaze2cap.utils.logging_ import setup_logging
from blaze2cap.utils.checkpoint import save_checkpoint, load_checkpoint, auto_resume
from blaze2cap.utils.train_utils import CudaPreFetcher, set_random_seed, Timer
from blaze2cap.utils.skeleton_config import get_totalcapture_skeleton

# --- OPTIMIZATION (RTX 4090) ---
torch.set_float32_matmul_precision('high') # Enable TF32 for significantly faster FP32 matmuls
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True # Enable CUDNN auto-tuner

# --- CONFIGURATION (Optimized for RTX 4090 GPU - 24GB VRAM) ---
CONFIG = {
    "experiment_name": "motion_transformer_RTX4090",
    "data_root": "./blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset",
    "save_dir": "./checkpoints",
    "log_dir": "./logs",
    
    # Model Hyperparameters (unchanged)
    "num_joints": 27,
    "input_feats": 18,
    "d_model": 256,
    "num_layers": 4,
    "n_head": 4,
    "d_ff": 512,
    "dropout": 0.1,
    "max_len": 512,
    
    # Training Hyperparameters (RTX 4090 / L40S Optimized)
    # OPTIMIZATION NOTE:
    # "Effective Batch Size" = batch_size * max_windows_per_sample
    # We aim for ~2048 - 4096 windows per step for the 4090.
    # Previous settings (128 * 512 = 65k) were causing System RAM OOM (100GB+).
    # New settings (32 * 64 = 2048) will use ~2GB RAM buffer and saturate Tensor Cores.
    "batch_size": 32,         # Number of FILES to load per step
    "num_workers": 6,         # Reduced workers to save overhead
    "max_windows_per_sample": 64,   # Number of WINDOWS to sample per file
    "lr": 1e-4,
    "weight_decay": 0.01,
    "epochs": 1000,
    "window_size": 64,
    "warmup_pct": 0.1,
    
    # Loss Weights
    # "lambda_root_vel": 1.0,
    # "lambda_root_rot": 1.0,
    # "lambda_pose_rot": 1.0,
    # "lambda_pose_pos": 4.0,
    # "lambda_smooth": 10.0,
    # "lambda_accel": 20.0,

    # Loss Weights - AGGRESSIVE RETUNING
    "lambda_root_vel": 50.0,   # WAS 1.0. Needs huge boost to match Rotation scale.
    "lambda_root_rot": 10.0,    # WAS 1.0. Facing direction is critical.
    
    "lambda_pose_rot": 2.0,     # Keep standard (Baseline)
    
    "lambda_pose_pos": 10.0,    # WAS 4.0. Boost MPJPE to force correct structure.
    
    # Turn OFF Smoothness for now. 
    # If the structure is wrong, smoothing it just makes it "smoothly wrong".
    "lambda_smooth": 0.0,       # WAS 10.0. 
    "lambda_accel": 0.0,        # WAS 20.0.
    
    # System
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_amp": True,
    "resume_checkpoint": "auto",
    "gradient_clip": 1.0,
}

# Fix fragmentation issues
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler, device, epoch, config):
    """Train for one epoch with mixed precision support."""
    model.train()
    timer = Timer()
    
    # Track all loss components
    stats = defaultdict(float)
    
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
            # Optimize: use non_blocking=True for asynchronous transfer
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
        
        src = batch["source"]
        mask = batch["mask"]
        tgt = batch["target"]

        # 2. Forward Pass with Mixed Precision
        # Optimize: set_to_none=True saves memory bandwidth
        optimizer.zero_grad(set_to_none=True)
        
        with autocast('cuda', enabled=config["use_amp"] and device == "cuda"):
            # Passing key_padding_mask=None ensures the model attends to ALL history (including padding).
            # The padding is valid static data essential for context.
            preds = model(src, key_padding_mask=None)
            timer.tick("forward")
            
            # 3. Compute Loss
            loss_dict = criterion(preds, tgt, mask)
            loss = loss_dict["loss"]
        
            # Skip step on non-finite loss
            if not torch.isfinite(loss):
                if device == "cuda":
                    torch.cuda.synchronize() # Only sync on error
                # Log the error
                import logging
                logging.getLogger(__name__).warning(f"NaN/Inf loss detected at step {batch_idx}. Skipping.")
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
        
        # 5. Scheduler
        scheduler.step()
        timer.tick("backward")
        
        # 6. Accumulate Stats
        stats["loss"] += loss.item()
        for k, v in loss_dict.items():
            if k != "loss":
                stats[k] += v.item()
        
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
    
    N = max(batch_idx, 1)
    avg_stats = {k: v / N for k, v in stats.items()}
    avg_stats["time_stats"] = timer.report()
    return avg_stats


def collate_flatten_windows(batch, max_windows_per_sample=None):
    """Concatenate variable-length window stacks across samples."""
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
    """Validate the model and compute MPJPE/MARE metrics."""
    # Free up memory before validation (fragmentation cleanup)
    torch.cuda.empty_cache()
    
    model.eval()

    if skeleton_config is None:
        skeleton_config = get_totalcapture_skeleton()

    pbar = tqdm(total=len(loader), desc=f"Epoch {epoch} [VAL]")

    mare_sum = 0.0
    mare_count = 0
    mpjpe_sum = 0.0
    mpjpe_count = 0

    evaluator = MotionEvaluator(skeleton_config['parents'], skeleton_config['offsets'])

    with torch.no_grad():
        for batch in loader:
            # Use non_blocking=True
            src = batch["source"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)

            # Passing key_padding_mask=None ensures full historical context
            root_out, body_out = model(src, key_padding_mask=None)
            pred_combined = torch.cat([root_out, body_out], dim=2)

            if tgt.dim() == 3 and tgt.shape[-1] == 132:
                tgt = tgt.view(tgt.shape[0], tgt.shape[1], 22, 6)

            metrics_batch = evaluator.compute_metrics(pred_combined, tgt)
            
            B = pred_combined.shape[0]
            mpjpe_sum += metrics_batch["MPJPE"] * B
            mare_sum += metrics_batch["MARE"] * B
            mpjpe_count += B
            
            pbar.update(1)

    pbar.close()

    metrics = {
        "MPJPE": mpjpe_sum / max(mpjpe_count, 1),
        "MARE": mare_sum / max(mpjpe_count, 1)
    }
    return metrics


def main():
    # 1. Setup
    set_random_seed(CONFIG["seed"])
    logger = setup_logging(CONFIG["log_dir"], log_file="train.log")
    device = CONFIG["device"]
    logger.info(f"Starting experiment: {CONFIG['experiment_name']} on {device}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
        logger.info(f"CUDNN Benchmark: {torch.backends.cudnn.benchmark}")
    
    # 2. Data
    logger.info("Initializing Datasets...")
    train_dataset = PoseSequenceDataset(CONFIG["data_root"], CONFIG["window_size"], split="train", max_windows=CONFIG.get("max_windows_per_sample"))
    val_dataset = PoseSequenceDataset(CONFIG["data_root"], CONFIG["window_size"], split="val", max_windows=None)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=CONFIG["num_workers"] > 0,
        prefetch_factor=2 if CONFIG["num_workers"] > 0 else None,
        collate_fn=functools.partial(collate_flatten_windows, max_windows_per_sample=None)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, # Use 8 as requested by user - fits in memory while keeping full sequences
        shuffle=False, 
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        persistent_workers=CONFIG["num_workers"] > 0,
        prefetch_factor=2 if CONFIG["num_workers"] > 0 else None,
        collate_fn=functools.partial(collate_flatten_windows, max_windows_per_sample=None)
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
    
    # Log original model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params:,} total params, {trainable_params:,} trainable")

    # 4. Optimize Model (torch.compile)
    # DISABLED: Triton is not supported on Windows, causing crashes.
    # if hasattr(torch, "compile"):
    #     logger.info("Compiling model with torch.compile() for RTX 4090...")
    #     try:
    #         # mode='reduce-overhead' is great for smaller batches, 'max-autotune' for throughput
    #         # Since we increased batch size, default or max-autotune is good. 
    #         # Sticking to default for stability on Windows.
    #         model = torch.compile(model) 
    #     except Exception as e:
    #         logger.warning(f"torch.compile failed: {e}. continuing without compilation.")
    
    # 5. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG["lr"], 
        weight_decay=CONFIG["weight_decay"]
    )
    
    # OneCycleLR
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG["lr"],
        steps_per_epoch=len(train_loader),
        epochs=CONFIG["epochs"],
        pct_start=CONFIG["warmup_pct"]
    )
    
    # 6. Loss
    # Use static skeleton config from utils
    # Removed fragile dynamic extraction
    skel_config = get_totalcapture_skeleton()
    parents = torch.tensor(skel_config['parents'], dtype=torch.long)
    offsets = skel_config['offsets'] # Already a tensor
    
    logger.info("Static Skeleton (TotalCapture) loaded successfully.")

    criterion = MotionCorrectionLoss(
        parents=parents,
        offsets=offsets,
        lambda_root_vel=CONFIG["lambda_root_vel"],
        lambda_root_rot=CONFIG["lambda_root_rot"],
        lambda_pose_rot=CONFIG["lambda_pose_rot"],
        lambda_pose_pos=CONFIG["lambda_pose_pos"],
        lambda_smooth=CONFIG["lambda_smooth"],
        lambda_accel=CONFIG["lambda_accel"]
    ).to(device)
    
    # 7. Mixed Precision
    scaler = GradScaler('cuda', enabled=CONFIG["use_amp"] and device == "cuda")
    
    # 8. Resume
    start_epoch = 0
    if CONFIG["resume_checkpoint"] == "auto":
        resume_path = auto_resume(CONFIG["save_dir"])
        if resume_path:
            start_epoch = load_checkpoint(resume_path, model, optimizer, scheduler)
            logger.info(f"Resumed from epoch {start_epoch}")
    elif CONFIG["resume_checkpoint"]:
        start_epoch = load_checkpoint(
            CONFIG["resume_checkpoint"], model, optimizer, scheduler
        )
        logger.info(f"Resumed from checkpoint: {CONFIG['resume_checkpoint']}")

    # 9. Training Loop
    best_mpjpe = float("inf")
    
    for epoch in range(start_epoch + 1, CONFIG["epochs"] + 1):
        logger.info(f"--- Epoch {epoch}/{CONFIG['epochs']} ---")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, 
            criterion, scaler, device, epoch, CONFIG
        )
        
        # Logging
        log_str = f"Train Loss: {train_metrics['loss']:.5f} | "
        log_str += f"RootV: {train_metrics.get('l_root_vel', 0):.4f} "
        log_str += f"RootR: {train_metrics.get('l_root_rot', 0):.4f} "
        log_str += f"PoseR: {train_metrics.get('l_pose_rot', 0):.4f} "
        log_str += f"MPJPE: {train_metrics.get('l_pose_pos', 0):.4f} "
        log_str += f"Smth: {train_metrics.get('l_smooth', 0):.4f} "
        log_str += f"Accel: {train_metrics.get('l_accel', 0):.4f}"
        
        logger.info(log_str)
        logger.debug(f"Timing:\n{train_metrics['time_stats']}")
        
        # Validate
        # Validate
        # Construct skeleton config for evaluator
        val_skel_config = None
        if parents is not None and offsets is not None:
             val_skel_config = {
                 'parents': parents,
                 'offsets': offsets,
                 'joint_names': get_totalcapture_skeleton()['joint_names']
             }

        val_metrics = validate(model, val_loader, criterion, device, epoch, skeleton_config=val_skel_config)
        mpjpe = val_metrics["MPJPE"]
        mare = val_metrics["MARE"]
        
        logger.info(f"Val MPJPE: {mpjpe:.5f} mm | Val MARE: {mare:.5f}")
        
        # Save Checkpoint
        is_best = mpjpe < best_mpjpe
        if is_best:
            best_mpjpe = mpjpe
            logger.info(f"New best MPJPE: {best_mpjpe:.5f}")
            
        if epoch % 10 == 0 or is_best:
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