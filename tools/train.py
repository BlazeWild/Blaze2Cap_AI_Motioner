"""
Blaze2Cap Training Script (Hydra-Zen Enabled)
=============================================
Restored Hydra-Zen configuration while keeping RTX 4090 optimizations.

Usage:
    python -m tools.train
"""

import os
import sys
import functools
import logging
from collections import defaultdict

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# --- Hydra Zen Imports ---
import hydra
from hydra.core.config_store import ConfigStore
from hydra_zen import builds, make_config, instantiate, zen, to_yaml

# --- Blaze2Cap Modules ---
from blaze2cap.modules.models import MotionTransformer
from blaze2cap.modules.data_loader import PoseSequenceDataset
from blaze2cap.modeling.loss import MotionCorrectionLoss
from blaze2cap.modeling.eval_motion import evaluate_motion, MotionEvaluator
from blaze2cap.utils.logging_ import setup_logging
from blaze2cap.utils.checkpoint import save_checkpoint, load_checkpoint, auto_resume
from blaze2cap.utils.train_utils import CudaPreFetcher, set_random_seed, Timer
from blaze2cap.utils.skeleton_config import get_totalcapture_skeleton

# --- OPTIMIZATION (RTX 4090) ---
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# --- 1. CONFIGURATION (Hydra-Zen Style) ---
# Define the Optimizer/Scheduler Configs (Matching your optimization.py)
OptimizerConfig = builds(optim.AdamW, populate_full_signature=True)
SchedulerConfig = builds(optim.lr_scheduler.OneCycleLR, populate_full_signature=True)

# Main Experiment Config
ExperimentConfig = make_config(
    experiment_name="motion_transformer_hydra_zen",
    data_root="./blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset",
    save_dir="./checkpoints",
    log_dir="./logs",
    
    # Model
    num_joints=27,
    input_feats=18,
    d_model=256,
    num_layers=4,
    n_head=4,
    d_ff=512,
    dropout=0.1,
    max_len=512,
    
    # Training
    batch_size=32,
    num_workers=6,
    max_windows_per_sample=64,
    lr=1e-4,
    weight_decay=0.01,
    epochs=1000,
    window_size=64,
    warmup_pct=0.1,
    
    # Optimizer & Scheduler (Nested Configs)
    optimizer=OptimizerConfig(lr=1e-4, weight_decay=0.01),
    scheduler=SchedulerConfig(max_lr=1e-4, pct_start=0.1), # Steps/Epochs injected dynamically
    
    # Loss Weights
    lambda_root_vel=1.0,
    lambda_root_rot=2.0,
    lambda_pose_rot=1.0,
    lambda_pose_pos=4.0,
    lambda_floor=5.0,
    lambda_tilt=5.0,
    lambda_contact=0.5,
    lambda_smooth=10.0,
    lambda_accel=20.0,
    
    # System
    seed=42,
    use_amp=True,
    resume_checkpoint="auto",
    gradient_clip=1.0,
)

# Store the config
cs = ConfigStore.instance()
cs.store(name="config", node=ExperimentConfig)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler, device, epoch, cfg):
    """Train for one epoch (Optimized Loop)."""
    model.train()
    timer = Timer()
    stats = defaultdict(float)
    
    use_prefetcher = device == "cuda" and cfg.num_workers > 0
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
        
        if not use_prefetcher:
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
        
        src = batch["source"]
        mask = batch["mask"]
        tgt = batch["target"]

        optimizer.zero_grad(set_to_none=True)
        
        with autocast('cuda', enabled=cfg.use_amp and device == "cuda"):
            preds = model(src, key_padding_mask=None)
            timer.tick("forward")
            
            # Loss
            loss_dict = criterion(preds, tgt, mask)
            loss = loss_dict["loss"]
        
            if not torch.isfinite(loss):
                logging.getLogger(__name__).warning(f"NaN/Inf loss at step {batch_idx}. Skipping.")
                if use_prefetcher: batch = next(prefetcher, None)
                else: batch = next(batch_iter, None)
                batch_idx += 1
                continue
        
        # Backward
        if cfg.use_amp and device == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.gradient_clip)
            optimizer.step()
        
        scheduler.step()
        timer.tick("backward")
        
        # Stats
        stats["loss"] += loss.item()
        for k, v in loss_dict.items():
            if k != "loss": stats[k] += v.item()
        
        # Next
        if use_prefetcher: batch = next(prefetcher, None)
        else: batch = next(batch_iter, None)
        
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
    if len(batch) == 0:
        return {"source": torch.empty(0), "mask": torch.empty(0), "target": torch.empty(0)}
    
    sources, masks, targets = [], [], []
    for item in batch:
        src, msk, tgt = item["source"], item["mask"], item["target"]
        if max_windows_per_sample is not None and src.shape[0] > max_windows_per_sample:
            idx = torch.randperm(src.shape[0])[:max_windows_per_sample]
            src, msk, tgt = src[idx], msk[idx], tgt[idx]
        sources.append(src)
        masks.append(msk)
        targets.append(tgt)

    return {
        "source": torch.cat(sources, dim=0),
        "mask": torch.cat(masks, dim=0),
        "target": torch.cat(targets, dim=0)
    }

def validate(model, loader, criterion, device, epoch, skeleton_config=None):
    torch.cuda.empty_cache()
    model.eval()
    if skeleton_config is None:
        skeleton_config = get_totalcapture_skeleton()

    pbar = tqdm(total=len(loader), desc=f"Epoch {epoch} [VAL]")
    mare_sum, mare_count = 0.0, 0
    mpjpe_sum, mpjpe_count = 0.0, 0
    evaluator = MotionEvaluator(skeleton_config['parents'], skeleton_config['offsets'])

    with torch.no_grad():
        for batch in loader:
            src = batch["source"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)
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
    return {
        "MPJPE": mpjpe_sum / max(mpjpe_count, 1),
        "MARE": mare_sum / max(mpjpe_count, 1)
    }

# --- MAIN with HYDRA ZEN ---
@zen
def main(cfg):
    set_random_seed(cfg.seed)
    logger = setup_logging(cfg.log_dir, log_file="train.log")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Starting experiment: {cfg.experiment_name} on {device}")
    
    # 1. Data
    logger.info("Initializing Datasets...")
    train_dataset = PoseSequenceDataset(cfg.data_root, cfg.window_size, split="train", max_windows=cfg.max_windows_per_sample)
    val_dataset = PoseSequenceDataset(cfg.data_root, cfg.window_size, split="val", max_windows=None)
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, 
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=cfg.num_workers > 0, prefetch_factor=2 if cfg.num_workers > 0 else None,
        collate_fn=functools.partial(collate_flatten_windows, max_windows_per_sample=None)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, 
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0, prefetch_factor=2 if cfg.num_workers > 0 else None,
        collate_fn=functools.partial(collate_flatten_windows, max_windows_per_sample=None)
    )
    
    # 2. Model
    model = MotionTransformer(
        num_joints=cfg.num_joints, input_feats=cfg.input_feats,
        d_model=cfg.d_model, num_layers=cfg.num_layers,
        n_head=cfg.n_head, d_ff=cfg.d_ff, dropout=cfg.dropout,
        max_len=cfg.max_len
    ).to(device)
    
    # 3. Hydra-Zen Instantiation (Optimizer & Scheduler)
    # Instantiate Optimizer
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    
    # Instantiate Scheduler (inject steps dynamically)
    # Note: OneCycleLR needs 'steps_per_epoch' and 'epochs' or 'total_steps'
    scheduler = instantiate(
        cfg.scheduler, 
        optimizer=optimizer, 
        steps_per_epoch=len(train_loader),
        epochs=cfg.epochs
    )
    
    # 4. Loss
    skel_config = get_totalcapture_skeleton()
    parents = torch.tensor(skel_config['parents'], dtype=torch.long)
    offsets = skel_config['offsets']

    criterion = MotionCorrectionLoss(
        parents=parents, offsets=offsets,
        lambda_root_vel=cfg.lambda_root_vel,
        lambda_root_rot=cfg.lambda_root_rot,
        lambda_pose_rot=cfg.lambda_pose_rot,
        lambda_pose_pos=cfg.lambda_pose_pos,
        lambda_smooth=cfg.lambda_smooth,
        lambda_accel=cfg.lambda_accel,
        lambda_floor=cfg.lambda_floor,
        lambda_tilt=cfg.lambda_tilt,
        lambda_contact=cfg.lambda_contact
    ).to(device)
    
    scaler = GradScaler('cuda', enabled=cfg.use_amp and device == "cuda")
    
    # 5. Resume
    start_epoch = 0
    if cfg.resume_checkpoint == "auto":
        resume_path = auto_resume(cfg.save_dir)
        if resume_path:
            start_epoch = load_checkpoint(resume_path, model, optimizer, scheduler)
            logger.info(f"Resumed from epoch {start_epoch}")

    best_mpjpe = float("inf")
    
    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        logger.info(f"--- Epoch {epoch}/{cfg.epochs} ---")
        
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, device, epoch, cfg)
        
        log_str = f"Loss: {train_metrics['loss']:.4f} | MPJPE: {train_metrics.get('l_mpjpe', 0):.3f} | "
        log_str += f"Tilt: {train_metrics.get('l_tilt', 0):.3f} | Floor: {train_metrics.get('l_floor', 0):.3f} | Ct: {train_metrics.get('l_contact', 0):.4f}"
        logger.info(log_str)
        
        # Validate
        val_skel_config = {'parents': parents, 'offsets': offsets}
        val_metrics = validate(model, val_loader, criterion, device, epoch, skeleton_config=val_skel_config)
        
        mpjpe = val_metrics["MPJPE"]
        logger.info(f"Val MPJPE: {mpjpe:.4f} mm | Val MARE: {val_metrics['MARE']:.4f}")
        
        is_best = mpjpe < best_mpjpe
        if is_best:
            best_mpjpe = mpjpe
            logger.info(f"New Best! {best_mpjpe:.4f}")
            
        if epoch % 10 == 0 or is_best:
            # Note: save_checkpoint might need 'cfg' as a dict for serialization, hydra configs are DictConfig
            save_checkpoint(cfg.save_dir, epoch, model, optimizer, scheduler, config=dict(cfg), metrics=val_metrics, is_best=is_best)

    logger.info(f"Done. Best MPJPE: {best_mpjpe:.4f}")

if __name__ == "__main__":
    # Generate the config store and run
    # If you run this script directly, it defaults to the 'config' node we stored above.
    main()