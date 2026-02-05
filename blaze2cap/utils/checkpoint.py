import logging
import os
from collections import OrderedDict
from typing import Union, Tuple, Any, List

import torch

# create a logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # ensure logs print to console

def auto_resume(ckpt_folder):
    """
    Automatically finds the latest checkpoint in a folder based on modification time.
    """
    if not os.path.exists(ckpt_folder):
        return None
    try:
        ckpt_files = [
            os.path.join(ckpt_folder, f)
            for f in os.listdir(ckpt_folder)
            if f.endswith('.pth')
        ]
    except FileNotFoundError:
        return None
    
    if len(ckpt_files)>0:
        # pick the latest modified file
        latest_ckpt = max(ckpt_files, key=os.path.getmtime)
        logger.info(f"Auto-resuming from latest checkpoint: {latest_ckpt}")
        return latest_ckpt
    else:
        return None
    
def save_checkpoint(ckpt_folder, epoch, model, optimizer=None, scheduler=None, 
                    config=None, metrics=None, is_best=False, prefix="checkpoint"):
    """
    Saves the full training state (Model, Optimizer, Scheduler, Config, Metrics).
    
    Args:
        ckpt_folder: Directory to save checkpoints
        epoch: Current epoch number
        model: PyTorch model (handles DataParallel)
        optimizer: Optional optimizer to save state
        scheduler: Optional scheduler to save state
        config: Optional config dict to save
        metrics: Optional dict of validation metrics
        is_best: If True, also saves as 'best_model.pth'
        prefix: Prefix for checkpoint filename
    """
    # unwrap data parallel if present to save clean weights
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
        
    state_dict = {
        "epoch": epoch,
        "model": model_state,
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "config": config,
        "metrics": metrics
    }
    
    os.makedirs(ckpt_folder, exist_ok=True)
    ckpt_path = os.path.join(ckpt_folder, f"{prefix}_epoch{epoch}.pth")
    torch.save(state_dict, ckpt_path)
    logger.info(f"Checkpoint saved at {ckpt_path}")
    
    # Save best model separately
    if is_best:
        best_path = os.path.join(ckpt_folder, "best_model.pth")
        torch.save(state_dict, best_path)
        logger.info(f"New best model saved at {best_path}")
    
    return ckpt_path

def load_checkpoint(ckpt_file, model, optimizer=None, scheduler=None, restart_train=False):
    """
    Robustly loads a checkpoint. Handles size mismatches and missing keys gracefully.
    """
    logger.info(f"Loading checkpoint from {ckpt_file}....")
    # load to cpu to avoid OOM error
    checkpoint = torch.load(ckpt_file, map_location='cpu')
    # hnadle module 'DataParallel' prefix
    # if the saved model has 'module.' but your current model doesn't, strip it
    state_dict_model = checkpoint['model']
    new_state_dict = OrderedDict()
    
    for k,v in state_dict_model.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v
        
    # load model weights
    try:
        model.load_state_dict(new_state_dict, strict=False)
        logger.info("Model weights loaded successfully.")
    except RuntimeError as e:
        logger.warning(f"Strict load failed. Trying robust load .... Error: {e}")
        # robust load: only load that match size
        model_dict = model.state_dict()
        valid_dict = {k:v for k,v in new_state_dict.items() if k in model_dict and model_dict[k].size() == v.shape}
        model.load_state_dict(valid_dict, strict=False)
        logger.info(f"Loaded {len(valid_dict)}/{len(model_dict)} layers")
    
    # optimizer and scheduler
    start_epoch = 0
    if not restart_train:
        if optimizer and checkpoint.get("optimizer"):
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info("Optimizer state loaded.")
        if scheduler and checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
            logger.info("Scheduler state loaded.")
        start_epoch = checkpoint.get("epoch", 0)+1
        
    # cleanup gpu memory
    del checkpoint
    del new_state_dict
    torch.cuda.empty_cache()
    
    return start_epoch
    