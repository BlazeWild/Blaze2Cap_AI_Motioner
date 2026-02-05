"""
Evaluate Trained Model
======================
Load a trained checkpoint and evaluate on the test set.

Usage:
    cd Blaze2Cap_full
    python -m test.evaluate --checkpoint ./checkpoints/best_model.pth
"""

import os
import sys
import argparse

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from blaze2cap import (
    MotionTransformer,
    PoseSequenceDataset,
    load_checkpoint,
    evaluate_motion
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Blaze2Cap Model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file (.pth)")
    parser.add_argument("--data_root", type=str, default="./training_dataset_both_in_out",
                        help="Path to dataset root")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--window_size", type=int, default=64,
                        help="Window size for data loading")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto/cuda/cpu)")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to evaluate (train/test)")
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get config from checkpoint if available
    config = checkpoint.get('config', {})
    
    # Initialize model with default or saved config
    model = MotionTransformer(
        num_joints=config.get('num_joints', 25),
        input_feats=config.get('input_feats', 18),
        d_model=config.get('d_model', 256),
        num_layers=config.get('num_layers', 4),
        n_head=config.get('n_head', 4),
        d_ff=config.get('d_ff', 512),
        dropout=0.0,  # No dropout during evaluation
        max_len=config.get('max_len', 512)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model'], strict=False)
    model = model.to(device)
    model.eval()
    
    # Print checkpoint info
    epoch = checkpoint.get('epoch', 'unknown')
    metrics = checkpoint.get('metrics', {})
    print(f"  Epoch: {epoch}")
    if metrics:
        print(f"  Saved metrics: {metrics}")
    
    return model


def evaluate(model, loader, device):
    """Run evaluation on dataset."""
    all_preds = []
    all_targets = []
    
    print(f"\nEvaluating on {len(loader)} batches...")
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            src = batch["source"].to(device)
            mask = batch["mask"].to(device)
            tgt = batch["target"].to(device)
            
            # Forward pass
            pred = model.forward_combined(src, key_padding_mask=mask)
            
            # Reshape target if needed
            if tgt.dim() == 3 and tgt.shape[-1] == 132:
                tgt = tgt.view(tgt.shape[0], tgt.shape[1], 22, 6)
            
            all_preds.append(pred.cpu())
            all_targets.append(tgt.cpu())
    
    # Concatenate
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    print(f"\nTotal samples: {preds.shape[0]}")
    print(f"Prediction shape: {preds.shape}")
    print(f"Target shape: {targets.shape}")
    
    # Calculate metrics
    # Using default skeleton config (should be replaced with real values)
    skeleton_config = {
        'parents': [0, 0, 1, 2, 3, 4, 5, 6, 5, 8, 9, 10, 5, 12, 13, 14, 0, 16, 17, 0, 19, 20],
        'offsets': torch.ones(22, 3) * 0.1
    }
    
    metrics = evaluate_motion(preds, targets, skeleton_config)
    
    return metrics, preds, targets


def main():
    args = parse_args()
    
    # Device setup
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("BLAZE2CAP MODEL EVALUATION")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data root: {args.data_root}")
    print(f"Split: {args.split}")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    dataset = PoseSequenceDataset(args.data_root, args.window_size, split=args.split)
    
    loader = DataLoader(
        dataset,
        batch_size=1,  # Variable sequence lengths
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate
    metrics, preds, targets = evaluate(model, loader, device)
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  MPJPE: {metrics['MPJPE']:.6f}")
    print(f"  MARE:  {metrics['MARE']:.6f}")
    print("=" * 60)
    
    return metrics


if __name__ == "__main__":
    main()
