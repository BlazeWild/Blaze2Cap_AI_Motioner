"""
Test Model Forward Pass
=======================
Validates the MotionTransformer model architecture and output shapes.

Usage:
    cd Blaze2Cap_full
    python -m test.test_model
"""

import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np

from blaze2cap import MotionTransformer


def test_model_forward():
    """Test basic forward pass with expected input/output shapes."""
    print("=" * 60)
    print("TEST: Model Forward Pass")
    print("=" * 60)
    
    # Config
    batch_size = 4
    seq_len = 32
    num_joints = 25
    input_feats = 18
    
    # Create model
    model = MotionTransformer(
        num_joints=num_joints,
        input_feats=input_feats,
        d_model=256,
        num_layers=4,
        n_head=4,
        d_ff=512,
        dropout=0.1,
        max_len=512
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,}")
    
    # Create dummy input
    # Shape: [Batch, Seq, Joints, Features] or [Batch, Seq, Joints*Features]
    x_4d = torch.randn(batch_size, seq_len, num_joints, input_feats)
    x_3d = x_4d.view(batch_size, seq_len, -1)  # Flattened
    
    print(f"\nInput 4D shape: {x_4d.shape}")
    print(f"Input 3D shape: {x_3d.shape}")
    
    # Test forward (split heads)
    model.eval()
    with torch.no_grad():
        root_out, body_out = model(x_4d)
        print(f"\nOutput (split heads):")
        print(f"  Root: {root_out.shape} (expected: [4, 32, 2, 6])")
        print(f"  Body: {body_out.shape} (expected: [4, 32, 20, 6])")
        
        assert root_out.shape == (batch_size, seq_len, 2, 6), f"Root shape mismatch!"
        assert body_out.shape == (batch_size, seq_len, 20, 6), f"Body shape mismatch!"
        print("  ✓ Split head shapes correct!")
        
        # Test combined forward
        combined = model.forward_combined(x_3d)
        print(f"\nOutput (combined): {combined.shape} (expected: [4, 32, 22, 6])")
        assert combined.shape == (batch_size, seq_len, 22, 6), f"Combined shape mismatch!"
        print("  ✓ Combined shape correct!")
    
    # Test with padding mask
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask[:, :5] = True  # First 5 frames are padding
    
    with torch.no_grad():
        root_out, body_out = model(x_4d, key_padding_mask=mask)
        print(f"\nWith padding mask: ✓ Runs without error")
    
    print("\n" + "=" * 60)
    print("✓ All model tests passed!")
    print("=" * 60)


def test_model_gpu():
    """Test model on GPU if available."""
    if not torch.cuda.is_available():
        print("\n[SKIP] GPU test - CUDA not available")
        return
    
    print("\n" + "=" * 60)
    print("TEST: GPU Forward Pass")
    print("=" * 60)
    
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model = MotionTransformer().to(device)
    x = torch.randn(8, 64, 25, 18).to(device)
    
    with torch.no_grad():
        output = model.forward_combined(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    print(f"Output device: {output.device}")
    print("✓ GPU forward pass successful!")


def test_model_mixed_precision():
    """Test model with mixed precision (FP16)."""
    if not torch.cuda.is_available():
        print("\n[SKIP] AMP test - CUDA not available")
        return
    
    print("\n" + "=" * 60)
    print("TEST: Mixed Precision (AMP)")
    print("=" * 60)
    
    device = torch.device("cuda")
    model = MotionTransformer().to(device)
    x = torch.randn(8, 64, 25, 18).to(device)
    
    from torch.cuda.amp import autocast
    
    with autocast():
        with torch.no_grad():
            output = model.forward_combined(x)
    
    print(f"Output dtype: {output.dtype}")
    print("✓ Mixed precision forward pass successful!")


if __name__ == "__main__":
    test_model_forward()
    test_model_gpu()
    test_model_mixed_precision()
