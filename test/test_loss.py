"""
Test Loss Functions
==================
Validates the MotionCorrectionLoss and its components.

Usage:
    cd Blaze2Cap_full
    python -m test.test_loss
"""

import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
from blaze2cap.modeling.loss import MotionCorrectionLoss

def test_loss_forward():
    """Test basic loss forward pass."""
    print("=" * 60)
    print("TEST: Loss Forward Pass")
    print("=" * 60)
    
    # Create loss function
    criterion = MotionCorrectionLoss(
        lambda_root_vel=1.0,
        lambda_root_rot=1.0,
        lambda_pose_rot=1.0,
        lambda_pose_pos=1.0,
        lambda_smooth=2.0,
        lambda_accel=5.0
    )
    
    # Create dummy predictions and targets
    batch_size = 4
    seq_len = 32
    
    # Model outputs: (root_out, body_out)
    root_out = torch.randn(batch_size, seq_len, 2, 6)
    body_out = torch.randn(batch_size, seq_len, 20, 6)
    inputs = (root_out, body_out)
    
    # Ground truth: [B, S, 132] or [B, S, 22, 6]
    targets = torch.randn(batch_size, seq_len, 22, 6)
    
    print(f"\nInputs:")
    print(f"  Root: {root_out.shape}")
    print(f"  Body: {body_out.shape}")
    print(f"Target: {targets.shape}")
    
    # Compute loss
    loss_dict = criterion(inputs, targets)
    
    print(f"\nLoss values:")
    print(f"  Total:      {loss_dict['loss'].item():.6f}")
    print(f"  Root Vel:   {loss_dict['l_root_vel'].item():.6f}")
    print(f"  Root Rot:   {loss_dict['l_root_rot'].item():.6f}")
    print(f"  Pose Rot:   {loss_dict['l_pose_rot'].item():.6f}")
    print(f"  Pose Pos:   {loss_dict['l_pose_pos'].item():.6f} (MPJPE)")
    print(f"  Smooth:     {loss_dict['l_smooth'].item():.6f}")
    print(f"  Accel:      {loss_dict['l_accel'].item():.6f}")
    
    # Verify all values are valid
    for k, v in loss_dict.items():
        assert not torch.isnan(v), f"{k} is NaN!"
        assert not torch.isinf(v), f"{k} is Inf!"
        
    print("\n✓ Loss values are valid (not NaN/Inf)")


def test_loss_with_mask():
    """Test loss with padding mask."""
    print("\n" + "=" * 60)
    print("TEST: Loss with Padding Mask")
    print("=" * 60)
    
    criterion = MotionCorrectionLoss()
    
    batch_size = 4
    seq_len = 32
    
    root_out = torch.randn(batch_size, seq_len, 2, 6)
    body_out = torch.randn(batch_size, seq_len, 20, 6)
    inputs = (root_out, body_out)
    targets = torch.randn(batch_size, seq_len, 22, 6)
    
    # Create mask: True = padding (ignore)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask[:, :8] = True  # First 8 frames are padding
    
    print(f"\nMask: {mask.shape}")
    print(f"  Padding frames: {mask.sum().item()} / {mask.numel()}")
    
    # Compute loss with mask
    loss_dict = criterion(inputs, targets, mask=mask)
    
    print(f"\nLoss with mask:")
    print(f"  Total: {loss_dict['loss'].item():.6f}")
    
    # Compare with unmasked
    loss_unmasked = criterion(inputs, targets)
    
    print(f"\nLoss without mask:")
    print(f"  Total: {loss_unmasked['loss'].item():.6f}")
    
    # They should be different
    if abs(loss_dict['loss'].item() - loss_unmasked['loss'].item()) > 1e-6:
        print("\n✓ Mask affects loss calculation correctly")
    else:
        print("\n⚠ Warning: Mask may not be applied correctly")


def test_loss_gradient_flow():
    """Test that gradients flow through all components."""
    print("\n" + "=" * 60)
    print("TEST: Gradient Flow")
    print("=" * 60)
    
    criterion = MotionCorrectionLoss(
        lambda_root_vel=1.0,
        lambda_root_rot=1.0,
        lambda_pose_rot=1.0,
        lambda_pose_pos=1.0, # This checks MPJPE differentiation
        lambda_smooth=1.0,
        lambda_accel=1.0
    )
    
    # Create inputs that require gradients
    root_out = torch.randn(2, 16, 2, 6, requires_grad=True)
    body_out = torch.randn(2, 16, 20, 6, requires_grad=True)
    inputs = (root_out, body_out)
    targets = torch.randn(2, 16, 22, 6)
    
    # Compute loss
    loss_dict = criterion(inputs, targets)
    loss = loss_dict['loss']
    
    # Backward
    loss.backward()
    
    print(f"\nLoss: {loss.item():.6f}")
    
    # Check Gradients
    print(f"Root gradient exists: {root_out.grad is not None}")
    print(f"Body gradient exists: {body_out.grad is not None}")
    
    root_grad_norm = root_out.grad.norm().item()
    body_grad_norm = body_out.grad.norm().item()
    
    print(f"Root gradient norm: {root_grad_norm:.6f}")
    print(f"Body gradient norm: {body_grad_norm:.6f}")
    
    assert root_out.grad is not None, "No gradient for root!"
    assert body_out.grad is not None, "No gradient for body!"
    assert root_grad_norm > 0, "Root gradient is zero!"
    assert body_grad_norm > 0, "Body gradient is zero!"
    
    print("\n✓ Gradients flow correctly through all components (including FK)")


def main():
    """Run all loss tests."""
    print("\n" + "=" * 60)
    print("BLAZE2CAP LOSS FUNCTION TESTS (HYBRID + MPJPE)")
    print("=" * 60)
    
    test_loss_forward()
    test_loss_with_mask()
    test_loss_gradient_flow()
    
    print("\n" + "=" * 60)
    print("✓ All loss tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
