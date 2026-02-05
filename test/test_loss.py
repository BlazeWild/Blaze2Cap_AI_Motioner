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

from blaze2cap import MotionCorrectionLoss


def test_loss_forward():
    """Test basic loss forward pass."""
    print("=" * 60)
    print("TEST: Loss Forward Pass")
    print("=" * 60)
    
    # Create loss function
    criterion = MotionCorrectionLoss(lambda_rot=1.0, lambda_smooth=5.0)
    
    # Create dummy predictions and targets
    batch_size = 4
    seq_len = 32
    
    # Model outputs: (root_out, body_out)
    root_out = torch.randn(batch_size, seq_len, 2, 6)
    body_out = torch.randn(batch_size, seq_len, 20, 6)
    inputs = (root_out, body_out)
    
    # Ground truth: [B, S, 132] or [B, S, 22, 6]
    targets = torch.randn(batch_size, seq_len, 132)
    
    print(f"\nInputs:")
    print(f"  Root: {root_out.shape}")
    print(f"  Body: {body_out.shape}")
    print(f"Target: {targets.shape}")
    
    # Compute loss
    loss_dict = criterion(inputs, targets)
    
    print(f"\nLoss values:")
    print(f"  Total:    {loss_dict['loss'].item():.6f}")
    print(f"  Rotation: {loss_dict['l_rot'].item():.6f}")
    print(f"  Smooth:   {loss_dict['l_smooth'].item():.6f}")
    
    # Verify all values are valid
    assert not torch.isnan(loss_dict['loss']), "Loss is NaN!"
    assert not torch.isinf(loss_dict['loss']), "Loss is Inf!"
    print("\n✓ Loss values are valid (not NaN/Inf)")


def test_loss_with_mask():
    """Test loss with padding mask."""
    print("\n" + "=" * 60)
    print("TEST: Loss with Padding Mask")
    print("=" * 60)
    
    criterion = MotionCorrectionLoss(lambda_rot=1.0, lambda_smooth=5.0)
    
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
    print(f"  Total:    {loss_dict['loss'].item():.6f}")
    print(f"  Rotation: {loss_dict['l_rot'].item():.6f}")
    print(f"  Smooth:   {loss_dict['l_smooth'].item():.6f}")
    
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
    """Test that gradients flow through loss."""
    print("\n" + "=" * 60)
    print("TEST: Gradient Flow")
    print("=" * 60)
    
    criterion = MotionCorrectionLoss(lambda_rot=1.0, lambda_smooth=5.0)
    
    # Create inputs that require gradients
    root_out = torch.randn(2, 16, 2, 6, requires_grad=True)
    body_out = torch.randn(2, 16, 20, 6, requires_grad=True)
    inputs = (root_out, body_out)
    targets = torch.randn(2, 16, 22, 6)
    
    # Compute loss and backward
    loss_dict = criterion(inputs, targets)
    loss = loss_dict['loss']
    loss.backward()
    
    print(f"\nLoss: {loss.item():.6f}")
    print(f"Root gradient exists: {root_out.grad is not None}")
    print(f"Body gradient exists: {body_out.grad is not None}")
    
    if root_out.grad is not None:
        print(f"Root gradient norm: {root_out.grad.norm().item():.6f}")
    if body_out.grad is not None:
        print(f"Body gradient norm: {body_out.grad.norm().item():.6f}")
    
    assert root_out.grad is not None, "No gradient for root output!"
    assert body_out.grad is not None, "No gradient for body output!"
    
    print("\n✓ Gradients flow correctly")


def test_loss_smoothness_weight():
    """Test that smoothness weight affects loss behavior."""
    print("\n" + "=" * 60)
    print("TEST: Smoothness Weight Effect")
    print("=" * 60)
    
    # Create two loss functions with different smoothness weights
    criterion_low = MotionCorrectionLoss(lambda_rot=1.0, lambda_smooth=0.5)
    criterion_high = MotionCorrectionLoss(lambda_rot=1.0, lambda_smooth=5.0)
    
    # Create jittery predictions (high velocity variance)
    root_out = torch.randn(2, 16, 2, 6)
    body_out = torch.randn(2, 16, 20, 6)
    # Add jitter
    body_out[:, 1::2] += 1.0  # Alternate frames offset
    
    inputs = (root_out, body_out)
    targets = torch.zeros(2, 16, 22, 6)  # Smooth target (all zeros)
    
    loss_low = criterion_low(inputs, targets)
    loss_high = criterion_high(inputs, targets)
    
    print(f"\nJittery input vs smooth target:")
    print(f"\nLow smoothness (λ=0.5):")
    print(f"  Total:  {loss_low['loss'].item():.4f}")
    print(f"  Smooth: {loss_low['l_smooth'].item():.4f}")
    
    print(f"\nHigh smoothness (λ=5.0):")
    print(f"  Total:  {loss_high['loss'].item():.4f}")
    print(f"  Smooth: {loss_high['l_smooth'].item():.4f}")
    
    # High smoothness should penalize jitter more
    if loss_high['loss'] > loss_low['loss']:
        print("\n✓ Higher smoothness weight increases penalty for jittery motion")
    else:
        print("\n⚠ Unexpected: High smoothness loss should be greater")


def main():
    """Run all loss tests."""
    print("\n" + "=" * 60)
    print("BLAZE2CAP LOSS FUNCTION TESTS")
    print("=" * 60)
    
    test_loss_forward()
    test_loss_with_mask()
    test_loss_gradient_flow()
    test_loss_smoothness_weight()
    
    print("\n" + "=" * 60)
    print("✓ All loss tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
