"""
Test Data Loader
================
Validates the PoseSequenceDataset and data loading pipeline.

Usage:
    cd Blaze2Cap_full
    python -m test.test_dataloader
"""

import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader
import numpy as np

from blaze2cap import PoseSequenceDataset


def test_dataset_loading(data_root="./training_dataset_both_in_out", window_size=64):
    """Test dataset initialization and basic loading."""
    print("=" * 60)
    print("TEST: Dataset Loading")
    print("=" * 60)
    
    # Check if data exists
    if not os.path.exists(data_root):
        print(f"[SKIP] Data root not found: {data_root}")
        print("Please update the path or generate the dataset first.")
        return None
    
    json_path = os.path.join(data_root, "dataset_map.json")
    if not os.path.exists(json_path):
        print(f"[SKIP] dataset_map.json not found at {json_path}")
        print("Run: python -m blaze2cap.data.generate_json")
        return None
    
    # Load dataset
    print(f"\nLoading dataset from: {data_root}")
    print(f"Window size: {window_size}")
    
    try:
        dataset = PoseSequenceDataset(data_root, window_size, split="train")
        print(f"✓ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return None
    
    return dataset


def test_sample_shapes(dataset, window_size=64):
    """Test that samples have correct shapes."""
    if dataset is None:
        return
    
    print("\n" + "=" * 60)
    print("TEST: Sample Shapes")
    print("=" * 60)
    
    # Get first sample
    sample = dataset[0]
    
    source = sample["source"]
    mask = sample["mask"]
    target = sample["target"]
    
    print(f"\nSample shapes:")
    print(f"  Source: {source.shape}")
    print(f"  Mask:   {mask.shape}")
    print(f"  Target: {target.shape}")
    
    # Validate shapes
    F = source.shape[0]  # Number of frames
    N = window_size
    
    # Source should be (F, window_size, 450) where 450 = 25 joints * 18 features
    expected_source = (F, N, 450)
    assert source.shape[1] == N, f"Source window mismatch: {source.shape[1]} vs {N}"
    assert source.shape[2] == 450, f"Source features mismatch: {source.shape[2]} vs 450"
    print(f"  ✓ Source shape valid: (F={F}, window={N}, features=450)")
    
    # Mask should be (F, window_size)
    assert mask.shape == (F, N), f"Mask shape mismatch: {mask.shape}"
    print(f"  ✓ Mask shape valid: (F={F}, window={N})")
    
    # Target should be (F, 132) where 132 = 22 joints * 6D
    assert target.shape == (F, 132), f"Target shape mismatch: {target.shape}"
    print(f"  ✓ Target shape valid: (F={F}, features=132)")
    
    # Check dtypes
    print(f"\nData types:")
    print(f"  Source: {source.dtype}")
    print(f"  Mask:   {mask.dtype}")
    print(f"  Target: {target.dtype}")


def test_dataloader_batching(dataset, batch_size=4):
    """Test DataLoader batching works correctly."""
    if dataset is None:
        return
    
    print("\n" + "=" * 60)
    print("TEST: DataLoader Batching")
    print("=" * 60)
    
    # Note: With variable sequence lengths, we need custom collation
    # For now, test with batch_size=1 or use padding
    
    loader = DataLoader(
        dataset,
        batch_size=1,  # Start with 1 to avoid shape mismatches
        shuffle=True,
        num_workers=0
    )
    
    print(f"\nDataLoader created:")
    print(f"  Batches: {len(loader)}")
    print(f"  Batch size: 1 (variable sequence lengths)")
    
    # Get one batch
    batch = next(iter(loader))
    
    print(f"\nBatch shapes (batch_size=1):")
    print(f"  Source: {batch['source'].shape}")
    print(f"  Mask:   {batch['mask'].shape}")
    print(f"  Target: {batch['target'].shape}")
    
    print("\n✓ DataLoader batching works!")


def test_mask_values(dataset):
    """Test that mask values are correct."""
    if dataset is None:
        return
    
    print("\n" + "=" * 60)
    print("TEST: Mask Values")
    print("=" * 60)
    
    sample = dataset[0]
    mask = sample["mask"]
    
    # PyTorch convention: True = padding (ignore), False = valid
    padding_count = mask.sum().item()
    valid_count = (~mask).sum().item()
    total = mask.numel()
    
    print(f"\nMask statistics:")
    print(f"  Total elements: {total}")
    print(f"  Padding (True): {padding_count} ({100*padding_count/total:.1f}%)")
    print(f"  Valid (False):  {valid_count} ({100*valid_count/total:.1f}%)")
    
    # First few frames should have more padding (causal masking)
    first_frame_padding = mask[0].sum().item()
    last_frame_padding = mask[-1].sum().item()
    
    print(f"\n  First frame padding: {first_frame_padding}")
    print(f"  Last frame padding:  {last_frame_padding}")
    
    if first_frame_padding >= last_frame_padding:
        print("  ✓ Causal padding pattern looks correct")


def main():
    """Run all dataloader tests."""
    # Configuration
    DATA_ROOT = "./training_dataset_both_in_out"
    WINDOW_SIZE = 64
    
    print("\n" + "=" * 60)
    print("BLAZE2CAP DATALOADER TESTS")
    print("=" * 60)
    
    # Run tests
    dataset = test_dataset_loading(DATA_ROOT, WINDOW_SIZE)
    test_sample_shapes(dataset, WINDOW_SIZE)
    test_dataloader_batching(dataset)
    test_mask_values(dataset)
    
    print("\n" + "=" * 60)
    print("✓ All dataloader tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
