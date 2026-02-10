import torch
import numpy as np
from blaze2cap.modeling.loss import MotionCorrectionLoss
from blaze2cap.utils.skeleton_config import get_raw_skeleton

def test_loss_nan():
    print("--- Testing MotionCorrectionLoss for NaNs ---")
    criterion = MotionCorrectionLoss()
    
    # Create dummy inputs
    B, S = 2, 5
    # Zero input to trigger potential divide-by-zero in normalization
    # Root: (B, S, 2, 6) -> All zeros
    pred_root = torch.zeros((B, S, 2, 6), requires_grad=True)
    # Body: (B, S, 20, 6) -> All zeros
    pred_body = torch.zeros((B, S, 20, 6), requires_grad=True)
    
    # Targets
    tgt = torch.zeros((B, S, 22, 6))
    
    # Forward
    try:
        loss_dict = criterion((pred_root, pred_body), tgt)
        loss = loss_dict["loss"]
        loss.backward()
        print(f"Loss value: {loss.item()}")
        if torch.isnan(loss):
            print("FAILURE: Loss is NaN!")
        else:
            print("SUCCESS: Loss is finite.")
    except Exception as e:
        print(f"FAILURE: Exception during loss calculation: {e}")

def test_skeleton_extraction():
    print("\n--- Testing Dynamic Skeleton Extraction ---")
    # path = '/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/blazepose_augmented/S1/acting1/cam1/blaze_S1_acting1_cam1_seg0_s1_o0.npy'
    # Use explicit path relative to CWD
    path = "blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/blazepose_augmented/S1/acting1/cam1/blaze_S1_acting1_cam1_seg0_s1_o0.npy"
    
    try:
        data = np.load(path)
        # (F, 25, 7)
        raw_pos = data[:, :, :3]
        
        # Test extraction
        offsets = get_raw_skeleton(raw_pos)
        
        print("Offsets shape:", offsets.shape)
        print("Mean bone length:", torch.norm(offsets, dim=1).mean().item())
        
        # Check integrity
        if torch.isnan(offsets).any():
             print("FAILURE: Offsets contain NaNs")
        else:
             print("SUCCESS: Offsets extracted.")
            
    except Exception as e:
        print(f"FAILURE: Could not load data or extract: {e}")

if __name__ == "__main__":
    test_loss_nan()
    test_skeleton_extraction()
