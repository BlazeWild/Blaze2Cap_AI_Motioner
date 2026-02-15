
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
blaze2cap_root = os.path.dirname(os.path.dirname(current_dir))
if blaze2cap_root not in sys.path:
    sys.path.insert(0, blaze2cap_root)

from blaze2cap.modules.models import MotionTransformer
from blaze2cap.modules.pose_processing import process_blazepose_frames

# --- CONFIGURATION ---
CHECKPOINT_PATH = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/checkpoints/khjhgjhgjhg.pth"
BLAZE_FILE_PATH = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/blazepose_final/S1/acting1/cam1/blazepose_S1_acting1_cam1_seg0_s1_o0.npy"
GT_FILE_PATH = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/gt_final/S1/acting1/cam1/gt_S1_acting1_cam1_seg0_s1_o0.npy"

WINDOW_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def cont6d_to_mat(d6):
    d6 = torch.tensor(d6) if not isinstance(d6, torch.Tensor) else d6
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * torch.sum(b1 * a2, dim=-1, keepdim=True))
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1).numpy()

def get_angular_magnitude(rot_mats):
    """
    Computes magnitude of rotation angle (0 to 180 deg) from matrices.
    Using trace: 2*cos(theta) + 1 = trace(R)
    """
    # Trace of 3x3: R[0,0] + R[1,1] + R[2,2]
    # Handle NaNs from bad matrices
    trace = rot_mats[..., 0, 0] + rot_mats[..., 1, 1] + rot_mats[..., 2, 2]
    val = (trace - 1.0) / 2.0
    val = np.clip(val, -1.0, 1.0)
    angles_rad = np.arccos(val)
    return np.degrees(angles_rad)

def main():
    print("1. Loading Model...")
    model = MotionTransformer(
        num_joints=28, 
        input_feats=14, 
        num_joints_out=21,
        d_model=512, 
        num_layers=6, 
        n_head=8, 
        d_ff=1024, 
        dropout=0.1
    ).to(DEVICE)
    
    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        model.eval()
        print("   Model loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    print("2. Loading Data...")
    if not os.path.exists(BLAZE_FILE_PATH) or not os.path.exists(GT_FILE_PATH):
        print("Error: Files not found.")
        print(f"Blaze: {os.path.exists(BLAZE_FILE_PATH)}")
        print(f"GT: {os.path.exists(GT_FILE_PATH)}")
        return

    raw_blaze = np.nan_to_num(np.load(BLAZE_FILE_PATH).astype(np.float32))
    gt_data = np.nan_to_num(np.load(GT_FILE_PATH).astype(np.float32))
    
    min_len = min(len(raw_blaze), len(gt_data))
    raw_blaze = raw_blaze[:min_len]
    gt_data = gt_data[:min_len]

    print(f"   Processing {min_len} frames...")
    # Process Inputs
    features, _ = process_blazepose_frames(raw_blaze, WINDOW_SIZE)
    input_tensor = torch.from_numpy(features).to(DEVICE)
    
    # Run Inference
    preds_list = []
    with torch.no_grad():
        for i in range(len(input_tensor)):
            batch = input_tensor[i:i+1] # (1, W, 28, 14)
            out = model(batch)      # (1, W, 21, 6)
            pred_last = out[0, -1]  # (21, 6)
            preds_list.append(pred_last.cpu().numpy())
    
    preds = np.stack(preds_list)    # (F, 21, 6)
    
    # 3. Analyze Angles
    print("3. analyzing Angles...")
    
    # --- HIP ANALYSIS ---
    # GT Index 1 is HipRot (Index 0 is Pos)
    # Pred Index 0 is HipRot
    
    gt_hip_6d = gt_data[:, 1, :]
    pred_hip_6d = preds[:, 0, :]
    
    gt_hip_mat = cont6d_to_mat(gt_hip_6d)
    pred_hip_mat = cont6d_to_mat(pred_hip_6d)
    
    # Convert to Euler for direct component comparison
    r_gt_hip = R.from_matrix(gt_hip_mat)
    r_pred_hip = R.from_matrix(pred_hip_mat)
    
    euler_gt_hip = r_gt_hip.as_euler('xyz', degrees=True)
    euler_pred_hip = r_pred_hip.as_euler('xyz', degrees=True)
    
    # --- BODY ANALYSIS (Average) ---
    # GT Indices 2-21 (20 joints)
    # Pred Indices 1-20 (20 joints)
    
    gt_body_6d = gt_data[:, 2:22, :]
    pred_body_6d = preds[:, 1:21, :]
    
    gt_body_mat = cont6d_to_mat(gt_body_6d)      # (F, 20, 3, 3)
    pred_body_mat = cont6d_to_mat(pred_body_6d)  # (F, 20, 3, 3)
    
    gt_body_mag = get_angular_magnitude(gt_body_mat)     # (F, 20)
    pred_body_mag = get_angular_magnitude(pred_body_mat) # (F, 20)
    
    gt_body_avg = np.mean(gt_body_mag, axis=1)
    pred_body_avg = np.mean(pred_body_mag, axis=1)
    
    # 4. Plot
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))
    
    # Plot 1: Hip Euler X (Pitch) - Often main rotation component
    axes[0].plot(euler_gt_hip[:, 0], label='GT Hip X (Delta)', color='green', alpha=0.7)
    axes[0].plot(euler_pred_hip[:, 0], label='Pred Hip X (Delta)', color='blue', alpha=0.7)
    axes[0].set_title("Hip Orientation DELTA (Euler X Component)")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: Hip Euler Y (Yaw)
    axes[1].plot(euler_gt_hip[:, 1], label='GT Hip Y' , color='green', alpha=0.7)
    axes[1].plot(euler_pred_hip[:, 1], label='Pred Hip Y', color='blue', alpha=0.7)
    axes[1].set_title("Hip Orientation DELTA (Euler Y Component)")
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot 3: Hip Euler Z (Roll)
    axes[2].plot(euler_gt_hip[:, 2], label='GT Hip Z' , color='green', alpha=0.7)
    axes[2].plot(euler_pred_hip[:, 2], label='Pred Hip Z', color='blue', alpha=0.7)
    axes[2].set_title("Hip Orientation DELTA (Euler Z Component)")
    axes[2].legend()
    axes[2].grid(True)
    
    # Plot 4: Average Body Joint Rotation Magnitude
    axes[3].plot(gt_body_avg, label='GT Body Avg Mag', color='green', alpha=0.7)
    axes[3].plot(pred_body_avg, label='Pred Body Avg Mag', color='blue', alpha=0.7)
    axes[3].set_title("Average Body Joint Rotation Magnitude (Degrees)")
    axes[3].legend()
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
