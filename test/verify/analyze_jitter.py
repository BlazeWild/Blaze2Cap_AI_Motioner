
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# --- CONFIGURATION ---
GT_FILE_PATH = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/gt_final/S1/acting1/cam1/gt_S1_acting1_cam1_seg0_s1_o0.npy"

def cont6d_to_mat(d6):
    d6 = torch.tensor(d6)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * torch.sum(b1 * a2, dim=-1, keepdim=True))
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1).numpy()

def analyze_jitter(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    data = np.load(file_path) # (F, 22, 6)
    
    # 1. Get Hip Delta 6D -> Matrix
    hip_deltas = data[:, 1, :]
    delta_mats = cont6d_to_mat(hip_deltas)
    
    # 2. Accumulate to get Absolute Rotation
    abs_rots = []
    curr = np.eye(3)
    for i in range(len(delta_mats)):
        curr = curr @ delta_mats[i]
        abs_rots.append(curr)
    abs_rots = np.stack(abs_rots)
    
    # 3. Convert to Euler Angles for Plotting
    # Abs Euler
    r_abs = R.from_matrix(abs_rots)
    euler_abs = r_abs.as_euler('xyz', degrees=True)
    
    # Delta Euler
    r_delta = R.from_matrix(delta_mats)
    euler_delta = r_delta.as_euler('xyz', degrees=True)
    
    # 4. Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot Absolute
    axes[0].plot(euler_abs[:, 0], label='X (Pitch)', alpha=0.7)
    axes[0].plot(euler_abs[:, 1], label='Y (Yaw)', alpha=0.7)
    axes[0].plot(euler_abs[:, 2], label='Z (Roll)', alpha=0.7)
    axes[0].set_title("Reconstructed ABSOLUTE Hip Rotation (Euler Degrees)")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot Delta
    axes[1].plot(euler_delta[:, 0], label='Delta X', alpha=0.7)
    axes[1].plot(euler_delta[:, 1], label='Delta Y', alpha=0.7)
    axes[1].plot(euler_delta[:, 2], label='Delta Z', alpha=0.7)
    axes[1].set_title("Frame-to-Frame DELTA Rotation (Euler Degrees)")
    axes[1].set_ylim(-10, 10) # Zoom in
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_jitter(GT_FILE_PATH)
