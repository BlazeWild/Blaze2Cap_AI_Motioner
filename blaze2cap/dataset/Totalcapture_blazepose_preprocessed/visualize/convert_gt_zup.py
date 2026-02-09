"""
Script 1: Converter
===================
Loads original Y-Down .npy file.
Applies Y-Down -> Z-Up transformation to:
1. Root Velocity
2. Root Rotation
3. Body Rotations
Saves as new _zup.npy file.
"""

import numpy as np
import torch
import torch.nn.functional as F
import os

# --- FILE PATHS ---
INPUT_FILE = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/gtfinal/S1/acting1/cam1/gt_S1_acting1_cam1_seg0_s1_o0.npy" # <--- EDIT THIS
OUTPUT_FILE = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/blaze2cap/dataset/Totalcapture_blazepose_preprocessed/visualize/S1_acting1_cam1_converted_zup.npy"
def cont6d_to_mat(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = a2 - (b1 * torch.sum(b1 * a2, dim=-1, keepdim=True))
    b2 = F.normalize(b2, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)

def mat_to_cont6d(mat):
    return torch.cat([mat[..., 0], mat[..., 1]], dim=-1)

def main():
    print(f"Processing: {INPUT_FILE}")
    raw_data = np.load(INPUT_FILE)
    data = torch.from_numpy(raw_data).float()
    
    # Reshape
    original_len = data.shape[0]
    if data.shape[-1] == 132:
        data = data.view(original_len, 22, 6)
        
    # --- YOUR TRANSFORMATION LOGIC ---
    # 1. Define Matrix M
    M = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0]
    ]).float()
    
    # 2. Transform Velocity (Index 0)
    # logic from your script: root_vel_local = data[:, 0, :3] * -1.0
    # logic from your script: root_vel_local = torch.matmul(root_vel_local, COORD_TRANSFORM_T)
    vel = data[:, 0, :3] * -1.0
    vel = torch.matmul(vel, M.T)
    
    # 3. Transform Rotations (Index 1-21)
    # logic from your script: all_rot_mats = torch.matmul(M, torch.matmul(all_rot_mats, MT))
    rot_6d = data[:, 1:, :] 
    rot_mat = cont6d_to_mat(rot_6d)
    
    M_exp = M.view(1, 1, 3, 3)
    MT_exp = M.T.view(1, 1, 3, 3)
    
    rot_mat = torch.matmul(M_exp, torch.matmul(rot_mat, MT_exp))
    
    # Convert back to 6D to save
    rot_6d_new = mat_to_cont6d(rot_mat)
    
    # 4. Save
    vel_6d = torch.zeros(original_len, 1, 6)
    vel_6d[:, 0, :3] = vel
    
    final_data = torch.cat([vel_6d, rot_6d_new], dim=1)
    final_data = final_data.view(original_len, 132) # Flatten
    
    np.save(OUTPUT_FILE, final_data.numpy())
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()