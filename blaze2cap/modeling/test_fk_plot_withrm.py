"""
GT Motion Visualizer (Fixed Scale World)
========================================
Visualizes Ground Truth motion with:
- Fixed Scale (-2m to +2m)
- Fixed Floor (Z=0)
- Global Trajectory Line
- NO Camera Follow (Static World View)
"""

import os
import sys
import random
import glob
# from cairo import Filter
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# --- Project Imports ---
sys.path.append(str(Path(__file__).resolve().parents[2])) 
from blaze2cap.utils.skeleton_config import get_totalcapture_skeleton

# --- Configuration ---
DATASET_ROOT = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/gt_final"

# Filter Options
FILTER_SUBJECT = "S1"
FILTER_ACTION = "acting1"
FILTER_CAM = "cam1"

# FILTER_SUBJECT = None
# FILTER_ACTION = None
# FILTER_CAM = None
def cont6d_to_mat(d6):
    """Converts 6D rotation representation to 3x3 rotation matrix."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = a2 - (b1 * torch.sum(b1 * a2, dim=-1, keepdim=True))
    b2 = F.normalize(b2, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)

def reconstruct_global_motion(data_npy, parents, offsets):
    """
    Applies Coordinate Transform FIRST, then calculates FK.
    """
    data = torch.from_numpy(data_npy).float()
    F_frames = data.shape[0]
    
    if data.shape[-1] == 132:
        data = data.view(F_frames, 22, 6)

    # ==========================================
    # 1. DEFINE COORDINATE TRANSFORM (Y-Down -> Z-Up)
    # ==========================================
    # We want:
    # X_new = X_old
    # Y_new = Z_old  (Forward)
    # Z_new = -Y_old (Up)
    
    COORD_TRANSFORM = torch.tensor([
        [1.0, 0.0, 0.0],  # X -> X
        [0.0, 0.0, -1.0],  # Y -> Z
        [0.0, -1.0, 0.0]  # Z -> -Y
    ]).float()
    
    # Inverse is needed for rotation similarity transform (M @ R @ M.T)
    COORD_TRANSFORM_T = COORD_TRANSFORM.T

    # ==========================================
    # 2. TRANSFORM INPUTS (BEFORE CALCULATION)
    # ==========================================
    
    # A. Transform Offsets (Bone Lengths)
    # [22, 3] -> [22, 3]
    offsets = torch.matmul(offsets, COORD_TRANSFORM_T) # Vector mult is v @ M.T

    # B. Transform Root Velocity
    root_vel_local = data[:, 0, :3] * -1.0 
    # [F, 3] -> [F, 3]
    root_vel_local = torch.matmul(root_vel_local, COORD_TRANSFORM_T)

    # C. Transform Rotations (Change of Basis)
    # We need to transform EVERY rotation matrix in the file.
    # Formula: R_new = M @ R_old @ M.T
    
    # 1. Get all rotations (Root + Body)
    all_rot_data = data[:, 1:, :] # [F, 21, 6]
    all_rot_mats = cont6d_to_mat(all_rot_data) # [F, 21, 3, 3]
    
    # 2. Apply Transform
    # Expand M for broadcasting: [1, 1, 3, 3]
    M = COORD_TRANSFORM.view(1, 1, 3, 3)
    MT = COORD_TRANSFORM_T.view(1, 1, 3, 3)
    
    # Perform M @ R @ MT
    all_rot_mats = torch.matmul(M, torch.matmul(all_rot_mats, MT))
    
    # Split back into Root Rot and Body Rots
    root_rot_delta_mat = all_rot_mats[:, 0] # Index 0 is Root
    body_rot_mats = all_rot_mats[:, 1:]     # Index 1..20 is Body

    # ==========================================
    # 3. YOUR CUSTOM MATRICES (Optional Tweaks)
    # ==========================================
    # If you still want to flip the path or pose additionally, do it here.
    # Currently set to Identity since the COORD_TRANSFORM handles the axes.
    
    # ... (Add extra transforms here if needed) ...

    # ==========================================
    # 4. FK ACCUMULATION LOOP (Now in Z-Up Space)
    # ==========================================
    curr_root_pos = torch.zeros(3)
    curr_root_rot = torch.eye(3) # Identity start
    # Apply initial coordinate transform to root rotation if needed?
    # Usually starting at Identity in world space is fine.
    
    root_positions = []
    root_orientations = []
    
    for f in range(F_frames):
        # Update Pos
        vel_step_world = torch.matmul(curr_root_rot, root_vel_local[f])
        curr_root_pos = curr_root_pos + vel_step_world
        
        # Update Rot
        curr_root_rot = torch.matmul(curr_root_rot, root_rot_delta_mat[f])
        
        root_positions.append(curr_root_pos.clone())
        root_orientations.append(curr_root_rot.clone())
        
    root_positions = torch.stack(root_positions)       
    root_orientations = torch.stack(root_orientations) 

    # ==========================================
    # 5. BODY LOOP
    # ==========================================
    all_global_pos = []
    
    for f in range(F_frames):
        frame_global_pos = [root_positions[f], root_positions[f]]
        frame_global_rots = [root_orientations[f], root_orientations[f]]
        
        for i in range(2, 22):
            parent_idx = parents[i]
            offset = offsets[i] # Already Z-Up
            
            parent_R = frame_global_rots[parent_idx]
            parent_P = frame_global_pos[parent_idx]
            local_R = body_rot_mats[f, i-2] # Already Z-Up
            
            global_R = torch.matmul(parent_R, local_R)
            rotated_offset = torch.matmul(parent_R, offset)
            global_P = parent_P + rotated_offset
            
            frame_global_pos.append(global_P)
            frame_global_rots.append(global_R)
            
        all_global_pos.append(torch.stack(frame_global_pos))

    return torch.stack(all_global_pos).numpy()

def find_file():
    subj = FILTER_SUBJECT if FILTER_SUBJECT else "*"
    act = FILTER_ACTION if FILTER_ACTION else "*"
    cam = FILTER_CAM if FILTER_CAM else "*"
    pattern = os.path.join(DATASET_ROOT, subj, act, cam, "*.npy")
    files = glob.glob(pattern)
    if not files: raise ValueError(f"No files: {pattern}")
    return random.choice(files)

def main():
    skel_cfg = get_totalcapture_skeleton()
    parents = skel_cfg['parents']
    offsets = skel_cfg['offsets'] # [22, 3]
    
    filepath = find_file()
    print(f"Loading: {filepath}")
    raw_data = np.load(filepath)
    
    # Returns data that is ALREADY Z-Up
    positions = reconstruct_global_motion(raw_data, parents, offsets)
    num_frames = positions.shape[0]
    
    # Plot Setup
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(f"File: {os.path.basename(filepath)}", fontsize=12)
    plt.subplots_adjust(bottom=0.25)
    
    ax = fig.add_subplot(111, projection='3d')
    
    # Info Text
    info_text = fig.text(0.05, 0.90, "", fontsize=10, fontfamily='monospace', verticalalignment='top')
    
    scat = ax.scatter([], [], [], c='r', s=15)
    lines = [ax.plot([], [], [], 'b-', lw=2)[0] for _ in range(1, 22)]
    traj_line, = ax.plot([], [], [], 'g--', lw=1)
    
    # Standard Z-Up Limits
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_zlim(-1.0, 2.0)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, num_frames-1, valinit=0, valstep=1)

    def update_view(val):
        frame = int(slider.val)
        current_pose = positions[frame] 
        
        # --- PLOTTING IS NOW RAW (X=X, Y=Y, Z=Z) ---
        xs = current_pose[:, 0]
        ys = current_pose[:, 1]
        zs = current_pose[:, 2]
        
        scat._offsets3d = (xs, ys, zs)
        
        for i, line in enumerate(lines):
            child = i + 1
            parent = parents[child]
            p1 = current_pose[parent]
            p2 = current_pose[child]
            
            line.set_data([p1[0], p2[0]], [p1[1], p2[1]]) 
            line.set_3d_properties([p1[2], p2[2]])
            
        history = positions[:frame+1, 0, :]
        traj_line.set_data(history[:, 0], history[:, 1])
        traj_line.set_3d_properties(history[:, 2])
        
        # Info Text
        root_p = current_pose[0]
        info_str = (f"Frame: {frame}\n"
                    f"Root Pos (Z-Up):\n"
                    f"  X: {root_p[0]:.3f}\n"
                    f"  Y: {root_p[1]:.3f}\n"
                    f"  Z: {root_p[2]:.3f}")
        info_text.set_text(info_str)
        
        fig.canvas.draw_idle()

    slider.on_changed(update_view)
    update_view(0)
    plt.show()

if __name__ == "__main__":
    main()