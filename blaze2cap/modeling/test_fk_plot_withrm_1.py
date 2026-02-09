"""
GT Motion Visualizer (Z-Up Pre-Transform + Rotation Info)
=========================================================
1. Loads Y-Down Data.
2. Transforms everything to Z-Up immediately.
3. Displays Orientation ONLY (Global + Delta) - Position removed.
"""

import os
import sys
import random
import glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import math

# --- Project Imports ---
sys.path.append(str(Path(__file__).resolve().parents[2])) 
from blaze2cap.utils.skeleton_config import get_totalcapture_skeleton

# --- Configuration ---
DATASET_ROOT = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/gt_augmented"

FILTER_SUBJECT = "S1"
FILTER_ACTION = "acting1"
FILTER_CAM = "cam1"

def rotation_matrix_to_euler(R):
    """
    Converts a 3x3 Rotation Matrix to Euler Angles (Roll, Pitch, Yaw).
    Assumes Z-Up coordinate system (Yaw is around Z).
    Returns (Roll, Pitch, Yaw) in degrees.
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.degrees([x, y, z]) # Returns [Roll(X), Pitch(Y), Yaw(Z)]

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
    Returns: positions, global_rot_mats, delta_rot_mats
    """
    data = torch.from_numpy(data_npy).float()
    F_frames = data.shape[0]
    
    if data.shape[-1] == 132:
        data = data.view(F_frames, 22, 6)

    # ==========================================
    # 1. DEFINE COORDINATE TRANSFORM (Y-Down -> Z-Up)
    # ==========================================
    COORD_TRANSFORM = torch.tensor([
        [1.0, 0.0, 0.0],   # X -> X
        [0.0, 0.0, -1.0],  # Y -> Z
        [0.0, -1.0, 0.0]   # Z -> -Y
    ]).float()
    
    COORD_TRANSFORM_T = COORD_TRANSFORM.T

    # ==========================================
    # 2. TRANSFORM INPUTS
    # ==========================================
    
    # A. Offsets
    offsets = torch.matmul(offsets, COORD_TRANSFORM_T)

    # B. Root Velocity
    root_vel_local = data[:, 0, :3] * -1.0 
    root_vel_local = torch.matmul(root_vel_local, COORD_TRANSFORM_T)

    # C. Rotations
    all_rot_data = data[:, 1:, :] # [F, 21, 6]
    all_rot_mats = cont6d_to_mat(all_rot_data) # [F, 21, 3, 3]
    
    # Apply Change of Basis: M @ R @ MT
    M = COORD_TRANSFORM.view(1, 1, 3, 3)
    MT = COORD_TRANSFORM_T.view(1, 1, 3, 3)
    all_rot_mats = torch.matmul(M, torch.matmul(all_rot_mats, MT))
    
    root_rot_delta_mat = all_rot_mats[:, 0] # Index 0 is Root Delta
    body_rot_mats = all_rot_mats[:, 1:]     # Index 1..20 is Body

    # ==========================================
    # 3. FK ACCUMULATION LOOP
    # ==========================================
    curr_root_pos = torch.zeros(3)
    curr_root_rot = torch.eye(3) 
    
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
    # 4. BODY LOOP
    # ==========================================
    all_global_pos = []
    
    for f in range(F_frames):
        frame_global_pos = [root_positions[f], root_positions[f]]
        frame_global_rots = [root_orientations[f], root_orientations[f]]
        
        for i in range(2, 22):
            parent_idx = parents[i]
            offset = offsets[i] 
            
            parent_R = frame_global_rots[parent_idx]
            parent_P = frame_global_pos[parent_idx]
            local_R = body_rot_mats[f, i-2] 
            
            global_R = torch.matmul(parent_R, local_R)
            rotated_offset = torch.matmul(parent_R, offset)
            global_P = parent_P + rotated_offset
            
            frame_global_pos.append(global_P)
            frame_global_rots.append(global_R)
            
        all_global_pos.append(torch.stack(frame_global_pos))

    # Return Pos, Global Rotations, and Delta Rotations
    return (torch.stack(all_global_pos).numpy(), 
            root_orientations.numpy(), 
            root_rot_delta_mat.numpy())

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
    offsets = skel_cfg['offsets']
    
    filepath = find_file()
    print(f"Loading: {filepath}")
    raw_data = np.load(filepath)
    
    # Unpack returned values
    positions, global_rots, delta_rots = reconstruct_global_motion(raw_data, parents, offsets)
    num_frames = positions.shape[0]
    
    # Plot Setup
    fig = plt.figure(figsize=(12, 8)) # Wider figure for text
    fig.suptitle(f"File: {os.path.basename(filepath)}", fontsize=12)
    plt.subplots_adjust(bottom=0.25, left=0.30) # Make room on left for text
    
    ax = fig.add_subplot(111, projection='3d')
    
    # --- INFO TEXT ---
    # Placed on the left side of the figure
    info_text = fig.text(0.02, 0.5, "", fontsize=11, fontfamily='monospace', verticalalignment='center')
    
    scat = ax.scatter([], [], [], c='r', s=15)
    lines = [ax.plot([], [], [], 'b-', lw=2)[0] for _ in range(1, 22)]
    traj_line, = ax.plot([], [], [], 'g--', lw=1)
    
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_zlim(-1.0, 2.0)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax_slider = plt.axes([0.30, 0.1, 0.60, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, num_frames-1, valinit=0, valstep=1)

    def update_view(val):
        frame = int(slider.val)
        current_pose = positions[frame] 
        
        # Plotting (Already Z-Up)
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
        
        # --- Update Info Text ---
        # Get Angles
        g_r, g_p, g_y = rotation_matrix_to_euler(global_rots[frame])
        d_r, d_p, d_y = rotation_matrix_to_euler(delta_rots[frame])

        info_str = (
            f"Frame: {frame}\n\n"
            f"HIP GLOBAL ORIENTATION:\n"
            f"  Roll : {g_r:.1f}°\n"
            f"  Pitch: {g_p:.1f}°\n"
            f"  Yaw  : {g_y:.1f}°\n\n"
            f"HIP DELTA ORIENTATION:\n"
            f"  Roll : {d_r:.3f}°\n"
            f"  Pitch: {d_p:.3f}°\n"
            f"  Yaw  : {d_y:.3f}°"
        )
        info_text.set_text(info_str)
        
        fig.canvas.draw_idle()

    slider.on_changed(update_view)
    update_view(0)
    plt.show()

if __name__ == "__main__":
    main()