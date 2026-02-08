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
DATASET_ROOT = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/gt_augmented"

# Filter Options
# FILTER_SUBJECT = "S1"
# FILTER_ACTION = "acting1"
# FILTER_CAM = "cam1"

FILTER_SUBJECT = None
FILTER_ACTION = None
FILTER_CAM = None

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
    Reconstructs 3D positions by accumulating root deltas and applying FK.
    """
    data = torch.from_numpy(data_npy).float()
    F_frames = data.shape[0]
    
    if data.shape[-1] == 132:
        data = data.view(F_frames, 22, 6)

    # --- PART A: Reconstruct Root Trajectory ---
    # Velocity is often stored inverted or needs inversion depending on coordinate system
    root_vel_local = data[:, 0, :3] * -1.0 # [F, 3] Fix for reversed trajectory
    root_rot_delta_mat = cont6d_to_mat(data[:, 1, :]) # [F, 3, 3]
    
    curr_root_pos = torch.zeros(3)
    curr_root_rot = torch.eye(3)
    
    root_positions = []
    root_orientations = []
    
    for f in range(F_frames):
        # Update Pos: P_new = P_old + (R_old @ V_local)
        vel_step_world = torch.matmul(curr_root_rot, root_vel_local[f])
        curr_root_pos = curr_root_pos + vel_step_world
        
        # Update Rot: R_new = R_old @ R_delta
        curr_root_rot = torch.matmul(curr_root_rot, root_rot_delta_mat[f])
        
        root_positions.append(curr_root_pos.clone())
        root_orientations.append(curr_root_rot.clone())
        
    root_positions = torch.stack(root_positions)       
    root_orientations = torch.stack(root_orientations) 

    # --- PART B: Forward Kinematics (Body) ---
    body_rot_mats = cont6d_to_mat(data[:, 2:, :]) # [F, 20, 3, 3]
    
    all_global_pos = []
    
    for f in range(F_frames):
        # Init with Root (Joint 0) and Hips_Rot (Joint 1)
        frame_global_pos = [root_positions[f], root_positions[f]]
        frame_global_rots = [root_orientations[f], root_orientations[f]]
        
        # Loop Body Joints (Indices 2 to 21)
        for i in range(2, 22):
            parent_idx = parents[i]
            offset = offsets[i].view(3)
            
            parent_R = frame_global_rots[parent_idx]
            parent_P = frame_global_pos[parent_idx]
            local_R = body_rot_mats[f, i-2] 
            
            global_R = torch.matmul(parent_R, local_R)
            rotated_offset = torch.matmul(parent_R, offset)
            global_P = parent_P + rotated_offset
            
            frame_global_pos.append(global_P)
            frame_global_rots.append(global_R)
            
        all_global_pos.append(torch.stack(frame_global_pos))

    return torch.stack(all_global_pos).numpy() # [F, 22, 3]

def find_file():
    subj = FILTER_SUBJECT if FILTER_SUBJECT else "*"
    act = FILTER_ACTION if FILTER_ACTION else "*"
    cam = FILTER_CAM if FILTER_CAM else "*"
    pattern = os.path.join(DATASET_ROOT, subj, act, cam, "*.npy")
    files = glob.glob(pattern)
    if not files: raise ValueError(f"No files: {pattern}")
    return random.choice(files)

def main():
    # 1. Setup
    skel_cfg = get_totalcapture_skeleton()
    parents = skel_cfg['parents']
    offsets = skel_cfg['offsets']
    
    filepath = find_file()
    print(f"Loading: {filepath}")
    raw_data = np.load(filepath)
    
    positions = reconstruct_global_motion(raw_data, parents, offsets)
    num_frames = positions.shape[0]
    
    # 2. Plot Setup
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(f"File: {os.path.basename(filepath)}", fontsize=12)
    plt.subplots_adjust(bottom=0.2)
    
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize Objects
    scat = ax.scatter([], [], [], c='r', s=15)
    lines = [ax.plot([], [], [], 'b-', lw=2)[0] for _ in range(1, 22)]
    traj_line, = ax.plot([], [], [], 'g--', lw=1)
    
    # --- STATIC AXIS LIMITS ---
    # Lock the world view to 4x4 meters centered at origin
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1, 1.0) # Z is Up (0 to 2m)
    
    ax.set_xlabel('X (Right)')
    ax.set_ylabel('Y (Forward)')
    ax.set_zlabel('Z (Up)')

    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, num_frames-1, valinit=0, valstep=1)

    def update_view(val):
        frame = int(slider.val)
        current_pose = positions[frame]
        
        # Mapping: Data(Y-Down) -> Plot(Z-Up)
        xs = current_pose[:, 0]
        ys = current_pose[:, 2]  # Z becomes Y (Forward)
        zs = -current_pose[:, 1] # -Y becomes Z (Up)
        
        # Update Body
        scat._offsets3d = (xs, ys, zs)
        for i, line in enumerate(lines):
            child = i + 1
            parent = parents[child]
            p1 = current_pose[parent]
            p2 = current_pose[child]
            
            line.set_data([p1[0], p2[0]], [p1[2], p2[2]]) 
            line.set_3d_properties([-p1[1], -p2[1]])
            
        # Update Trajectory
        history = positions[:frame+1, 0, :]
        traj_line.set_data(history[:, 0], history[:, 2])
        traj_line.set_3d_properties(-history[:, 1])
        
        ax.set_title(f"Frame {frame} / {num_frames}")
        fig.canvas.draw_idle()

    slider.on_changed(update_view)
    update_view(0)
    plt.show()

if __name__ == "__main__":
    main()