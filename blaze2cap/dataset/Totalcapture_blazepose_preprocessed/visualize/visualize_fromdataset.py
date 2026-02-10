"""
Script: Visualizer for Random GT Files from Dataset
===================================================
Randomly selects and visualizes GT files from the gt_final dataset folder
using skeleton configuration from skeleton_config.py
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import math
import os
import random
import sys
import glob

# Add path to import skeleton config  
utils_path = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/blaze2cap/utils"
sys.path.append(utils_path)
from skeleton_config import PARENTS, OFFSETS_METERS, JOINT_NAMES

# --- CONFIG ---
GT_FINAL_DIR = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/gt_final"



FILTER_SUBJECT = "S1"
FILTER_ACTION = "acting1"
FILTER_CAM = "cam1"


# FILTER_SUBJECT = "S1"
# FILTER_ACTION = "acting1"
# FILTER_CAM = "cam1"

def get_random_gt_file():
    """Randomly select a GT file from the dataset using filters"""
    subj = FILTER_SUBJECT if FILTER_SUBJECT else "*"
    act = FILTER_ACTION if FILTER_ACTION else "*"
    cam = FILTER_CAM if FILTER_CAM else "*"
    
    pattern = os.path.join(GT_FINAL_DIR, subj, act, cam, "*.npy")
    files = glob.glob(pattern)
    
    if not files:
        raise ValueError(f"No files found matching pattern: {pattern}")
    
    selected_file = random.choice(files)
    print(f"Randomly selected: {os.path.relpath(selected_file, GT_FINAL_DIR)}")
    return selected_file

def rotation_matrix_to_euler(R):
    """Convert rotation matrix to Euler angles in degrees"""
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    if sy < 1e-6:
        x, y, z = math.atan2(-R[1, 2], R[1, 1]), math.atan2(-R[2, 0], sy), 0
    else:
        x, y, z = math.atan2(R[2, 1], R[2, 2]), math.atan2(-R[2, 0], sy), math.atan2(R[1, 0], R[0, 0])
    return np.degrees([x, y, z])

def cont6d_to_mat(d6):
    """Convert 6D continuous representation to rotation matrices"""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = a2 - (b1 * torch.sum(b1 * a2, dim=-1, keepdim=True))
    b2 = F.normalize(b2, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)

def run_fk_from_gt(data_npy):
    """Forward kinematics - RAW, no transformations (already done in preprocessing)"""
    data = torch.from_numpy(data_npy).float()
    F_frames = data.shape[0]
    
    # Handle different input formats
    if data.shape[-1] == 132: 
        data = data.view(F_frames, 22, 6)
    # Data is already (frames, 22, 6) format - no frame index to remove

    # --- 1. GET DATA (RAW - Already Transformed) ---
    root_vel = data[:, 0, :3]  # Index 0: Root Velocity
    rot_data = data[:, 1:, :]  # Index 1-21: Rotations
    
    rot_mats = cont6d_to_mat(rot_data)
    root_deltas = rot_mats[:, 0]
    body_rots = rot_mats[:, 1:]

    # --- 2. ROOT LOOP (ACCUMULATE MOTION) ---
    curr_pos = torch.zeros(3)
    curr_rot = torch.eye(3)
    
    root_pos_list = []
    root_rot_list = []
    
    for f in range(F_frames):
        # 1. Rotate Velocity from Local to World using current facing
        step = torch.matmul(curr_rot, root_vel[f])
        curr_pos += step
        
        # 2. Update Rotation
        curr_rot = torch.matmul(curr_rot, root_deltas[f])
        
        root_pos_list.append(curr_pos.clone())
        root_rot_list.append(curr_rot.clone())

    # --- 3. BODY LOOP ---
    all_poses = []
    for f in range(F_frames):
        g_pos = [root_pos_list[f], root_pos_list[f]]
        g_rot = [root_rot_list[f], root_rot_list[f]]
        
        for i in range(2, 22):
            pid = PARENTS[i]
            off = OFFSETS_METERS[i]
            
            p_rot = g_rot[pid]
            p_pos = g_pos[pid]
            l_rot = body_rots[f, i-2]
            
            g_rot.append(torch.matmul(p_rot, l_rot))
            g_pos.append(p_pos + torch.matmul(p_rot, off))
            
        all_poses.append(torch.stack(g_pos))
        
    return (torch.stack(all_poses).numpy(), 
            np.stack(root_rot_list), 
            root_deltas.numpy(),
            root_vel.numpy())

def main():
    """Main visualization function"""
    try:
        # Randomly select a GT file
        gt_file = get_random_gt_file()
        
        print(f"Loading: {gt_file}")
        
        # Load and process data
        data = np.load(gt_file)
        pos, glb_rot, del_rot, vel = run_fk_from_gt(data)
        
        # Setup matplotlib
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Info text panel (Left side)
        info = fig.text(0.02, 0.5, "", fontfamily='monospace', fontsize=9, 
                       verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        # File info panel (Top left)
        file_info = fig.text(0.02, 0.95, f"File: {os.path.basename(gt_file)}", 
                            fontfamily='monospace', fontsize=10, 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        # Initialize plot elements
        scat = ax.scatter([],[],[], c='red', s=20)
        lines = [ax.plot([],[],[], 'blue', linewidth=2)[0] for _ in range(21)]
        traj, = ax.plot([],[],[], 'green', linestyle='--', alpha=0.7, linewidth=2)
        
        # Set axis limits and labels
        ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_zlim(-1, 2)
        ax.set_xlabel('X', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z', fontsize=12, fontweight='bold')
        ax.set_title('GT Motion Data Visualization (Z-Up, Transformed)', fontsize=14, fontweight='bold')
        
        # Frame slider
        slider_ax = plt.axes([0.3, 0.02, 0.6, 0.03])
        slider = Slider(slider_ax, 'Frame', 0, len(data)-1, valstep=1, 
                       valfmt='%d', facecolor='lightblue')
        
        def update(val):
            f = int(slider.val)
            p = pos[f]
            
            # Update joint positions
            scat._offsets3d = (p[:,0], p[:,1], p[:,2])
            
            # Update skeleton bones
            for i, line in enumerate(lines):
                child_idx = i + 1  # Joint indices 1..21
                parent_idx = PARENTS[child_idx]
                
                if parent_idx >= 0:  # Valid parent
                    p1 = p[parent_idx]
                    p2 = p[child_idx]
                    
                    line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
                    line.set_3d_properties([p1[2], p2[2]])
            
            # Update trajectory (path of root joint up to current frame)
            if f > 0:
                path = pos[:f+1, 0, :]  # Root joint (index 0)
                traj.set_data(path[:, 0], path[:, 1])
                traj.set_3d_properties(path[:, 2])
            
            # Update info panel
            gr, gp, gy = rotation_matrix_to_euler(glb_rot[f])
            v = vel[f]
            
            info_text = (
                f"Frame: {f}/{len(data)-1}\n\n"
                f"ROOT POSITION:\n"
                f"  X: {p[0,0]:7.3f} m\n"
                f"  Y: {p[0,1]:7.3f} m\n"
                f"  Z: {p[0,2]:7.3f} m\n\n"
                f"ROOT VELOCITY:\n"
                f"  X: {v[0]:7.3f} m/s\n"
                f"  Y: {v[1]:7.3f} m/s\n"
                f"  Z: {v[2]:7.3f} m/s\n\n"
                f"ROOT ORIENTATION:\n"
                f"  Roll:  {gr:6.1f}°\n"
                f"  Pitch: {gp:6.1f}°\n"
                f"  Yaw:   {gy:6.1f}°\n\n"
                f"SKELETON CONFIG:\n"
                f"  Joints: {len(JOINT_NAMES)}\n"
                f"  Bones:  {len(lines)}"
            )
            
            info.set_text(info_text)
            fig.canvas.draw_idle()
        
        # Connect slider to update function
        slider.on_changed(update)
        
        # Initial update
        update(0)
        
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    print("GT Dataset Visualizer")
    print("=" * 50)
    print(f"Searching for GT files in: {GT_FINAL_DIR}")
    print(f"Filters: Subject={FILTER_SUBJECT or 'ANY'}, Action={FILTER_ACTION or 'ANY'}, Camera={FILTER_CAM or 'ANY'}")
    print("=" * 50)
    main()
