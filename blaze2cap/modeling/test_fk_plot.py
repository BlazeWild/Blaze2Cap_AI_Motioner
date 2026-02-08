"""
GT Motion Visualizer
====================
Visualizes Ground Truth motion files (.npy) by applying Forward Kinematics.
Usage: python -m tools.visualize_gt
"""

import os
import sys
from pathlib import Path
# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import random
import glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
# Import 3D axes
from mpl_toolkits.mplot3d import Axes3D

# --- Project Imports ---
# Make sure your project root is in PYTHONPATH
from blaze2cap.utils.skeleton_config import get_totalcapture_skeleton

# --- Configuration ---
DATASET_ROOT = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/gt_augmented"

# Filter Options (Set to None to pick random)
FILTER_SUBJECT = None   # e.g., "S1", "S2"
FILTER_ACTION = None # e.g., "walking1", "acting1"
FILTER_CAM = None       # e.g., "cam1"

def cont6d_to_mat(d6):
    """
    Converts 6D rotation representation to 3x3 rotation matrix.
    Args:
        d6: Tensor [..., 6]
    Returns:
        Tensor [..., 3, 3]
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = a2 - (b1 * torch.sum(b1 * a2, dim=-1, keepdim=True))
    b2 = F.normalize(b2, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)

def forward_kinematics(data_npy, parents, offsets):
    """
    Reconstructs 3D positions from the .npy motion data.
    
    Args:
        data_npy: Numpy array [F, 132] (flattened) or [F, 22, 6]
        parents: List of parent indices
        offsets: Tensor [22, 3] of bone lengths
    
    Returns:
        positions: Numpy array [F, 22, 3]
    """
    # 1. Convert to Tensor
    data = torch.from_numpy(data_npy).float()
    F_frames = data.shape[0]
    
    # 2. Reshape if flattened: [F, 132] -> [F, 22, 6]
    if data.shape[-1] == 132:
        data = data.view(F_frames, 22, 6)
        
    # 3. Extract Components
    # Index 0: Root Velocity (Not used for pose reconstruction in local frame)
    # Index 1: Root Rotation
    # Index 2-21: Body Rotations
    
    # We reconstruct in a "Root-Centered" frame (Hips at 0,0,0)
    # So we ignore the root velocity/position data in Index 0.
    
    # Combine Root Rot + Body Rot
    # Shape: [F, 21, 6] -> But we need 22 joints worth of rotations for the loop?
    # No, the loop logic:
    # Joint 0 (Hips): Fixed rotation is Identity (since we want local frame) 
    # OR we use the actual Root Rotation from Index 1?
    # -> Usually, to see the pose correctly, we apply Index 1 (Root Rot) to the Hips.
    
    # Let's slice all 6D data: [F, 22, 6]
    # CAUTION: Index 0 is Linear Velocity, NOT a rotation.
    # So we take Index 1..21 (21 rotations).
    rot_data = data[:, 1:, :] # [F, 21, 6]
    
    # Convert to Matrices
    rot_mats = cont6d_to_mat(rot_data) # [F, 21, 3, 3]
    
    # 4. FK Loop
    global_pos = []
    global_rots = []
    
    # Frame-by-frame processing (could be vectorized but loop is clearer)
    # But wait, we can vectorize across frames [F] easily.
    
    # Initialize Root (Joint 0)
    # Position: (0,0,0)
    root_pos = torch.zeros((F_frames, 3))
    
    # Rotation: Identity? Or Index 1?
    # If we want to visualize the "facing direction", we use Index 1.
    # Rot Index 0 in 'rot_mats' corresponds to Joint 1 (Hips Rot) from the config?
    # In skeleton_config: Joint 1 is "Hips_rot".
    # So rot_mats[:, 0] is the rotation for Hips.
    root_rot = rot_mats[:, 0] # [F, 3, 3]
    
    global_pos.append(root_pos)
    global_rots.append(root_rot)
    
    # Iterate joints 1 to 21
    # Note: joint indices in PARENTS list match 0..21
    # rot_mats indices: 0 corresponds to Joint 1, 1 to Joint 2...
    
    for i in range(1, 22):
        parent_idx = parents[i]
        offset = offsets[i].view(1, 3, 1) # [1, 3, 1] for broadcasting
        
        # Parent Global Transforms
        parent_rot = global_rots[parent_idx] # [F, 3, 3]
        parent_p = global_pos[parent_idx]    # [F, 3]
        
        # Local Rotation for current joint
        # Map: Joint i uses Rotation (i-1) from our sliced array
        local_rot = rot_mats[:, i-1]         # [F, 3, 3]
        
        # Global Rot: R_parent @ R_local
        curr_rot = torch.matmul(parent_rot, local_rot)
        global_rots.append(curr_rot)
        
        # Global Pos: P_parent + (R_parent @ Offset)
        rotated_offset = torch.matmul(parent_rot, offset).squeeze(-1) # [F, 3]
        curr_p = parent_p + rotated_offset
        global_pos.append(curr_p)
        
    # Stack: [F, 22, 3]
    return torch.stack(global_pos, dim=1).numpy()

def find_file():
    """Finds a random .npy file matching filters."""
    # Construct search pattern
    # Structure: .../S1/acting1/cam1/gt_....npy
    
    subj = FILTER_SUBJECT if FILTER_SUBJECT else "*"
    act = FILTER_ACTION if FILTER_ACTION else "*"
    cam = FILTER_CAM if FILTER_CAM else "*"
    
    pattern = os.path.join(DATASET_ROOT, subj, act, cam, "*.npy")
    files = glob.glob(pattern)
    
    if not files:
        raise ValueError(f"No files found matching: {pattern}")
        
    return random.choice(files)

def update_plot(frame_idx, pos_data, scat, lines, parents, ax):
    """Animation update function."""
    current_pose = pos_data[frame_idx] # [22, 3]
    
    # 1. Update Scatter (Joints)
    # Matplotlib 3D expects (xs, ys, zs)
    # Coordinate System of TotalCapture (from config):
    # X=Right, Y=Down, Z=Forward
    # To plot correctly in Matplotlib (Z=Up):
    # Map Y(Down) -> -Z
    # Map Z(Fwd) -> Y
    xs = current_pose[:, 0]
    ys = current_pose[:, 2] 
    zs = -current_pose[:, 1]
    
    scat._offsets3d = (xs, ys, zs)
    
    # 2. Update Lines (Bones)
    for i, (line, parent_idx) in enumerate(zip(lines, parents[1:])):
        # Joint index is i+1 (since we skipped root)
        child_idx = i + 1
        
        p1 = current_pose[parent_idx]
        p2 = current_pose[child_idx]
        
        line.set_data([p1[0], p2[0]], [p1[2], p2[2]]) # X, Y(mapped from Z)
        line.set_3d_properties([-p1[1], -p2[1]])      # Z(mapped from -Y)
        
    ax.set_title(f"Frame: {frame_idx}")
    return scat, lines

def main():
    # 1. Load Skeleton Config
    skel_cfg = get_totalcapture_skeleton()
    parents = skel_cfg['parents']
    offsets = skel_cfg['offsets'] # Tensor
    
    # 2. Load File
    try:
        filepath = find_file()
        print(f"Visualizing: {filepath}")
    except ValueError as e:
        print(e)
        return

    raw_data = np.load(filepath)
    # 3. Compute FK
    # positions: [F, 22, 3]
    positions = forward_kinematics(raw_data, parents, offsets)
    num_frames = positions.shape[0]
    
    # 4. Setup Plot
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(f"File: {os.path.basename(filepath)}", fontsize=12)
    # Adjust subplot to make room for slider
    plt.subplots_adjust(bottom=0.25)
    
    ax = fig.add_subplot(111, projection='3d')
    
    # Set Fixed Scale (-1 to 1 meter approx)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    
    ax.set_xlabel('X (Lateral)')
    ax.set_ylabel('Y (Depth/Fwd)') # Mapped from Z
    ax.set_zlabel('Z (Up/Vertical)') # Mapped from -Y
    
    # Initial Plot
    scat = ax.scatter([], [], [], c='r', s=20)
    
    # Create line objects for bones
    lines = []
    # Loop over joints 1..21 (skipping root which has parent -1)
    for _ in range(1, 22):
        line, = ax.plot([], [], [], 'b-', lw=2)
        lines.append(line)
        
    # Draw first frame
    update_plot(0, positions, scat, lines, parents, ax)
    
    # Add Slider
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(
        ax=ax_slider,
        label='Frame',
        valmin=0,
        valmax=num_frames - 1,
        valinit=0,
        valstep=1
    )
    
    def update(val):
        frame_idx = int(slider.val)
        update_plot(frame_idx, positions, scat, lines, parents, ax)
        fig.canvas.draw_idle()
        
    slider.on_changed(update)
    
    plt.show()

if __name__ == "__main__":
    main()