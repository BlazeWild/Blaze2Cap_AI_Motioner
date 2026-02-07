"""
Test Forward Kinematics with Ground Truth 6D Rotations
Uses offsets from skeleton_config.py to calculate bone lengths and positions
Visualizes skeleton with interactive frame slider
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

# Add parent directories to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..', '..'))

from tools.skeleton_config import OFFSETS_METERS, PARENTS, JOINT_NAMES


def rotation_6d_to_matrix(d6):
    """
    Convert 6D rotation representation to 3x3 rotation matrix using Gram-Schmidt.
    
    Args:
        d6: (6,) numpy array representing first 2 columns of rotation matrix
    Returns:
        R: (3, 3) rotation matrix
    """
    # First two columns of rotation matrix
    col0 = d6[0:3]
    col1 = d6[3:6]
    
    # Gram-Schmidt orthonormalization
    b1 = col0 / (np.linalg.norm(col0) + 1e-8)
    b2 = col1 - np.dot(b1, col1) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    
    # Third column from cross product
    b3 = np.cross(b1, b2)
    
    R = np.stack([b1, b2, b3], axis=1)
    return R


def forward_kinematics(gt_6d, offsets, parents):
    """
    Perform forward kinematics using 6D rotations and offsets.
    
    Args:
        gt_6d: (22, 6) array of 6D rotations
               Joint 0: root position [x, y, z, 0, 0, 0]
               Joint 1: root rotation (6D)
               Joints 2-21: local 6D rotations (parent-relative)
        offsets: (22, 3) tensor of bone offsets in meters
        parents: (22,) array of parent indices
    
    Returns:
        positions: (22, 3) array of 3D joint positions
    """
    num_joints = 22
    positions = np.zeros((num_joints, 3))
    rotations = np.zeros((num_joints, 3, 3))
    
    # Joint 0: Root position (absolute position in world space)
    positions[0] = gt_6d[0, 0:3]
    rotations[0] = np.eye(3)  # Not used, joint 1 has the actual root rotation
    
    # Joint 1: Root rotation (global rotation)
    rotations[1] = rotation_6d_to_matrix(gt_6d[1])
    positions[1] = positions[0]  # Same position as root
    
    # Joints 2-21: Child joints with local (parent-relative) rotations
    for j in range(2, num_joints):
        parent_idx = parents[j]
        
        # Get local rotation from 6D representation
        local_rot = rotation_6d_to_matrix(gt_6d[j])
        
        # Accumulate rotation: global_rot = parent_rot @ local_rot
        rotations[j] = rotations[parent_idx] @ local_rot
        
        # Get offset from config and calculate bone length
        offset = offsets[j].numpy()
        bone_length = np.linalg.norm(offset)
        
        if bone_length > 1e-6:
            # Bone direction from offset
            bone_direction = offset / bone_length
            
            # Apply parent rotation to bone direction
            bone_vector = rotations[parent_idx] @ bone_direction * bone_length
            
            # Position = parent position + rotated bone vector
            positions[j] = positions[parent_idx] + bone_vector
        else:
            # Zero-length bone (e.g., Hips_rot)
            positions[j] = positions[parent_idx]
    
    return positions


def main():
    # Configuration
    gt_file = os.path.join(script_dir, '..', 'dataset', 'Totalcapture_blazepose_preprocessed', 
                          'Dataset', 'gt_augmented', 'S1', 'acting1', 'cam1', 
                          'gt_S1_acting1_cam1_seg0_s1_o0.npy')
    
    print("=" * 80)
    print("Forward Kinematics Test with Ground Truth 6D Rotations")
    print("=" * 80)
    print(f"GT file: {gt_file}")
    print()
    
    # Load ground truth data
    gt_data = np.load(gt_file).astype(np.float32)
    print(f"GT data shape: {gt_data.shape}")
    print(f"  Frames: {gt_data.shape[0]}")
    print(f"  Joints: {gt_data.shape[0]}")
    print(f"  Features: {gt_data.shape[2]} (6D rotation)")
    print()
    
    # Compute FK for all frames
    print("Computing forward kinematics for all frames...")
    all_positions = []
    
    for frame_idx in range(gt_data.shape[0]):
        gt_6d = gt_data[frame_idx]
        positions = forward_kinematics(gt_6d, OFFSETS_METERS, PARENTS)
        all_positions.append(positions)
    
    all_positions = np.array(all_positions)
    print(f"✓ Computed positions shape: {all_positions.shape}")
    print()
    
    # Create interactive plot
    print("Creating interactive visualization...")
    print("Use the slider to navigate through frames!")
    print("=" * 80)
    
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.15)
    
    # Create line objects for each bone connection
    bone_lines = []
    for j in range(1, len(PARENTS)):
        parent = PARENTS[j]
        if parent >= 0:
            line, = ax.plot([], [], [], color='blue', linewidth=2, marker='o', markersize=4)
            bone_lines.append((line, parent, j))
    
    def update_plot(frame_idx):
        frame_idx = int(frame_idx)
        positions = all_positions[frame_idx]
        
        # Update bone lines
        for line, parent, child in bone_lines:
            line.set_data_3d(
                [positions[parent, 0], positions[child, 0]],
                [positions[parent, 1], positions[child, 1]],
                [positions[parent, 2], positions[child, 2]]
            )
        
        ax.set_title(f'Frame {frame_idx}/{len(all_positions)-1} - FK with GT 6D Rotations', 
                    fontsize=14, fontweight='bold')
        
        fig.canvas.draw_idle()
    
    # Set FIXED axis limits: -1 to 1 on all axes
    ax.set_xlabel('X (meters) - Right', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (meters) - Down', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (meters) - Away', fontsize=12, fontweight='bold')
    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    
    ax.grid(True, alpha=0.3)
    
    # Set view angle
    ax.view_init(elev=20, azim=45)
    
    # Initial plot
    update_plot(0)
    
    # Create slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, len(all_positions)-1, 
                    valinit=0, valstep=1, color='blue')
    
    slider.on_changed(update_plot)
    
    plt.show()
    
    print("\nViewer closed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
