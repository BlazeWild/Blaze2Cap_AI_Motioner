"""
Plot skeleton using bone lengths from skeleton_config.py
Hips position at (0, 0, 0)
Bone lengths from indices 2-21 (Spine through LeftFoot)
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blaze2cap.utils.skeleton_config import OFFSETS_METERS, PARENTS, JOINT_NAMES


def plot_skeleton_with_bone_lengths():
    """
    Plot skeleton in T-pose using bone lengths.
    Hips at (0, 0, 0), bone lengths applied from index 2 onwards.
    """
    num_joints = 22
    positions = np.zeros((num_joints, 3))
    
    # Joint 0: Hips_pos at origin
    positions[0] = np.array([0.0, 0.0, 0.0])
    
    # Joint 1: Hips_rot (same as Hips_pos)
    positions[1] = positions[0]
    
    # Joints 2-21: Calculate positions using offsets from skeleton_config.py
    # Offsets are already transformed to the correct coordinate system
    for j in range(2, num_joints):
        parent_idx = PARENTS[j]
        
        # Get offset directly from config
        offset = OFFSETS_METERS[j].numpy()
        
        # Calculate position
        positions[j] = positions[parent_idx] + offset
    
    # Convert inches coordinate system to meters for visualization
    # BVH uses: X=lateral, Y=vertical, Z=depth
    # We'll keep the same for visualization
    
    # Create plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot bones
    for j in range(1, num_joints):
        parent = PARENTS[j]
        if parent >= 0:
            # Plot bone as line
            ax.plot([positions[parent, 0], positions[j, 0]],
                   [positions[parent, 1], positions[j, 1]],
                   [positions[parent, 2], positions[j, 2]],
                   color='blue', linewidth=2, marker='o', markersize=4)
    
    # Plot joint labels
    for j in range(num_joints):
        ax.text(positions[j, 0], positions[j, 1], positions[j, 2],
               f'{j}', fontsize=8, color='red')
    
    # Set labels and limits
    ax.set_xlabel('X (meters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (meters)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (meters)', fontsize=12, fontweight='bold')
    
    # Fixed axis limits: -1 to 1 on all axes
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    
    ax.set_title('Skeleton with Bone Lengths (X=Right, Y=Down, Z=Away)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set view angle
    ax.view_init(elev=20, azim=45)
    
    print("=" * 60)
    print("Skeleton Offsets Visualization")
    print("Coordinate System: X=Right, Y=Down, Z=Away from camera")
    print("=" * 60)
    print(f"Hips position: {positions[0]}")
    print(f"Number of joints: {num_joints}")
    print(f"Using transformed offsets in meters from skeleton_config.py")
    print(f"\nFixed axis bounds: [-1, 1] on all axes")
    print("=" * 60)
    
    plt.show()


if __name__ == "__main__":
    plot_skeleton_with_bone_lengths()
