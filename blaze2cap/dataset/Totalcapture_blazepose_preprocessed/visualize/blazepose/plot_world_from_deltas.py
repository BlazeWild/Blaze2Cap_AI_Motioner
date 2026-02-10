"""
Plot World X, Y, Z positions reconstructed from deltas (velocities)
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import os

# CONFIG
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "processed_features_27_18.npy")

# Feature channel indices
WORLD_POS_START = 0   # Channels 0-2 (not used, we use deltas)
WORLD_VEL_START = 3   # Channels 3-5 (delta_world)

# Joint indices
I_NOSE = 0
I_L_EAR = 1;   I_R_EAR = 2
I_L_SH = 3;    I_R_SH = 4
I_L_ELB = 5;   I_R_ELB = 6
I_L_WRIST = 7; I_R_WRIST = 8
I_L_PINKY = 9; I_R_PINKY = 10
I_L_INDEX = 11; I_R_INDEX = 12
I_L_THUMB = 13; I_R_THUMB = 14
I_L_HIP = 15;  I_R_HIP = 16
I_L_KNEE = 17; I_R_KNEE = 18
I_L_ANKLE = 19; I_R_ANKLE = 20
I_L_HEEL = 21; I_R_HEEL = 22
I_L_FOOT = 23; I_R_FOOT = 24
I_NECK = 25
I_MIDHIP = 26

# Bone connections for skeleton visualization
BONES_VISUAL = [
    # Spine (MidHip <-> Neck)
    (I_MIDHIP, I_NECK),
    # Pelvis hub
    (I_MIDHIP, I_L_HIP), (I_MIDHIP, I_R_HIP),
    # Shoulder hub
    (I_NECK, I_L_SH), (I_NECK, I_R_SH),
    # Arms (Left)
    (3, 5), (5, 7), (7, 9), (7, 11), (7, 13),
    # Arms (Right)
    (4, 6), (6, 8), (8, 10), (8, 12), (8, 14),
    # Legs (Left)
    (15, 17), (17, 19), (19, 23), (19, 21),
    # Legs (Right)
    (16, 18), (18, 20), (20, 24), (20, 22),
    # Face
    (0, 1), (0, 2),
]

def reconstruct_from_deltas(data):
    """
    Reconstruct world positions from velocity deltas.
    Input: (Frames, 27, 18)
    Output: (Frames, 27, 3) - Reconstructed world positions
    """
    F, J, _ = data.shape
    
    # Extract actual positions (channels 0-2) - centered world coords
    actual_positions = data[:, :, WORLD_POS_START:WORLD_POS_START+3]  # (F, 27, 3)
    
    # Extract velocities (channels 3-5)
    velocities = data[:, :, WORLD_VEL_START:WORLD_VEL_START+3]  # (F, 27, 3)
    
    # Extract anchor flags (channel 17)
    anchor_flags = data[:, :, 17]  # (F, 27)
    
    # Initialize positions array
    positions = np.zeros((F, J, 3), dtype=np.float32)
    
    # Accumulate velocities for each joint
    for j in range(J):
        # Start from actual frame 0 position
        positions[0, j] = actual_positions[0, j]
        
        for f in range(1, F):
            # If anchor flag is 0, reset to actual position (new sequence start)
            if anchor_flags[f, j] == 0:
                positions[f, j] = actual_positions[f, j]
            else:
                # Accumulate velocity: pos[t] = pos[t-1] + vel[t]
                positions[f, j] = positions[f-1, j] + velocities[f, j]
    
    return positions

def main():
    print(f"Loading: {INPUT_FILE}")
    data = np.load(INPUT_FILE)
    print(f"Data shape: {data.shape}")
    
    # Reconstruct positions from deltas
    print("Reconstructing world positions from velocity deltas...")
    positions = reconstruct_from_deltas(data)
    
    F, J, _ = positions.shape
    print(f"Reconstructed positions shape: {positions.shape}")
    
    # Setup plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot for joints
    scat = ax.scatter([], [], [], c='red', s=30, alpha=0.8)
    
    # Lines for skeleton bones
    lines = [ax.plot([], [], [], 'b-', linewidth=2)[0] for _ in BONES_VISUAL]
    
    # Trajectory line (root joint)
    traj, = ax.plot([], [], [], 'green', linewidth=2, alpha=0.7, label='Root trajectory')
    
    # Set fixed axes limits [-1, 1]
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.invert_yaxis()  # Invert Y so +Y goes down (screen coordinates)
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z', fontsize=12, fontweight='bold')
    ax.set_title('World Positions (Reconstructed from Deltas)', fontsize=14, fontweight='bold')
    ax.legend()
    
    # Info text
    info = fig.text(0.02, 0.95, "", fontfamily='monospace', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    # Frame slider
    slider_ax = plt.axes([0.3, 0.02, 0.6, 0.03])
    slider = Slider(slider_ax, 'Frame', 0, F-1, valstep=1, valfmt='%d', facecolor='lightblue')
    
    def update(val):
        f = int(slider.val)
        pos = positions[f]  # (27, 3)
        
        # Update scatter plot
        scat._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        
        # Update skeleton bones
        for line, (p1, p2) in zip(lines, BONES_VISUAL):
            line.set_data_3d(
                [pos[p1, 0], pos[p2, 0]],
                [pos[p1, 1], pos[p2, 1]],
                [pos[p1, 2], pos[p2, 2]]
            )
        
        # Update trajectory (root joint - index 26)
        if f > 0:
            path = positions[:f+1, 26, :]  # MidHip trajectory
            traj.set_data(path[:, 0], path[:, 1])
            traj.set_3d_properties(path[:, 2])
        
        # Update info
        root_pos = pos[26]  # MidHip
        info.set_text(
            f"Frame: {f}/{F-1}\n"
            f"Root Position:\n"
            f"  X: {root_pos[0]:6.3f}\n"
            f"  Y: {root_pos[1]:6.3f}\n"
            f"  Z: {root_pos[2]:6.3f}"
        )
        
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    update(0)
    plt.show()

if __name__ == "__main__":
    main()
