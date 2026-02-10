"""
Plot Screen X, Y positions (ACTUAL, not reconstructed from deltas)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os

# CONFIG
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "processed_features_27_18.npy")

# Feature channel indices
SCREEN_POS_START = 12  # Channels 12-13 (actual centered screen coords)

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

def main():
    print(f"Loading: {INPUT_FILE}")
    data = np.load(INPUT_FILE)
    print(f"Data shape: {data.shape}")
    
    # Extract actual screen positions (channels 12-13)
    positions = data[:, :, SCREEN_POS_START:SCREEN_POS_START+2]  # (F, 27, 2)
    
    F, J, _ = positions.shape
    print(f"Screen positions shape: {positions.shape}")
    
    # Setup plot with aspect ratio matching data range (X: ~3.6, Y: ~2.0 -> ratio 1.8:1)
    fig, ax = plt.subplots(figsize=(16, 9))  # 16:9 matches the aspect ratio
    
    # Scatter plot for joints
    scat = ax.scatter([], [], c='red', s=50, alpha=0.8, zorder=3)
    
    # Lines for skeleton bones
    lines = [ax.plot([], [], 'b-', linewidth=2)[0] for _ in BONES_VISUAL]
    
    # Trajectory line (root joint)
    traj, = ax.plot([], [], 'green', linewidth=2, alpha=0.7, label='Root trajectory', zorder=2)
    
    # Set axis limits to show full data range (X: -1.8 to 1.8, Y: -1 to 1 with margins)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.2, 1.2)
    ax.invert_yaxis()  # Invert Y so +Y goes down (screen coordinates)
    ax.set_xlabel('Screen X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Screen Y', fontsize=12, fontweight='bold')
    ax.set_title('Screen Positions (Actual - Centered)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Info text
    info = fig.text(0.02, 0.95, "", fontfamily='monospace', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    # Frame slider
    slider_ax = plt.axes([0.3, 0.02, 0.6, 0.03])
    slider = Slider(slider_ax, 'Frame', 0, F-1, valstep=1, valfmt='%d', facecolor='lightblue')
    
    def update(val):
        f = int(slider.val)
        pos = positions[f]  # (27, 2)
        
        # Update scatter plot
        scat.set_offsets(pos)
        
        # Update skeleton bones
        for line, (p1, p2) in zip(lines, BONES_VISUAL):
            line.set_data(
                [pos[p1, 0], pos[p2, 0]],
                [pos[p1, 1], pos[p2, 1]]
            )
        
        # Update trajectory (root joint - index 26)
        if f > 0:
            path = positions[:f+1, 26, :]  # MidHip trajectory
            traj.set_data(path[:, 0], path[:, 1])
        
        # Update info
        root_pos = pos[26]  # MidHip
        info.set_text(
            f"Frame: {f}/{F-1}\n"
            f"Root Screen Position:\n"
            f"  X: {root_pos[0]:6.3f}\n"
            f"  Y: {root_pos[1]:6.3f}"
        )
        
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    update(0)
    plt.show()

if __name__ == "__main__":
    main()
