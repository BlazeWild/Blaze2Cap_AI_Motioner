import numpy as np
import matplotlib
# Force TkAgg for interactive plots (rotation/zooming)
try:
    matplotlib.use('TkAgg')
except ImportError:
    pass
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import torch
import os

# ==========================================
# CONFIGURATION
# ==========================================
# 22 Joint Hierarchy
PARENTS = [-1, 0, 1, 2, 3, 4, 5, 6, 5, 8, 9, 10, 5, 12, 13, 14, 1, 16, 17, 1, 19, 20]

# Standard Offsets (Meters) - Explicitly defined for standalone running
OFFSETS = np.array([
    [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
    [0.0, 0.0462, -0.0692], [0.0, 0.0160, -0.0909], [0.0, 0.0080, -0.0920], [0.0, 0.0, -0.0923],
    [0.0, -0.0346, -0.2424], [0.0, -0.0221, -0.1250],
    [0.0292, -0.0486, -0.1576], [0.1449, 0.0, 0.0], [0.2889, 0.0, 0.0], [0.2196, 0.0, 0.0],
    [-0.0292, -0.0486, -0.1576], [-0.1449, 0.0, 0.0], [-0.2889, 0.0, 0.0], [-0.2196, 0.0, 0.0],
    [0.0866, 0.0, 0.0253], [0.0, 0.0, 0.3789], [0.0, 0.0, 0.3754],
    [-0.0866, 0.0, 0.0253], [0.0, 0.0, 0.3789], [0.0, 0.0, 0.3754]
], dtype=np.float32)

# File to Visualize
INFERENCE_FILE = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/test/inference_test/pred_blaze_S1_acting1_cam1_seg0_s1_o0.npy"

# ==========================================
# MATH UTILS
# ==========================================
def rotation_6d_to_matrix(r6d):
    """(..., 6) -> (..., 3, 3)"""
    x_raw = r6d[..., 0:3]
    y_raw = r6d[..., 3:6]
    x = x_raw / (np.linalg.norm(x_raw, axis=-1, keepdims=True) + 1e-8)
    z = np.cross(x, y_raw)
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    y = np.cross(z, x)
    return np.stack((x, y, z), axis=-1)

def compute_fk_from_deltas(data, offsets):
    """
    Reconstruct Absolute Motion from Deltas.
    """
    num_frames = data.shape[0]
    
    # 1. Initialize Root State (0,0,0 & Identity)
    curr_root_pos = np.array([0.0, 0.0, 0.0])
    curr_root_rot = np.eye(3)
    
    # Storage
    all_global_pos = np.zeros((num_frames, 22, 3))
    all_global_rot = np.zeros((num_frames, 22, 3, 3))
    
    # Pre-convert body rotations to matrices for speed
    # Data[2:] are local rotations
    body_rots_6d = data[:, 2:, :] 
    body_rots_mat = rotation_6d_to_matrix(body_rots_6d)

    for f in range(num_frames):
        # --- Root Update (Accumulate Deltas) ---
        if f > 0:
            # Velocity & Rotation Delta
            vel_local = data[f, 0, :3]
            rot_delta_6d = data[f, 1, :]
            R_delta = rotation_6d_to_matrix(rot_delta_6d)
            
            # Update Position: P_curr = P_prev + (R_prev @ v_local)
            # Note: Using CURRENT rotation for velocity projection if that's how it was trained, 
            # but standard is usually Previous. Sticking to Previous for safety.
            # However, if your model predicts velocity in the NEW frame, swap this.
            # Based on previous chats: v_local = R_prev.T @ (p_new - p_old) -> p_new = p_old + R_prev @ v_local
            
            curr_root_pos = curr_root_pos + (curr_root_rot @ vel_local)
            curr_root_rot = curr_root_rot @ R_delta

        # --- Body FK ---
        # Joint 0 (Hips Position Handle)
        all_global_pos[f, 0] = curr_root_pos
        all_global_rot[f, 0] = np.eye(3) 
        
        # Joint 1 (Hips Rotation Handle)
        all_global_pos[f, 1] = curr_root_pos
        all_global_rot[f, 1] = curr_root_rot
        
        # Joints 2..21 (Body)
        for i in range(2, 22):
            parent_idx = PARENTS[i]
            offset = offsets[i]
            
            # Local Rotation from data
            local_rot = body_rots_mat[f, i-2]
            
            # Parent Global State
            parent_rot = all_global_rot[f, parent_idx]
            parent_pos = all_global_pos[f, parent_idx]
            
            # FK Calculation
            curr_rot = parent_rot @ local_rot
            curr_pos = parent_pos + (parent_rot @ offset)
            
            all_global_rot[f, i] = curr_rot
            all_global_pos[f, i] = curr_pos
            
    return all_global_pos

# ==========================================
# PLOTTING
# ==========================================
def main():
    if not os.path.exists(INFERENCE_FILE):
        print(f"File not found: {INFERENCE_FILE}")
        return

    print(f"Loading: {INFERENCE_FILE}")
    data = np.load(INFERENCE_FILE)
    print(f"Data Shape: {data.shape}")
    
    # Compute FK
    print("Computing Forward Kinematics...")
    global_pos = compute_fk_from_deltas(data, OFFSETS)
    
    print("Launching Plotter...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.2)
    
    # Create lines for bones
    lines = [ax.plot([],[],[], 'b-o', ms=2, mec='k', lw=2)[0] for _ in range(22)]
    
    # --- STATIC AXIS LIMITS (Fixed -2 to 2) ---
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set initial view once (User can rotate freely afterwards)
    ax.view_init(elev=20, azim=-45)

    def update(val):
        f = int(slider.val)
        xyz = global_pos[f]
        
        for i, line in enumerate(lines):
            if i == 0: continue
            p = PARENTS[i]
            
            # Update line data only
            line.set_data_3d(
                [xyz[p,0], xyz[i,0]],
                [xyz[p,1], xyz[i,1]],
                [xyz[p,2], xyz[i,2]]
            )
        
        # NOTE: We DO NOT call set_xlim/ylim here. 
        # This keeps the camera static and allows the character to move out of frame 
        # or be rotated by the user.
            
        fig.canvas.draw_idle()
        
    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, len(data)-1, valinit=0, valfmt='%d')
    slider.on_changed(update)
    
    plt.show()

if __name__ == "__main__":
    main()