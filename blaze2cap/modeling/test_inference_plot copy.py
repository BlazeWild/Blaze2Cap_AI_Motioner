
"""
Inference & Visualization (Fixed Scale World)
=============================================
1. Loads a random BlazePose input file (.npy).
2. Preprocesses it (Virtual Joints + Deltas + Windowing).
3. Runs the MotionTransformer model (loaded from checkpoint).
4. Reconstructs global motion from model output.
5. Visualizes result using Fixed Scale World plot.
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
# Add the root of the repo (Blaze2Cap) to sys.path
repo_root = str(Path(__file__).resolve().parents[2])
sys.path.append(repo_root)

from blaze2cap.utils.skeleton_config import get_totalcapture_skeleton
from blaze2cap.modules.models import MotionTransformer
from blaze2cap.modules.pose_processing import process_blazepose_frames

# --- Configuration ---
# Path to input files (BlazePose Augmented)
INPUT_DATASET_ROOT = os.path.join(repo_root, "../training_dataset_both_in_out/blaze_augmented")
CHECKPOINT_FILE = os.path.join(repo_root, "checkpoints/milestone_epoch150.pth")
WINDOW_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Filter Options for random file selection
# FILTER_SUBJECT = "S1" # e.g., "S1"
# FILTER_ACTION = "acting1"  # e.g., "acting1"
# FILTER_CAM = "cam1"     # e.g., "cam1"

# Filter Options for random file selection
FILTER_SUBJECT = None # e.g., "S1"
FILTER_ACTION = None  # e.g., "acting1"
FILTER_CAM = None     # e.g., "cam1"

# --- 1. Helper Functions (Reconstruction) ---

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
    # NOTE: Model predicts translation delta directly.
    # In test_fk_plot_withrm.py, it used: root_vel_local = data[:, 0, :3] * -1.0
    # Let's keep consistency with the visualizer we are copying.
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
        # Note: offsets should move to device if we were doing this on GPU, 
        # but here everything is CPU/Numpy for plotting.
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

# --- 2. File & Model Management ---

def find_input_file():
    subj = FILTER_SUBJECT if FILTER_SUBJECT else "*"
    act = FILTER_ACTION if FILTER_ACTION else "*"
    cam = FILTER_CAM if FILTER_CAM else "*"
    
    # Check if path exists
    if not os.path.exists(INPUT_DATASET_ROOT):
        raise ValueError(f"Input dataset root not found: {INPUT_DATASET_ROOT}")

    pattern = os.path.join(INPUT_DATASET_ROOT, subj, act, cam, "*.npy")
    files = glob.glob(pattern)
    if not files: 
        # Fallback recursive search if directory structure is slightly different
        pattern_rec = os.path.join(INPUT_DATASET_ROOT, "**", "*.npy")
        files = glob.glob(pattern_rec, recursive=True)
        
    if not files: raise ValueError(f"No files found matching: {pattern}")
    return random.choice(files)

def load_best_model():
    # Load specific checkpoint
    best_ckpt = CHECKPOINT_FILE
    if not os.path.exists(best_ckpt):
        raise ValueError(f"Checkpoint not found: {best_ckpt}")
    
    print(f"Loading checkpoint: {best_ckpt}")
    
    model = MotionTransformer(
        num_joints=27, # Must match pose_processing.py (25 + 2 virtual)
        input_feats=18, 
        d_model=256, 
        num_layers=4, 
        n_head=4, 
        d_ff=512, 
        dropout=0.1,
        max_len=512
    ).to(DEVICE)
    
    state_dict = torch.load(best_ckpt, map_location=DEVICE)
    if 'model' in state_dict: state_dict = state_dict['model']
    
    # Strip 'module.' prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    return model

# --- 3. Main Routine ---

def main():
    # A. Setup
    skel_cfg = get_totalcapture_skeleton()
    parents = skel_cfg['parents']
    offsets = skel_cfg['offsets']
    
    # B. Load Input
    filepath = find_input_file()
    print(f"Input File: {filepath}")
    
    raw_input = np.load(filepath) # (F, 25, 7)
    
    # C. Preprocess
    # process_blazepose_frames returns (F, Window, Features)
    print("Preprocessing input...")
    X_windows, _ = process_blazepose_frames(raw_input, window_size=WINDOW_SIZE)
    
    input_tensor = torch.from_numpy(X_windows).float().to(DEVICE)
    
    # D. Inference
    print("Running inference...")
    model = load_best_model()
    
    all_preds = []
    BATCH_SIZE = 128
    
    with torch.no_grad():
        for i in range(0, len(input_tensor), BATCH_SIZE):
            batch = input_tensor[i : i+BATCH_SIZE]
            
            # Forward pass
            root_out, body_out = model(batch)
            
            # Take last frame of window [B, -1, ...]
            root_last = root_out[:, -1, :, :] # [B, 2, 6]
            body_last = body_out[:, -1, :, :] # [B, 20, 6]
            
            # Combine to [B, 22, 6]
            # root is joints 0-1, body is 2-21
            combined = torch.cat([root_last, body_last], dim=1)
            all_preds.append(combined.cpu())
            
    # Concatenate all batches
    full_pred_6d = torch.cat(all_preds, dim=0).numpy() # [F, 22, 6]
    print(f"Prediction Complete. Shape: {full_pred_6d.shape}")
    
    # E. Reconstruct 3D Positions
    print("Reconstructing global motion...")
    positions = reconstruct_global_motion(full_pred_6d, parents, offsets)
    num_frames = positions.shape[0]

    # F. Visualization (Exact copy of visualizer logic)
    print("Starting visualization...")
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(f"Inference: {os.path.basename(filepath)}", fontsize=12)
    plt.subplots_adjust(bottom=0.2)
    
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize Objects
    scat = ax.scatter([], [], [], c='r', s=15)
    lines = [ax.plot([], [], [], 'b-', lw=2)[0] for _ in range(1, 22)]
    traj_line, = ax.plot([], [], [], 'g--', lw=1)
    
    # --- STATIC AXIS LIMITS ---
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_zlim(-2.0, 2.0)
    
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
