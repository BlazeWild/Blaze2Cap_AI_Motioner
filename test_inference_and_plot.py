"""
Test Inference Script for Blaze2Cap Model
==========================================
Loads trained checkpoint, processes input BlazePose data, runs inference,
and visualizes the predicted 3D skeleton frame-by-frame.
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blaze2cap.modules.models import MotionTransformer
from tools.skeleton_config import OFFSETS_METERS, PARENTS, JOINT_NAMES


# ============================================================================
# 1. PREPROCESSING (from data_loader.py)
# ============================================================================

def preprocess_blazepose(raw_data, window_size=60):
    """
    Preprocess BlazePose data using the same logic as data_loader.py
    
    Input: (F, 25, 7) - [world_xyz, screen_xy, visibility, anchor]
    Output: (F, N, 450) - windowed features, (F, N) - masks
    """
    N = window_size
    F, J, _ = raw_data.shape
    
    # Hierarchy for BlazePose (25 joints)
    parents = np.arange(25)
    children = np.arange(25)
    
    # Define parent-child relationships
    parents[3], parents[4] = 1, 3
    children[3], children[4] = 4, 5
    parents[5], parents[6] = 4, 5
    children[5], children[6] = 6, 7
    parents[7], parents[8] = 6, 7
    children[7], children[8] = 9, 10
    parents[[9, 11, 13]] = 7
    parents[[10, 12, 14]] = 8
    children[15], children[16] = 17, 18
    parents[17], parents[18] = 15, 16
    children[17], children[18] = 19, 20
    parents[19], parents[20] = 17, 18
    children[19], children[20] = 21, 22
    parents[[21, 23]] = 19
    parents[[22, 24]] = 20
    
    # 1. Padding Prep
    validity = np.ones((F, 1), dtype=np.float32)
    padding_data = np.repeat(raw_data[0:1], N-1, axis=0)
    padding_validity = np.zeros((N-1, 1), dtype=np.float32)
    
    full_data = np.concatenate([padding_data, raw_data], axis=0)
    full_validity = np.concatenate([padding_validity, validity], axis=0)
    F_pad = full_data.shape[0]

    # 2. Extract Components
    pos_raw = full_data[:, :, 0:3]    # World XYZ
    screen_raw_01 = full_data[:, :, 3:5]  # Screen [0, 1]
    vis_raw = full_data[:, :, 5:6]    # Visibility
    anchor_raw = full_data[:, :, 6:7]  # Anchor flag
    
    is_anchor = (anchor_raw[:, 0, 0] == 0)

    # 3. Transform Screen Coords [0,1] -> [-1, 1]
    screen_norm = (screen_raw_01 * 2.0) - 1.0

    # 4. Feature: Centered Screen Position (Hip Relative)
    screen_hip_center = (screen_norm[:, 15] + screen_norm[:, 16]) / 2.0  # (F_pad, 2)
    screen_centered = screen_norm - screen_hip_center[:, None, :]

    # 5. Feature: Delta Screen
    delta_screen = np.zeros_like(screen_centered)
    delta_screen[1:] = screen_centered[1:] - screen_centered[:-1]
    delta_screen[is_anchor] = 0.0

    # 6. Feature: World Handling
    world_hip_center = (pos_raw[:, 15] + pos_raw[:, 16]) / 2.0
    world_centered = pos_raw - world_hip_center[:, None, :]
    
    delta_world = np.zeros_like(world_centered)
    delta_world[1:] = world_centered[1:] - world_centered[:-1]
    delta_world[is_anchor] = 0.0

    # 7. Feature: Vectors (Bone/Parent)
    bone_vecs = world_centered[:, children] - world_centered
    leaf_mask = (children == np.arange(25))
    bone_vecs[:, leaf_mask] = 0
    
    parent_vecs = world_centered - world_centered[:, parents]

    # 8. Stack Features (18 features per joint)
    features = np.concatenate([
        world_centered,   # 3
        delta_world,      # 3
        bone_vecs,        # 3
        parent_vecs,      # 3
        screen_centered,  # 2
        delta_screen,     # 2
        vis_raw,          # 1
        anchor_raw        # 1
    ], axis=2)  # (F_pad, 25, 18)
    
    # Flatten: (F_pad, 450)
    D = 18 * 25
    features_flat = features.reshape(F_pad, D)
    features_flat *= full_validity  # Zero out padding
    
    # Windowing using stride tricks
    strides = (features_flat.strides[0], features_flat.strides[0], features_flat.strides[1])
    X_windows = np.lib.stride_tricks.as_strided(
        features_flat, shape=(F, N, D), strides=strides
    )
    
    # Create padding masks
    idx_matrix = np.arange(F)[:, None] + np.arange(N)
    M_masks = (full_validity[idx_matrix, 0] <= 0.5)  # True = padding (ignore)
    
    return X_windows, M_masks


# ============================================================================
# 2. FORWARD KINEMATICS (FK)
# ============================================================================

def rotation_6d_to_matrix(d6):
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    
    Args:
        d6: (6,) array - first two columns of rotation matrix
    Returns:
        R: (3, 3) rotation matrix
    """
    a1 = d6[0:3]
    a2 = d6[3:6]
    
    # Gram-Schmidt orthonormalization
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    
    R = np.stack([b1, b2, b3], axis=1)
    return R


def forward_kinematics(gt_6d, offsets, parents):
    """
    Perform Forward Kinematics to compute 3D joint positions.
    
    Args:
        gt_6d: (22, 6) - Ground truth in 6D format
               Joint 0: [dx, dy, dz, 0, 0, 0] - position delta
               Joint 1: [6D rotation] - root orientation
               Joints 2-21: [6D rotation] - local rotations
        offsets: (22, 3) - Bone offsets in meters
        parents: List of parent indices
        
    Returns:
        positions: (22, 3) - Global 3D positions of all joints
    """
    num_joints = 22
    positions = np.zeros((num_joints, 3))
    rotations = np.zeros((num_joints, 3, 3))
    
    # Initialize root rotation as identity
    rotations[0] = np.eye(3)
    
    for j in range(num_joints):
        if j == 0:
            # Joint 0: Root position delta (use directly as position)
            positions[j] = gt_6d[j, 0:3]
            
        elif j == 1:
            # Joint 1: Root orientation (6D -> rotation matrix)
            rotations[j] = rotation_6d_to_matrix(gt_6d[j])
            # Position: same as parent (Hips position)
            positions[j] = positions[parents[j]]
            
        else:
            # Other joints: Local rotations
            parent_idx = parents[j]
            
            # Convert 6D to rotation matrix
            local_rot = rotation_6d_to_matrix(gt_6d[j])
            
            # Global rotation: parent_rotation @ local_rotation
            rotations[j] = rotations[parent_idx] @ local_rot
            
            # Global position: parent_pos + parent_rotation @ offset
            offset = offsets[j].numpy()
            positions[j] = positions[parent_idx] + rotations[parent_idx] @ offset
    
    return positions


# ============================================================================
# 3. PLOTTING
# ============================================================================

def plot_skeleton_frame(positions, parents, frame_idx, output_dir="output_frames"):
    """
    Plot a single frame of the skeleton in 3D.
    
    Args:
        positions: (22, 3) - 3D joint positions
        parents: List of parent indices
        frame_idx: Frame number for title
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for NaN or Inf values
    if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
        print(f"Warning: Frame {frame_idx} contains NaN or Inf values, skipping...")
        return None
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot bones
    for j in range(1, len(parents)):  # Skip root (index 0)
        parent = parents[j]
        if parent >= 0:
            start = positions[parent]
            end = positions[j]
            
            ax.plot([start[0], end[0]], 
                   [start[1], end[1]], 
                   [start[2], end[2]], 
                   color='blue', linewidth=2, marker='o', markersize=5)
    
    # Plot joints
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
              c='red', s=50, alpha=0.8)
    
    # Set labels and limits
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title(f'Frame {frame_idx}')
    
    # Set equal aspect ratio and reasonable limits
    max_range = 1.0  # meters
    mid_x = positions[:, 0].mean()
    mid_y = positions[:, 1].mean()
    mid_z = positions[:, 2].mean()
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set view angle (similar to BVH plot)
    ax.view_init(elev=20, azim=-60)
    
    # Save figure with lower DPI for faster rendering
    output_path = os.path.join(output_dir, f'frame_{frame_idx:04d}.png')
    plt.savefig(output_path, dpi=80, bbox_inches='tight')
    plt.close()
    
    return output_path


# ============================================================================
# 4. MAIN INFERENCE PIPELINE
# ============================================================================

def main():
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, "checkpoints", "checkpoint_epoch29.pth")
    input_path = os.path.join(script_dir, "..", "training_dataset_both_in_out", "blaze_augmented", "S1", "acting1", "cam1", "blaze_S1_acting1_cam1_seg0_s1_o0.npy")
    window_size = 60
    batch_size = 8  # Process in smaller batches to avoid OOM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("Blaze2Cap Inference Test")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Input: {input_path}")
    print()
    
    # 1. Load the model
    print("Loading model...")
    model = MotionTransformer(
        num_joints=25,
        input_feats=18,
        d_model=256,
        num_layers=4,
        n_head=4,
        d_ff=512,
        dropout=0.1,
        max_len=512  # Match training configuration
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
    print()
    
    # 2. Load and preprocess input data
    print("Loading and preprocessing input data...")
    raw_data = np.load(input_path).astype(np.float32)
    print(f"Raw data shape: {raw_data.shape}")  # (F, 25, 7)
    
    X_windows, M_masks = preprocess_blazepose(raw_data, window_size=window_size)
    print(f"Preprocessed shape: {X_windows.shape}")  # (F, N, 450)
    print(f"Mask shape: {M_masks.shape}")  # (F, N)
    print()
    
    # 3. Run inference in batches
    print("Running inference...")
    num_samples = X_windows.shape[0]
    all_root_preds = []
    all_body_preds = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            X_batch = torch.from_numpy(X_windows[i:end_idx]).to(device)
            M_batch = torch.from_numpy(M_masks[i:end_idx]).to(device)
            
            root_out, body_out = model(X_batch, key_padding_mask=M_batch)
            
            # Extract last frame prediction from each window
            root_pred = root_out[:, -1, :, :].cpu().numpy()  # (batch, 2, 6)
            body_pred = body_out[:, -1, :, :].cpu().numpy()  # (batch, 20, 6)
            
            all_root_preds.append(root_pred)
            all_body_preds.append(body_pred)
            
            if (i + batch_size) % 50 == 0:
                print(f"  Processed {min(i + batch_size, num_samples)}/{num_samples} frames")
    
    # Concatenate all predictions
    root_pred = np.concatenate(all_root_preds, axis=0)  # (F, 2, 6)
    body_pred = np.concatenate(all_body_preds, axis=0)  # (F, 20, 6)
    
    print(f"Root output shape: {root_pred.shape}")  # (F, 2, 6)
    print(f"Body output shape: {body_pred.shape}")  # (F, 20, 6)
    print()
    
    # 4. Combine predictions and accumulate root motion
    # Combine root and body: (F, 22, 6)
    predictions = np.concatenate([root_pred, body_pred], axis=1)
    print(f"Final predictions shape: {predictions.shape}")  # (F, 22, 6)
    
    # Accumulate root deltas to get absolute root motion
    print("Accumulating root motion (position + rotation deltas)...")
    accumulated_predictions = predictions.copy()
    
    # Start from origin
    accumulated_root_pos = np.zeros(3)
    accumulated_root_rot = np.eye(3)
    
    for i in range(len(predictions)):
        # Joint 0: Position delta -> accumulate
        delta_pos = predictions[i, 0, 0:3]
        accumulated_root_pos += delta_pos
        accumulated_predictions[i, 0, 0:3] = accumulated_root_pos
        
        # Joint 1: Rotation delta (6D) -> accumulate
        delta_rot_6d = predictions[i, 1, :]
        
        # Check for valid rotation
        if not (np.any(np.isnan(delta_rot_6d)) or np.any(np.isinf(delta_rot_6d))):
            delta_rot_mat = rotation_6d_to_matrix(delta_rot_6d)
            
            if not (np.any(np.isnan(delta_rot_mat)) or np.any(np.isinf(delta_rot_mat))):
                accumulated_root_rot = accumulated_root_rot @ delta_rot_mat
        
        # Convert back to 6D for FK
        accumulated_predictions[i, 1, 0:3] = accumulated_root_rot[:, 0]
        accumulated_predictions[i, 1, 3:6] = accumulated_root_rot[:, 1]
    
    predictions = accumulated_predictions
    print(f"✓ Root motion accumulated\n")
    
    # 5. Convert to 3D positions and plot
    print("Generating 3D skeleton visualizations...")
    num_frames = predictions.shape[0]
    print(f"Total frames available: {num_frames}")
    
    # Plot only first 300 frames
    max_frames_to_plot = min(300, num_frames)
    plot_every = 1  # Plot every frame
    print(f"Plotting first {max_frames_to_plot} frames (every {plot_every}th frame)...")
    
    output_dir = "output_frames"
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.abspath(output_dir)
    
    skipped_frames = 0
    plotted_frames = 0
    for frame_idx in range(0, max_frames_to_plot, plot_every):
        # Get GT for this frame
        gt_6d = predictions[frame_idx]  # (22, 6)
        
        # Perform FK to get 3D positions
        positions = forward_kinematics(gt_6d, OFFSETS_METERS, PARENTS)
        
        # Plot and save
        output_path = plot_skeleton_frame(positions, PARENTS, frame_idx, output_dir)
        
        if output_path is None:
            skipped_frames += 1
        else:
            plotted_frames += 1
        
        if (frame_idx + 1) % 50 == 0 or frame_idx == max_frames_to_plot - 1:
            print(f"  Progress: {frame_idx + 1}/{max_frames_to_plot} frames checked ({plotted_frames} valid plots, {skipped_frames} skipped)")
    
    print(f"\n✓ Frames saved to: {output_dir}/")
    print(f"  Total: {plotted_frames} valid frames plotted, {skipped_frames} skipped")
    print()
    
    # 6. Create a summary plot (frames that are valid - skip NaN frames)
    print("Creating summary visualization...")
    
    # Find valid frames for summary (skip first 59 NaN frames)
    valid_frame_start = 60
    summary_frames = [
        valid_frame_start,
        min(valid_frame_start + (max_frames_to_plot - valid_frame_start) // 2, max_frames_to_plot - 1),
        max_frames_to_plot - 1
    ]
    
    fig = plt.figure(figsize=(18, 6))
    plot_idx = 0
    for frame_idx in summary_frames:
        gt_6d = predictions[frame_idx]
        positions = forward_kinematics(gt_6d, OFFSETS_METERS, PARENTS)
        
        # Skip if NaN
        if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
            print(f"Skipping frame {frame_idx} in summary (NaN/Inf)")
            continue
        
        plot_idx += 1
        ax = fig.add_subplot(1, 3, plot_idx, projection='3d')
        
        # Plot bones
        for j in range(1, len(PARENTS)):
            parent = PARENTS[j]
            if parent >= 0:
                start = positions[parent]
                end = positions[j]
                ax.plot([start[0], end[0]], 
                       [start[1], end[1]], 
                       [start[2], end[2]], 
                       color='blue', linewidth=2, marker='o', markersize=5)
        
        # Plot joints
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  c='red', s=50, alpha=0.8)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Frame {frame_idx}')
        
        max_range = 1.0
        mid_x = positions[:, 0].mean()
        mid_y = positions[:, 1].mean()
        mid_z = positions[:, 2].mean()
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.view_init(elev=20, azim=-60)
    
    summary_path = "summary_prediction.png"
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"✓ Summary saved to: {summary_path}")
    
    plt.show()
    
    print()
    print("=" * 80)
    print("Inference complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
