"""
Interactive 3D Skeleton Viewer with Frame Slider
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

from .blaze2cap.modules.models import MotionTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blaze2cap.modules.models import MotionTransformer
from tools.skeleton_config import OFFSETS_METERS, PARENTS, JOINT_NAMES


def preprocess_blazepose(raw_data, window_size=60):
    """Preprocess BlazePose data"""
    N = window_size
    F, J, _ = raw_data.shape
    
    parents = np.arange(25)
    children = np.arange(25)
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
    
    validity = np.ones((F, 1), dtype=np.float32)
    padding_data = np.repeat(raw_data[0:1], N-1, axis=0)
    padding_validity = np.zeros((N-1, 1), dtype=np.float32)
    
    full_data = np.concatenate([padding_data, raw_data], axis=0)
    full_validity = np.concatenate([padding_validity, validity], axis=0)
    F_pad = full_data.shape[0]

    pos_raw = full_data[:, :, 0:3]
    screen_raw_01 = full_data[:, :, 3:5]
    vis_raw = full_data[:, :, 5:6]
    anchor_raw = full_data[:, :, 6:7]
    
    is_anchor = (anchor_raw[:, 0, 0] == 0)
    screen_norm = (screen_raw_01 * 2.0) - 1.0
    screen_hip_center = (screen_norm[:, 15] + screen_norm[:, 16]) / 2.0
    screen_centered = screen_norm - screen_hip_center[:, None, :]

    delta_screen = np.zeros_like(screen_centered)
    delta_screen[1:] = screen_centered[1:] - screen_centered[:-1]
    delta_screen[is_anchor] = 0.0

    world_hip_center = (pos_raw[:, 15] + pos_raw[:, 16]) / 2.0
    world_centered = pos_raw - world_hip_center[:, None, :]
    
    delta_world = np.zeros_like(world_centered)
    delta_world[1:] = world_centered[1:] - world_centered[:-1]
    delta_world[is_anchor] = 0.0

    bone_vecs = world_centered[:, children] - world_centered
    leaf_mask = (children == np.arange(25))
    bone_vecs[:, leaf_mask] = 0
    
    parent_vecs = world_centered - world_centered[:, parents]

    features = np.concatenate([
        world_centered, delta_world, bone_vecs, parent_vecs,
        screen_centered, delta_screen, vis_raw, anchor_raw
    ], axis=2)
    
    D = 18 * 25
    features_flat = features.reshape(F_pad, D)
    features_flat *= full_validity
    
    strides = (features_flat.strides[0], features_flat.strides[0], features_flat.strides[1])
    X_windows = np.lib.stride_tricks.as_strided(
        features_flat, shape=(F, N, D), strides=strides
    )
    
    idx_matrix = np.arange(F)[:, None] + np.arange(N)
    M_masks = (full_validity[idx_matrix, 0] <= 0.5)
    
    return X_windows, M_masks


def rotation_6d_to_matrix(d6):
    """Convert 6D rotation to 3x3 matrix"""
    a1 = d6[0:3]
    a2 = d6[3:6]
    
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    
    R = np.stack([b1, b2, b3], axis=1)
    return R


def forward_kinematics(gt_6d, offsets, parents):
    """Perform FK to get 3D positions - using skeleton config directly, no flips or rotations"""
    num_joints = 22
    positions = np.zeros((num_joints, 3))
    rotations = np.zeros((num_joints, 3, 3))
    
    # Root joint (pelvis) - absolute position from delta accumulation
    rotations[0] = np.eye(3)
    positions[0] = gt_6d[0, 0:3]
    
    for j in range(1, num_joints):
        parent_idx = parents[j]
        
        if j == 1:
            # Global root rotation (accumulated)
            rotations[j] = rotation_6d_to_matrix(gt_6d[j])
            positions[j] = positions[parent_idx]  # Same position as parent
        else:
            # Local rotation relative to parent
            local_rot = rotation_6d_to_matrix(gt_6d[j])
            rotations[j] = rotations[parent_idx] @ local_rot
            
            # Get offset from skeleton config
            if isinstance(offsets[j], torch.Tensor):
                offset = offsets[j].numpy()
            else:
                offset = np.array(offsets[j])
            
            # Apply parent rotation to offset and add to parent position
            positions[j] = positions[parent_idx] + rotations[parent_idx] @ offset
    
    return positions


def main():
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, "checkpoints", "checkpoint_epoch57.pth")
    input_path = os.path.join(script_dir, "..", "training_dataset_both_in_out", "blaze_augmented", "S3", "acting1", "cam1", "blaze_S3_acting1_cam1_seg0_s2_o0.npy")
    window_size = 60
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("Blaze2Cap Interactive Viewer")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Input: {input_path}")
    print()
    
    # Load model
    print("Loading model...")
    model = MotionTransformer(
        num_joints=25, input_feats=18, d_model=256,
        num_layers=4, n_head=4, d_ff=512, dropout=0.1, max_len=512
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"✓ Model loaded from epoch {checkpoint['epoch']}\n")
    
    # Load and preprocess
    print("Loading and preprocessing...")
    raw_data = np.load(input_path).astype(np.float32)
    print(f"Raw data shape: {raw_data.shape}")
    
    X_windows, M_masks = preprocess_blazepose(raw_data, window_size=window_size)
    print(f"Preprocessed shape: {X_windows.shape}\n")
    
    # Run inference
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
            
            root_pred = root_out[:, -1, :, :].cpu().numpy()
            body_pred = body_out[:, -1, :, :].cpu().numpy()
            
            all_root_preds.append(root_pred)
            all_body_preds.append(body_pred)
            
            if (i + batch_size) % 200 == 0:
                print(f"  Processed {min(i + batch_size, num_samples)}/{num_samples} frames")
    
    root_pred = np.concatenate(all_root_preds, axis=0)
    body_pred = np.concatenate(all_body_preds, axis=0)
    predictions = np.concatenate([root_pred, body_pred], axis=1)
    
    print(f"✓ Inference complete: {predictions.shape}")
    print(f"  Root NaN count: {np.isnan(root_pred).sum()}, Inf count: {np.isinf(root_pred).sum()}")
    print(f"  Body NaN count: {np.isnan(body_pred).sum()}, Inf count: {np.isinf(body_pred).sum()}")
    print(f"  Sample root[100,0]: {root_pred[100, 0]}")
    print(f"  Sample root[100,1]: {root_pred[100, 1]}")
    print()
    
    # Accumulate root motion (deltas -> absolute) - CRITICAL: Follow fk_all_with_pos_delta.py logic
    print("Accumulating root motion (position + rotation deltas)...")
    accumulated_predictions = predictions.copy()
    
    accumulated_root_pos = np.zeros(3)
    accumulated_root_rot = np.eye(3)
    
    nan_frames = 0
    for i in range(len(predictions)):
        if i == 0:
            # Force identity on frame 0
            accumulated_root_pos = np.zeros(3)
            accumulated_root_rot = np.eye(3)
            accumulated_predictions[i, 0, 0:3] = accumulated_root_pos
            accumulated_predictions[i, 1, 0:3] = accumulated_root_rot[:, 0]
            accumulated_predictions[i, 1, 3:6] = accumulated_root_rot[:, 1]
            continue

        # Joint 0: Position delta -> rotate by current root, then add
        v_local = predictions[i, 0, 0:3]
        
        # Only accumulate if valid
        if not (np.any(np.isnan(v_local)) or np.any(np.isinf(v_local))):
            # CRITICAL: curr_root_pos = curr_root_pos + (curr_root_rot @ v_local)
            accumulated_root_pos = accumulated_root_pos + (accumulated_root_rot @ v_local)
        
        accumulated_predictions[i, 0, 0:3] = accumulated_root_pos
        
        # Joint 1: Rotation delta (6D) -> accumulate
        delta_rot_6d = predictions[i, 1, :]
        
        # Check for valid rotation before processing
        if not (np.any(np.isnan(delta_rot_6d)) or np.any(np.isinf(delta_rot_6d))):
            try:
                delta_rot_mat = rotation_6d_to_matrix(delta_rot_6d)
                
                # Check if rotation matrix is valid
                if not (np.any(np.isnan(delta_rot_mat)) or np.any(np.isinf(delta_rot_mat))):
                    # CRITICAL: curr_root_rot = curr_root_rot @ R_delta
                    test_rot = accumulated_root_rot @ delta_rot_mat
                    if not (np.any(np.isnan(test_rot)) or np.any(np.isinf(test_rot))):
                        accumulated_root_rot = test_rot
                    else:
                        nan_frames += 1
                else:
                    nan_frames += 1
            except:
                nan_frames += 1
        else:
            nan_frames += 1
        
        # Convert back to 6D
        accumulated_predictions[i, 1, 0:3] = accumulated_root_rot[:, 0]
        accumulated_predictions[i, 1, 3:6] = accumulated_root_rot[:, 1]
    
    predictions = accumulated_predictions
    print(f"✓ Root motion accumulated ({nan_frames} frames skipped due to NaN)\n")
    
    # Compute all FK positions
    print("Computing FK for all frames...")
    all_positions = []
    valid_frames = []
    nan_count = 0
    
    for i in range(predictions.shape[0]):
        gt_6d = predictions[i]
        
        # Check input before FK
        if np.any(np.isnan(gt_6d)) or np.any(np.isinf(gt_6d)):
            nan_count += 1
            continue
            
        positions = forward_kinematics(gt_6d, OFFSETS_METERS, PARENTS)
        
        if not (np.any(np.isnan(positions)) or np.any(np.isinf(positions))):
            all_positions.append(positions)
            valid_frames.append(i)
    
    all_positions = np.array(all_positions)
    print(f"✓ Valid frames: {len(valid_frames)}/{predictions.shape[0]} (NaN in input: {nan_count})\n")
    
    # Create interactive plot
    print("Creating interactive viewer...")
    print("Use the slider to navigate through frames!")
    print("You can rotate/zoom the 3D view with your mouse!")
    print("=" * 80)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.15)
    
    # Pre-create scatter and line objects for efficient updates
    scat = ax.scatter([], [], [], c='red', s=100, alpha=0.8, edgecolors='black', linewidths=2)
    
    # Create line objects for each bone connection
    bone_lines = []
    for j in range(1, len(PARENTS)):
        parent = PARENTS[j]
        if parent >= 0:
            line, = ax.plot([], [], [], color='blue', linewidth=3, marker='o', markersize=6)
            bone_lines.append((line, parent, j))
    
    def update_plot(frame_idx):
        frame_idx = int(frame_idx)
        positions = all_positions[frame_idx]
        actual_frame = valid_frames[frame_idx]
        
        # Update scatter plot
        scat._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        
        # Update bone lines
        for line, parent, child in bone_lines:
            line.set_data_3d(
                [positions[parent, 0], positions[child, 0]],
                [positions[parent, 1], positions[child, 1]],
                [positions[parent, 2], positions[child, 2]]
            )
        
        ax.set_title(f'Frame {actual_frame} (Valid: {frame_idx}/{len(valid_frames)-1})', 
                    fontsize=14, fontweight='bold')
        
        fig.canvas.draw_idle()
    
    # Set fixed axis limits to -2, 2 for all axes (same scale)
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_zlabel('Z (meters)', fontsize=12)
    
    # Fixed limits: -2 to 2 for all axes
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    
    # Calculate actual motion range for debugging
    all_x = all_positions[:, :, 0].flatten()
    all_y = all_positions[:, :, 1].flatten()
    all_z = all_positions[:, :, 2].flatten()
    
    print(f"Motion range: X[{all_x.min():.2f},{all_x.max():.2f}] Y[{all_y.min():.2f},{all_y.max():.2f}] Z[{all_z.min():.2f},{all_z.max():.2f}]")
    print()
    
    # Initial plot
    update_plot(0)
    
    # Create slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, len(valid_frames)-1, 
                    valinit=0, valstep=1, color='blue')
    
    slider.on_changed(update_plot)
    
    plt.show()
    
    print("\nViewer closed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
