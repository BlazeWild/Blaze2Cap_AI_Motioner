# pose_processing_posonly.py

import numpy as np

# ==========================================
# 1. NEW TOPOLOGY (19 JOINTS)
# ==========================================
# Original 27 Indices -> New 19 Indices Mapping
# We keep: 0-8 (Upper), 15-20 (Legs), 23-24 (Feet), 25-26 (Virtual)
KEEP_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 19, 20, 23, 24, 25, 26]

# New Indices for readable code in this file
I_L_HIP_NEW = 9   # Was 15
I_R_HIP_NEW = 10  # Was 16
I_MIDHIP_NEW = 18 # Was 26 (Last one)

def add_virtual_joints(raw_data):
    """
    Adds Neck (25) and MidHip (26) to the raw 25-joint input.
    Input: (F, 25, 7) -> Output: (F, 27, 7)
    """
    # Original Indices for calculation
    L_HIP, R_HIP = 15, 16
    L_SH, R_SH = 3, 4
    
    l_hip, r_hip = raw_data[:, L_HIP], raw_data[:, R_HIP]
    l_sh, r_sh = raw_data[:, L_SH], raw_data[:, R_SH]

    # Initialize (F, 7)
    neck = np.zeros((raw_data.shape[0], 7), dtype=np.float32)
    mid_hip = np.zeros((raw_data.shape[0], 7), dtype=np.float32)

    # Average Pos (0-3: XYZ)
    mid_hip[:, 0:3] = (l_hip[:, 0:3] + r_hip[:, 0:3]) / 2.0
    neck[:, 0:3]    = (l_sh[:, 0:3]  + r_sh[:, 0:3])  / 2.0
    
    # Metadata (Vis, Anchor)
    mid_hip[:, 5] = np.minimum(l_hip[:, 5], r_hip[:, 5])
    mid_hip[:, 6] = raw_data[:, 0, 6] # Copy anchor from Nose
    
    neck[:, 5] = np.minimum(l_sh[:, 5], r_sh[:, 5])
    neck[:, 6] = raw_data[:, 0, 6]

    return np.concatenate([raw_data, neck[:, np.newaxis], mid_hip[:, np.newaxis]], axis=1)

def align_to_canonical_frame(pos_data):
    """
    Rotates every frame independently so:
    1. MidHip is at (0,0,0) (Already done by centering)
    2. Left Hip is at (-X, 0, 0)
    3. Right Hip is at (+X, 0, 0)
    
    Input: (F, 19, 3)
    Output: (F, 19, 3)
    """
    # 1. Get Hip Vectors (F, 3)
    l_hip = pos_data[:, I_L_HIP_NEW]
    r_hip = pos_data[:, I_R_HIP_NEW]
    
    # Vector from L to R (Should become +X)
    x_axis = r_hip - l_hip
    
    # Normalize X
    x_len = np.linalg.norm(x_axis, axis=1, keepdims=True) + 1e-8
    x_axis = x_axis / x_len
    
    # 2. Define Y Axis (Up)
    # We estimate Up using the spine/torso direction roughly, or cross product
    # A robust way is to use a temporary 'up' vector (0, 1, 0) and cross product
    # But since hips might be tilted, we construct a local basis.
    
    # Let's assume the body is roughly upright. 
    # Global Up (0, 1, 0)
    global_y = np.zeros_like(x_axis)
    global_y[:, 1] = 1.0 # Standard Y-up
    
    # Z axis = X cross Y (Forward)
    z_axis = np.cross(x_axis, global_y)
    z_len = np.linalg.norm(z_axis, axis=1, keepdims=True) + 1e-8
    z_axis = z_axis / z_len
    
    # Recalculate Y to be orthogonal (Z cross X)
    y_axis = np.cross(z_axis, x_axis)
    
    # 3. Construct Rotation Matrix (F, 3, 3)
    # R = [x, y, z]^T
    # We want to Project vectors ONTO this basis. 
    # So new_v = v . R
    
    rot_mat = np.stack([x_axis, y_axis, z_axis], axis=2) # (F, 3, 3)
    
    # Apply Rotation: (F, J, 3) x (F, 3, 3) -> (F, J, 3)
    # Einstein summation: batch b, joint j, coord k, new_coord p
    pos_aligned = np.einsum('bjk,bkp->bjp', pos_data, rot_mat)
    
    return pos_aligned

def process_blazepose_frames(raw_data, window_size):
    # 1. Add Virtual Joints (F, 27, 7)
    full_27 = add_virtual_joints(raw_data)
    
    # 2. Filter to 19 Joints (F, 19, 7)
    data_19 = full_27[:, KEEP_INDICES, :]
    
    # 3. Extract Components
    pos = data_19[:, :, 0:3]   # World XYZ
    vis = data_19[:, :, 5:6]   # Visibility
    anchor = data_19[:, :, 6:7] # Anchor Flag
    
    # 4. Center at MidHip
    root_pos = pos[:, I_MIDHIP_NEW:I_MIDHIP_NEW+1, :]
    pos_centered = pos - root_pos
    
    # 5. Canonical Alignment (Rotate Hips to X-axis, Z=0)
    pos_canonical = align_to_canonical_frame(pos_centered)
    
    # 6. Calculate Velocity (Delta)
    # Since we rotated every frame independently, "velocity" here is
    # the velocity OF THE POSE, not the global movement.
    velocity = np.zeros_like(pos_canonical)
    velocity[1:] = pos_canonical[1:] - pos_canonical[:-1]
    
    # Handle Scene Cuts (Anchor=0)
    is_start = (anchor[:, 0, 0] == 0)
    velocity[is_start] = 0.0
    
    # 7. Stack Channels (8 Total)
    # [Wx, Wy, Wz, Vx, Vy, Vz, Vis, Anchor]
    features = np.concatenate([
        pos_canonical, # 3
        velocity,      # 3
        vis,           # 1
        anchor         # 1
    ], axis=2)
    
    # 8. Windowing
    F = features.shape[0]
    N = window_size
    
    # Pad first frame
    padding = np.repeat(features[0:1], N-1, axis=0)
    padded = np.concatenate([padding, features], axis=0)
    
    # Stride tricks for windows
    # We want shape (F, N, 19, 8). 
    # Current padded is (F_pad, 19, 8). 
    # Byte strides are (s0, s1, s2)
    s0, s1, s2 = padded.strides
    X_windows = np.lib.stride_tricks.as_strided(
        padded, shape=(F, N, 19, 8), strides=(s0, s0, s1, s2)
    )
    
    # Dummy mask (all valid)
    M_masks = np.zeros((F, N), dtype=bool)
    
    return X_windows.copy(), M_masks