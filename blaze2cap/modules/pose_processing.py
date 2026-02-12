# pose_processing.py

import numpy as np

# ==========================================
# 1. TOPOLOGY CONFIGURATION (27 JOINTS)
# ==========================================
NUM_JOINTS = 27

# Parent Map: 0-24 Original, 25 Neck, 26 MidHip
PARENTS = np.array([
    25, 0, 0, 25, 25, 3, 4, 5, 6,       # 0-8
    7, 8, 7, 8, 7, 8,                   # 9-14
    26, 26, 15, 16, 17, 18, 19, 20,     # 15-22
    19, 20,                             # 23-24
    26, 26                              # 25, 26
])

# Child Map
CHILDREN = np.array([
    0, 0, 0, 5, 6, 7, 8, 11, 12,        # 0-8
    9, 10, 11, 12, 13, 14,              # 9-14
    17, 18, 19, 20, 23, 24, 23, 24,     # 15-22
    23, 24,                             # 23-24
    0, 25                               # 25, 26
])
# Leaf nodes point to themselves
for i in [1, 2, 9, 10, 11, 12, 13, 14, 21, 22, 23, 24]: CHILDREN[i] = i

I_NOSE, I_L_SH, I_R_SH = 0, 3, 4
I_L_HIP, I_R_HIP, I_MIDHIP = 15, 16, 26

def add_virtual_joints(raw_data):
    """
    Input: (F, 25, 7) -> Output: (F, 27, 7)
    Adds Neck (25) and MidHip (26) by averaging shoulders/hips.
    """
    l_hip, r_hip = raw_data[:, I_L_HIP], raw_data[:, I_R_HIP]
    l_sh, r_sh = raw_data[:, I_L_SH], raw_data[:, I_R_SH]

    neck = np.zeros((raw_data.shape[0], 7), dtype=np.float32)
    mid_hip = np.zeros((raw_data.shape[0], 7), dtype=np.float32)

    # Position
    mid_hip[:, 0:5] = (l_hip[:, 0:5] + r_hip[:, 0:5]) / 2.0
    neck[:, 0:5]    = (l_sh[:, 0:5]  + r_sh[:, 0:5])  / 2.0
    
    # Metadata (Visibility & Anchor)
    mid_hip[:, 5] = np.minimum(l_hip[:, 5], r_hip[:, 5])
    neck[:, 5]    = np.minimum(l_sh[:, 5],  r_sh[:, 5])
    
    anchor_flag = raw_data[:, I_NOSE, 6]
    mid_hip[:, 6] = anchor_flag
    neck[:, 6]    = anchor_flag

    return np.concatenate([raw_data, neck[:, np.newaxis], mid_hip[:, np.newaxis]], axis=1)

def get_alignment_6d(pos_data):
    """
    Calculates the 6D Rotation Representation of the Hip Alignment.
    Constructs a rotation matrix where X-axis aligns with the vector (L_HIP -> R_HIP).
    Output: (F, 6)
    """
    # 1. Construct X-axis (Right Vector) from Hips
    # Vector from Left Hip to Right Hip
    r_vec = pos_data[:, I_R_HIP] - pos_data[:, I_L_HIP] # (F, 3)
    
    # Normalize X
    x_len = np.linalg.norm(r_vec, axis=1, keepdims=True) + 1e-8
    x_axis = r_vec / x_len
    
    # 2. Construct Y-axis (Up Vector) - Assumed Global Y (0,1,0)
    # We use global Y to keep the alignment planar (trajectory-like)
    y_global = np.zeros_like(x_axis)
    y_global[:, 1] = 1.0
    
    # 3. Construct Z-axis (Forward) via Cross Product
    z_axis = np.cross(x_axis, y_global)
    z_len = np.linalg.norm(z_axis, axis=1, keepdims=True) + 1e-8
    z_axis = z_axis / z_len
    
    # 4. Re-orthogonalize Y-axis (Y = Z cross X)
    y_axis = np.cross(z_axis, x_axis)
    
    # 5. Extract 6D Representation (First 2 columns: X and Y)
    # Shape: (F, 6) -> [x_x, x_y, x_z, y_x, y_y, y_z]
    feat_6d = np.concatenate([x_axis, y_axis], axis=1)
    
    return feat_6d

def align_to_canonical_frame(pos_data):
    """
    Rotates frame so MidHip=(0,0,0) and Hips align with X-axis (Z=0).
    This creates the 'Clean Pose' for MPJPE learning.
    """
    l_hip = pos_data[:, I_L_HIP]
    r_hip = pos_data[:, I_R_HIP]
    
    x_axis = r_hip - l_hip
    x_len = np.linalg.norm(x_axis, axis=1, keepdims=True) + 1e-8
    x_axis = x_axis / x_len
    
    global_y = np.zeros_like(x_axis); global_y[:, 1] = 1.0 
    z_axis = np.cross(x_axis, global_y)
    z_len = np.linalg.norm(z_axis, axis=1, keepdims=True) + 1e-8
    z_axis = z_axis / z_len
    y_axis = np.cross(z_axis, x_axis)
    
    rot_mat = np.stack([x_axis, y_axis, z_axis], axis=2) 
    # Rotate points
    pos_aligned = np.einsum('bjk,bkp->bjp', pos_data, rot_mat)
    return pos_aligned


def process_blazepose_frames(raw_data, window_size):
    """
    Extracts 20 Features (User Adjusted):
    - 0-2: Canonical Pos (3)
    - 3-5: Canonical Vel (3)
    - 6-8: Parent Vec (3)
    - 9-11: Child Vec (3)
    - 12: Visibility (1)
    - 13: Anchor (1)
    - 14-19: Alignment 6D Rotation (6)
    """
    # 1. Add Virtual Joints -> (F, 27, 7)
    data_27 = add_virtual_joints(raw_data)
    
    pos_world = data_27[:, :, 0:3]
    vis = data_27[:, :, 5:6]
    anchor = data_27[:, :, 6:7]
    
    # Center World Data
    root_pos = pos_world[:, I_MIDHIP:I_MIDHIP+1, :]
    pos_centered = pos_world - root_pos
    
    # --- A. VIEW CONTEXT (Alignment 6D) ---
    # Indices 14-19 (6 Dims)
    # We calculate the hip alignment in 6D and broadcast it to all joints
    align_6d = get_alignment_6d(pos_centered) # (F, 6)
    align_feat = np.repeat(align_6d[:, np.newaxis, :], NUM_JOINTS, axis=1) # (F, 27, 6)

    # --- B. CANONICAL POSE ---
    # Indices 0-11
    pos_canonical = align_to_canonical_frame(pos_centered)
    
    vel_canonical = np.zeros_like(pos_canonical)
    vel_canonical[1:] = pos_canonical[1:] - pos_canonical[:-1]
    vel_canonical[anchor[:, 0, 0] == 0] = 0.0
    
    parent_vecs = pos_canonical - pos_canonical[:, PARENTS]
    child_vecs = pos_canonical[:, CHILDREN] - pos_canonical
    
    leaf_mask = (CHILDREN == np.arange(NUM_JOINTS))
    child_vecs[:, leaf_mask, :] = parent_vecs[:, leaf_mask, :]

    # --- C. STACK 20 FEATURES ---
    features = np.concatenate([
        pos_canonical,  # 3 (0-2)
        vel_canonical,  # 3 (3-5)
        parent_vecs,    # 3 (6-8)
        child_vecs,     # 3 (9-11)
        vis,            # 1 (12)
        anchor,         # 1 (13)
        align_feat,     # 6 (14-19) - The Hip Ori 6D
    ], axis=2)
    # Total: 3+3+3+3+1+1+6 = 20
    
    # Windowing
    F, N = features.shape[0], window_size
    padded = np.concatenate([np.repeat(features[:1], N-1, axis=0), features], axis=0)
    s0, s1, s2 = padded.strides
    X_windows = np.lib.stride_tricks.as_strided(
        padded, shape=(F, N, 27, 20), strides=(s0, s0, s1, s2)
    )
    
    M_masks = np.zeros((F, N), dtype=bool)
    
    return X_windows.copy(), M_masks