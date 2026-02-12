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

def get_alignment_angle(pos_data):
    """
    Calculates the angle of the hips in World XZ space.
    This captures the 'Lost Orientation' before we canonicalize.
    """
    # Vector from Left Hip to Right Hip
    diff = pos_data[:, I_R_HIP] - pos_data[:, I_L_HIP]
    # Angle in XZ plane (Top-down view)
    angle = np.arctan2(diff[:, 2], diff[:, 0])
    return np.sin(angle), np.cos(angle)

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
    Extracts 19 Features:
    - 12 for Canonical Pose (Indices 0-11)
    - 2 for Metadata (Indices 12-13)
    - 5 for View Context/Root Motion (Indices 14-18)
    """
    # 1. Add Virtual Joints -> (F, 27, 7)
    data_27 = add_virtual_joints(raw_data)
    
    pos_world = data_27[:, :, 0:3]
    pos_screen = data_27[:, :, 3:5] 
    vis = data_27[:, :, 5:6]
    anchor = data_27[:, :, 6:7]
    
    # Center World Data (MidHip becomes 0,0,0 for translation, but rotation remains)
    root_pos = pos_world[:, I_MIDHIP:I_MIDHIP+1, :]
    pos_centered = pos_world - root_pos
    
    # --- A. VIEW CONTEXT (Solving the Root Ambiguity) ---
    # These features tell the model "Where am I facing?" and "How am I moving on screen?"
    
    # 1. Alignment Angle (2 Dims)
    # Measures hip rotation in world space *before* canonicalization.
    sin_a, cos_a = get_alignment_angle(pos_centered)
    align_feat = np.stack([sin_a, cos_a], axis=1)[:, np.newaxis, :]
    align_feat = np.repeat(align_feat, NUM_JOINTS, axis=1)
    
    # 2. Scale Change (1 Dim)
    # Measures depth speed (Zoom).
    screen_l_hip = pos_screen[:, I_L_HIP, :]
    screen_r_hip = pos_screen[:, I_R_HIP, :]
    width = np.linalg.norm(screen_l_hip - screen_r_hip, axis=1)
    
    scale_delta = np.zeros_like(width)
    scale_delta[1:] = (width[1:] - width[:-1]) * 100.0 # Scale up for numeric stability
    scale_feat = np.repeat(scale_delta[:, None, None], NUM_JOINTS, axis=1)
    
    # 3. Screen Velocity (2 Dims)
    # Measures lateral speed.
    s_vel = np.zeros_like(pos_screen[:, I_MIDHIP:I_MIDHIP+1, :])
    s_vel[1:] = (pos_screen[1:, I_MIDHIP:I_MIDHIP+1, :] - pos_screen[:-1, I_MIDHIP:I_MIDHIP+1, :]) * 10.0
    s_vel_feat = np.repeat(s_vel, NUM_JOINTS, axis=1)

    # --- B. CANONICAL POSE (Solving the Body) ---
    # These features are "Cleaned" so the model learns pure pose structure.
    
    # 4. Canonical Pos (3 Dims) - Hips forced to Z=0
    pos_canonical = align_to_canonical_frame(pos_centered)
    
    # 5. Local Velocity (3 Dims)
    vel_canonical = np.zeros_like(pos_canonical)
    vel_canonical[1:] = pos_canonical[1:] - pos_canonical[:-1]
    vel_canonical[anchor[:, 0, 0] == 0] = 0.0
    
    # 6. Structure Vectors (6 Dims)
    parent_vecs = pos_canonical - pos_canonical[:, PARENTS]
    child_vecs = pos_canonical[:, CHILDREN] - pos_canonical
    
    leaf_mask = (CHILDREN == np.arange(NUM_JOINTS))
    child_vecs[:, leaf_mask, :] = parent_vecs[:, leaf_mask, :]

    # --- C. STACK 19 FEATURES ---
    features = np.concatenate([
        pos_canonical,  # 3 (Pose)
        vel_canonical,  # 3 (Pose Vel)
        parent_vecs,    # 3 (Bone)
        child_vecs,     # 3 (Bone)
        vis,            # 1 (Meta)
        anchor,         # 1 (Meta)
        align_feat,     # 2 (Root Context)
        s_vel_feat,     # 2 (Root Context)
        scale_feat      # 1 (Root Context)
    ], axis=2)
    # Total: 3+3+3+3+1+1+2+2+1 = 19
    
    # Windowing
    F, N = features.shape[0], window_size
    padded = np.concatenate([np.repeat(features[:1], N-1, axis=0), features], axis=0)
    s0, s1, s2 = padded.strides
    X_windows = np.lib.stride_tricks.as_strided(
        padded, shape=(F, N, 27, 19), strides=(s0, s0, s1, s2)
    )
    
    M_masks = np.zeros((F, N), dtype=bool)
    
    return X_windows.copy(), M_masks