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
# Fix leaf nodes
for i in [1, 2, 9, 10, 11, 12, 13, 14, 21, 22, 23, 24]: CHILDREN[i] = i

I_NOSE, I_L_SH, I_R_SH = 0, 3, 4
I_L_HIP, I_R_HIP, I_MIDHIP = 15, 16, 26

def add_virtual_joints(raw_data):
    """
    Input: (F, 25, 7) -> Output: (F, 27, 7)
    Adds Neck (25) and MidHip (26)
    """
    l_hip, r_hip = raw_data[:, I_L_HIP], raw_data[:, I_R_HIP]
    l_sh, r_sh = raw_data[:, I_L_SH], raw_data[:, I_R_SH]

    neck = np.zeros((raw_data.shape[0], 7), dtype=np.float32)
    mid_hip = np.zeros((raw_data.shape[0], 7), dtype=np.float32)

    # Position
    mid_hip[:, 0:5] = (l_hip[:, 0:5] + r_hip[:, 0:5]) / 2.0
    neck[:, 0:5]    = (l_sh[:, 0:5]  + r_sh[:, 0:5])  / 2.0
    
    # Metadata
    mid_hip[:, 5] = np.minimum(l_hip[:, 5], r_hip[:, 5])
    neck[:, 5]    = np.minimum(l_sh[:, 5],  r_sh[:, 5])
    
    # Anchor
    anchor_flag = raw_data[:, I_NOSE, 6]
    mid_hip[:, 6] = anchor_flag
    neck[:, 6]    = anchor_flag

    return np.concatenate([raw_data, neck[:, np.newaxis], mid_hip[:, np.newaxis]], axis=1)

def get_alignment_angle(l_hip, r_hip):
    """Calculates hip angle in World Space (XZ)."""
    diff = r_hip - l_hip
    angle = np.arctan2(diff[:, 2], diff[:, 0])
    return np.sin(angle), np.cos(angle)

def align_to_canonical_frame(pos_data):
    """Rotates frame so MidHip=(0,0,0) and Hips align with X-axis."""
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
    pos_aligned = np.einsum('bjk,bkp->bjp', pos_data, rot_mat)
    return pos_aligned

def process_blazepose_frames(raw_data, window_size):
    """
    Input: (F, 25, 7)
    Output: (Windows, 27, 19)
    """
    # 1. Add Virtual Joints
    data_27 = add_virtual_joints(raw_data)
    
    pos_world = data_27[:, :, 0:3]
    pos_screen = data_27[:, :, 3:5] 
    vis = data_27[:, :, 5:6]
    anchor = data_27[:, :, 6:7]
    
    # Center World Data
    root_pos = pos_world[:, I_MIDHIP:I_MIDHIP+1, :]
    pos_centered = pos_world - root_pos
    
    # --- A. GLOBAL MOTION CUES (5 Dims) ---
    
    # 1. Alignment Angle (2 dims)
    sin_a, cos_a = get_alignment_angle(pos_centered[:, I_L_HIP], pos_centered[:, I_R_HIP])
    alignment_feat = np.stack([sin_a, cos_a], axis=1)[:, np.newaxis, :] 
    alignment_feat = np.repeat(alignment_feat, NUM_JOINTS, axis=1)      
    
    # 2. Screen Velocity (2 dims) - Scaled x10
    root_screen_vel = np.zeros_like(pos_screen[:, I_MIDHIP:I_MIDHIP+1, :])
    root_screen_vel[1:] = (pos_screen[1:, I_MIDHIP:I_MIDHIP+1, :] - pos_screen[:-1, I_MIDHIP:I_MIDHIP+1, :]) * 10.0
    root_screen_vel = np.repeat(root_screen_vel, NUM_JOINTS, axis=1)    
    
    # 3. Scale Change (1 dim) - Scaled x100
    screen_l_hip = pos_screen[:, I_L_HIP, :]
    screen_r_hip = pos_screen[:, I_R_HIP, :]
    hip_width = np.linalg.norm(screen_l_hip - screen_r_hip, axis=1)
    
    scale_change = np.zeros_like(hip_width)
    scale_change[1:] = (hip_width[1:] - hip_width[:-1]) * 100.0
    scale_change = scale_change[:, np.newaxis, np.newaxis]
    scale_change = np.repeat(scale_change, NUM_JOINTS, axis=1) 

    # --- B. CANONICAL POSE (12 Dims) ---
    
    # 4. Canonical Pos (3 dims)
    pos_canonical = align_to_canonical_frame(pos_centered)
    
    # 5. Local Velocity (3 dims)
    velocity = np.zeros_like(pos_canonical)
    velocity[1:] = pos_canonical[1:] - pos_canonical[:-1]
    is_start = (anchor[:, 0, 0] == 0)
    velocity[is_start] = 0.0
    
    # 6. Structure Vectors (6 dims)
    parent_vecs = pos_canonical - pos_canonical[:, PARENTS]
    child_vecs = pos_canonical[:, CHILDREN] - pos_canonical
    
    leaf_mask = (CHILDREN == np.arange(NUM_JOINTS))
    child_vecs[:, leaf_mask, :] = parent_vecs[:, leaf_mask, :]
    
    # --- C. STACK (19 Dims) ---
    features = np.concatenate([
        pos_canonical,  # 3
        velocity,       # 3
        parent_vecs,    # 3
        child_vecs,     # 3
        vis,            # 1
        anchor,         # 1
        alignment_feat, # 2
        root_screen_vel,# 2
        scale_change    # 1
    ], axis=2)
    # Total: 19
    
    # Windowing
    F = features.shape[0]
    N = window_size
    padding = np.repeat(features[0:1], N-1, axis=0)
    padded = np.concatenate([padding, features], axis=0)
    
    s0, s1, s2 = padded.strides
    X_windows = np.lib.stride_tricks.as_strided(
        padded, shape=(F, N, 27, 19), strides=(s0, s0, s1, s2)
    )
    
    M_masks = np.zeros((F, N), dtype=bool)
    
    return X_windows.copy(), M_masks