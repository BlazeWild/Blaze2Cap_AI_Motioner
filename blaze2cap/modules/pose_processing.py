import numpy as np

# ==========================================
# 1. TOPOLOGY CONFIGURATION (27 JOINTS)
# ==========================================
# Indices:
# 0:Nose, 1:LEar, 2:REar, 3:LSh, 4:RSh, 5:LElb, 6:RElb, 7:LWri, 8:RWri
# 9:LPinky, 10:RPinky, 11:LIndex, 12:RIndex, 13:LThumb, 14:RThumb
# 15:LHip, 16:RHip, 17:LKnee, 18:RKnee, 19:LAnk, 20:RAnk, 21:LHeel, 22:RHeel
# 23:LFoot, 24:RFoot, 25:Neck, 26:MidHip

NUM_JOINTS = 27

# Parent Definitions (For structural features)
PARENTS = np.array([
    25, 0, 0, 25, 25, 3, 4, 5, 6,       # 0-8
    7, 8, 7, 8, 7, 8,                   # 9-14
    26, 26, 15, 16, 17, 18, 19, 20,     # 15-22
    19, 20,                             # 23-24
    26, 26                              # 25, 26
])

# Child Definitions (For structural features)
CHILDREN = np.array([
    0, 0, 0, 5, 6, 7, 8, 11, 12,        # 0-8
    9, 10, 11, 12, 13, 14,              # 9-14
    17, 18, 19, 20, 23, 24, 23, 24,     # 15-22
    23, 24,                             # 23-24
    0, 25                               # 25, 26
])

# Fix leaf nodes to point to self
for i in [1, 2, 9, 10, 11, 12, 13, 14, 21, 22, 23, 24]:
    CHILDREN[i] = i

# Indices for calculations
I_NOSE = 0
I_L_SH = 3
I_R_SH = 4
I_L_HIP = 15
I_R_HIP = 16
I_MIDHIP = 26

def add_virtual_joints(raw_data):
    """
    Calculates virtual joints (Neck, MidHip) from raw 25-joint BlazePose data.
    Input: (F, 25, 7) -> Output: (F, 27, 7)
    """
    l_hip, r_hip = raw_data[:, I_L_HIP], raw_data[:, I_R_HIP]
    l_sh, r_sh = raw_data[:, I_L_SH], raw_data[:, I_R_SH]

    neck = np.zeros((raw_data.shape[0], 7), dtype=np.float32)
    mid_hip = np.zeros((raw_data.shape[0], 7), dtype=np.float32)

    # 1. Average Pos (World + Screen)
    mid_hip[:, 0:5] = (l_hip[:, 0:5] + r_hip[:, 0:5]) / 2.0
    neck[:, 0:5]    = (l_sh[:, 0:5]  + r_sh[:, 0:5])  / 2.0
    
    # 2. Metadata (Vis, Anchor)
    mid_hip[:, 5] = np.minimum(l_hip[:, 5], r_hip[:, 5])
    neck[:, 5]    = np.minimum(l_sh[:, 5],  r_sh[:, 5])
    
    # Copy Anchor from Nose
    anchor_flag = raw_data[:, I_NOSE, 6]
    mid_hip[:, 6] = anchor_flag
    neck[:, 6]    = anchor_flag

    return np.concatenate([raw_data, neck[:, np.newaxis], mid_hip[:, np.newaxis]], axis=1)

def get_alignment_angle(l_hip, r_hip):
    """
    Calculates the sin/cos of the hip angle in World Space (XZ plane).
    This tells the model the "View Angle" before we normalize it.
    """
    # Vector from L to R
    diff = r_hip - l_hip
    dx, dz = diff[:, 0], diff[:, 2]
    
    # Angle of hips relative to World X
    angle = np.arctan2(dz, dx)
    
    # Return sin, cos
    return np.sin(angle), np.cos(angle)

def align_to_canonical_frame(pos_data):
    """
    Rotates frame so MidHip=(0,0,0) and Hips align with X-axis.
    Input: (F, 27, 3) -> Output: (F, 27, 3)
    """
    l_hip = pos_data[:, I_L_HIP]
    r_hip = pos_data[:, I_R_HIP]
    
    # 1. X-Axis (L -> R)
    x_axis = r_hip - l_hip
    x_len = np.linalg.norm(x_axis, axis=1, keepdims=True) + 1e-8
    x_axis = x_axis / x_len
    
    # 2. Y-Axis (Global Up)
    global_y = np.zeros_like(x_axis)
    global_y[:, 1] = 1.0 
    
    # 3. Z-Axis (Forward)
    z_axis = np.cross(x_axis, global_y)
    z_len = np.linalg.norm(z_axis, axis=1, keepdims=True) + 1e-8
    z_axis = z_axis / z_len
    
    # 4. Corrected Y
    y_axis = np.cross(z_axis, x_axis)
    
    # 5. Rotate
    rot_mat = np.stack([x_axis, y_axis, z_axis], axis=2) # (F, 3, 3)
    pos_aligned = np.einsum('bjk,bkp->bjp', pos_data, rot_mat)
    
    return pos_aligned

def process_blazepose_frames(raw_data, window_size):
    """
    Feature Extraction Pipeline (19 Dimensions).
    """
    # 1. Add Virtual Joints -> (F, 27, 7)
    data_27 = add_virtual_joints(raw_data)
    
    # 2. Extract Channels
    pos_world = data_27[:, :, 0:3]
    pos_screen = data_27[:, :, 3:5] # Need screen for Root cues
    vis = data_27[:, :, 5:6]
    anchor = data_27[:, :, 6:7]
    
    # 3. Center World Data (Canonical Prep)
    root_pos = pos_world[:, I_MIDHIP:I_MIDHIP+1, :]
    pos_centered = pos_world - root_pos
    
    # --- A. VIEW CONTEXT FEATURES ---
    
    # 1. Alignment Angle (Before rotation)
    # Tells model: "Am I side-view or front-view?"
    sin_a, cos_a = get_alignment_angle(pos_centered[:, I_L_HIP], pos_centered[:, I_R_HIP])
    alignment_feat = np.stack([sin_a, cos_a], axis=1)[:, np.newaxis, :] # (F, 1, 2)
    alignment_feat = np.repeat(alignment_feat, NUM_JOINTS, axis=1)      # (F, 27, 2)
    
    # 2. Screen Velocity (Global Motion)
    # Use Root Screen Velocity (robust to limb flailing)
    root_screen_vel = np.zeros_like(pos_screen[:, I_MIDHIP:I_MIDHIP+1, :])
    root_screen_vel[1:] = pos_screen[1:, I_MIDHIP:I_MIDHIP+1, :] - pos_screen[:-1, I_MIDHIP:I_MIDHIP+1, :]
    root_screen_vel = np.repeat(root_screen_vel, NUM_JOINTS, axis=1)    # (F, 27, 2)
    
    # 3. Scale Change (Zoom/Z-Motion)
    # Rate of change of hip width on screen
    screen_l_hip = pos_screen[:, I_L_HIP, :]
    screen_r_hip = pos_screen[:, I_R_HIP, :]
    hip_width = np.linalg.norm(screen_l_hip - screen_r_hip, axis=1)
    
    scale_change = np.zeros_like(hip_width)
    scale_change[1:] = hip_width[1:] - hip_width[:-1]
    scale_change = scale_change[:, np.newaxis, np.newaxis] # (F, 1, 1)
    scale_change = np.repeat(scale_change, NUM_JOINTS, axis=1) # (F, 27, 1)

    # --- B. CANONICAL POSE FEATURES ---
    
    # 4. Canonical Alignment
    pos_canonical = align_to_canonical_frame(pos_centered)
    
    # 5. Local Velocity
    velocity = np.zeros_like(pos_canonical)
    velocity[1:] = pos_canonical[1:] - pos_canonical[:-1]
    is_start = (anchor[:, 0, 0] == 0)
    velocity[is_start] = 0.0
    
    # 6. Structure Vectors
    parent_vecs = pos_canonical - pos_canonical[:, PARENTS]
    child_vecs = pos_canonical[:, CHILDREN] - pos_canonical
    
    # Fix leaf nodes for child vectors
    leaf_mask = (CHILDREN == np.arange(NUM_JOINTS))
    child_vecs[:, leaf_mask, :] = parent_vecs[:, leaf_mask, :]
    
    # --- C. FINAL STACK (19 DIMS) ---
    # Pos(3) + Vel(3) + Par(3) + Chi(3) + Vis(1) + Anc(1) + 
    # Align(2) + ScreenVel(2) + Scale(1) = 19
    features = np.concatenate([
        pos_canonical,  # 0-2
        velocity,       # 3-5
        parent_vecs,    # 6-8
        child_vecs,     # 9-11
        vis,            # 17 (moved to end in logic, but standard index here) -> 12
        anchor,         # 13
        alignment_feat, # 14-15
        root_screen_vel,# 16-17
        scale_change    # 18
    ], axis=2)
    
    # Re-ordering to keep metadata at end is optional, but let's stick to simple concat.
    # Current Order:
    # 0-2: Pos
    # 3-5: Vel
    # 6-8: Parent
    # 9-11: Child
    # 12: Vis
    # 13: Anc
    # 14-15: Align (Sin, Cos)
    # 16-17: Screen Vel (X, Y)
    # 18: Scale Delta
    
    # --- D. WINDOWING ---
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