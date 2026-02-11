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

# Parent Definitions for Feature Extraction
PARENTS = np.array([
    25, 0, 0, 25, 25, 3, 4, 5, 6,       # 0-8
    7, 8, 7, 8, 7, 8,                   # 9-14 (Hands)
    26, 26, 15, 16, 17, 18, 19, 20,     # 15-22 (Legs)
    19, 20,                             # 23-24 (Feet)
    26, 26                              # 25 (Neck->MidHip), 26 (MidHip->Root)
])

# Child Definitions (Approximate flow for "Child Vector")
CHILDREN = np.array([
    0, 0, 0, 5, 6, 7, 8, 11, 12,        # 0-8
    9, 10, 11, 12, 13, 14,              # 9-14 (Leafs point to self)
    17, 18, 19, 20, 23, 24, 23, 24,     # 15-22
    23, 24,                             # 23-24
    0, 25                               # 25, 26
])

# Fix leaf nodes in CHILDREN (Child = Self) to avoid zero vectors
for i in [1, 2, 9, 10, 11, 12, 13, 14, 21, 22, 23, 24]:
    CHILDREN[i] = i

I_L_HIP = 15
I_R_HIP = 16
I_MIDHIP = 26
I_NOSE = 0
I_L_SH = 3
I_R_SH = 4

def add_virtual_joints(raw_data):
    """
    Input: (F, 25, 7) -> Output: (F, 27, 7)
    """
    l_hip, r_hip = raw_data[:, I_L_HIP], raw_data[:, I_R_HIP]
    l_sh, r_sh = raw_data[:, I_L_SH], raw_data[:, I_R_SH]

    neck = np.zeros((raw_data.shape[0], 7), dtype=np.float32)
    mid_hip = np.zeros((raw_data.shape[0], 7), dtype=np.float32)

    # 1. Average Pos
    mid_hip[:, 0:3] = (l_hip[:, 0:3] + r_hip[:, 0:3]) / 2.0
    neck[:, 0:3]    = (l_sh[:, 0:3]  + r_sh[:, 0:3])  / 2.0
    
    # 2. Metadata
    mid_hip[:, 5] = np.minimum(l_hip[:, 5], r_hip[:, 5])
    mid_hip[:, 6] = raw_data[:, 0, 6]
    neck[:, 5] = np.minimum(l_sh[:, 5], r_sh[:, 5])
    neck[:, 6] = raw_data[:, 0, 6]

    return np.concatenate([raw_data, neck[:, np.newaxis], mid_hip[:, np.newaxis]], axis=1)

def align_to_canonical_frame(pos_data):
    """
    Rotates frame so MidHip=(0,0,0), L_Hip=-X, R_Hip=+X, Up=+Y.
    """
    # 1. Hip Axis
    l_hip = pos_data[:, I_L_HIP]
    r_hip = pos_data[:, I_R_HIP]
    x_axis = r_hip - l_hip
    x_len = np.linalg.norm(x_axis, axis=1, keepdims=True) + 1e-8
    x_axis = x_axis / x_len
    
    # 2. Up Axis (Global Y)
    global_y = np.zeros_like(x_axis)
    global_y[:, 1] = 1.0 
    
    # 3. Forward Axis (Z)
    z_axis = np.cross(x_axis, global_y)
    z_len = np.linalg.norm(z_axis, axis=1, keepdims=True) + 1e-8
    z_axis = z_axis / z_len
    
    # 4. Orthogonal Up (Y)
    y_axis = np.cross(z_axis, x_axis)
    
    # 5. Rotate
    rot_mat = np.stack([x_axis, y_axis, z_axis], axis=2) # (F, 3, 3)
    pos_aligned = np.einsum('bjk,bkp->bjp', pos_data, rot_mat)
    
    return pos_aligned

def process_blazepose_frames(raw_data, window_size):
    # 1. Add Virtual Joints (F, 27, 7)
    data_27 = add_virtual_joints(raw_data)
    
    # 2. Extract Components
    pos = data_27[:, :, 0:3]
    vis = data_27[:, :, 5:6]
    anchor = data_27[:, :, 6:7]
    
    # 3. Center at MidHip
    root_pos = pos[:, I_MIDHIP:I_MIDHIP+1, :]
    pos_centered = pos - root_pos
    
    # 4. Canonical Alignment
    pos_canonical = align_to_canonical_frame(pos_centered)
    
    # 5. Features
    # A. Velocity
    velocity = np.zeros_like(pos_canonical)
    velocity[1:] = pos_canonical[1:] - pos_canonical[:-1]
    is_start = (anchor[:, 0, 0] == 0)
    velocity[is_start] = 0.0
    
    # B. Structure Vectors
    parent_vecs = pos_canonical - pos_canonical[:, PARENTS]
    child_vecs = pos_canonical[:, CHILDREN] - pos_canonical
    
    # 6. Stack (14 dims)
    # 3(pos) + 3(vel) + 3(parent) + 3(child) + 1(vis) + 1(anc)
    features = np.concatenate([
        pos_canonical, 
        velocity, 
        parent_vecs, 
        child_vecs, 
        vis, 
        anchor
    ], axis=2)
    
    # 7. Windowing
    F = features.shape[0]
    N = window_size
    padding = np.repeat(features[0:1], N-1, axis=0)
    padded = np.concatenate([padding, features], axis=0)
    
    s0, s1, s2 = padded.strides
    X_windows = np.lib.stride_tricks.as_strided(
        padded, shape=(F, N, 27, 14), strides=(s0, s0, s1, s2)
    )
    
    M_masks = np.zeros((F, N), dtype=bool)
    
    return X_windows.copy(), M_masks