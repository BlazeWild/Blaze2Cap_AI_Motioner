import numpy as np

# ==========================================
# 1. TOPOLOGY CONFIGURATION (27 JOINTS)
# ==========================================
I_NOSE = 0
I_L_EAR = 1;   I_R_EAR = 2
I_L_SH = 3;    I_R_SH = 4
I_L_ELB = 5;   I_R_ELB = 6
I_L_WRIST = 7; I_R_WRIST = 8
I_L_PINKY = 9; I_R_PINKY = 10
I_L_INDEX = 11; I_R_INDEX = 12
I_L_THUMB = 13; I_R_THUMB = 14
I_L_HIP = 15;  I_R_HIP = 16
I_L_KNEE = 17; I_R_KNEE = 18
I_L_ANKLE = 19; I_R_ANKLE = 20
I_L_HEEL = 21; I_R_HEEL = 22
I_L_FOOT = 23; I_R_FOOT = 24
I_NECK = 25    # Virtual
I_MIDHIP = 26  # Virtual (Root)

NUM_JOINTS = 27

PARENTS = np.arange(NUM_JOINTS)
CHILDREN = np.arange(NUM_JOINTS) 

# --- A. DEFINE PARENTS ---
PARENTS[I_MIDHIP] = I_MIDHIP
PARENTS[I_NECK] = I_MIDHIP
PARENTS[I_L_HIP] = I_MIDHIP; PARENTS[I_R_HIP] = I_MIDHIP
PARENTS[I_NOSE] = I_NECK
PARENTS[I_L_SH] = I_NECK;    PARENTS[I_R_SH] = I_NECK
PARENTS[I_L_EAR] = I_NOSE;   PARENTS[I_R_EAR] = I_NOSE

PARENTS[I_L_ELB] = I_L_SH
PARENTS[I_L_WRIST] = I_L_ELB
PARENTS[I_L_PINKY] = I_L_WRIST; PARENTS[I_L_INDEX] = I_L_WRIST; PARENTS[I_L_THUMB] = I_L_WRIST

PARENTS[I_R_ELB] = I_R_SH
PARENTS[I_R_WRIST] = I_R_ELB
PARENTS[I_R_PINKY] = I_R_WRIST; PARENTS[I_R_INDEX] = I_R_WRIST; PARENTS[I_R_THUMB] = I_R_WRIST

PARENTS[I_L_KNEE] = I_L_HIP
PARENTS[I_L_ANKLE] = I_L_KNEE
PARENTS[I_L_HEEL] = I_L_ANKLE; PARENTS[I_L_FOOT] = I_L_ANKLE

PARENTS[I_R_KNEE] = I_R_HIP
PARENTS[I_R_ANKLE] = I_R_KNEE
PARENTS[I_R_HEEL] = I_R_ANKLE; PARENTS[I_R_FOOT] = I_R_ANKLE

# --- B. DEFINE CHILDREN ---
CHILDREN[I_MIDHIP] = I_NECK
CHILDREN[I_NECK] = I_NOSE
CHILDREN[I_NOSE] = I_NOSE 

CHILDREN[I_L_SH] = I_L_ELB;     CHILDREN[I_R_SH] = I_R_ELB
CHILDREN[I_L_ELB] = I_L_WRIST;  CHILDREN[I_R_ELB] = I_R_WRIST
CHILDREN[I_L_WRIST] = I_L_INDEX; CHILDREN[I_R_WRIST] = I_R_INDEX

CHILDREN[I_L_HIP] = I_L_KNEE;   CHILDREN[I_R_HIP] = I_R_KNEE
CHILDREN[I_L_KNEE] = I_L_ANKLE; CHILDREN[I_R_KNEE] = I_R_ANKLE
CHILDREN[I_L_ANKLE] = I_L_FOOT; CHILDREN[I_R_ANKLE] = I_R_FOOT

leaf_indices = [I_L_EAR, I_R_EAR, I_L_PINKY, I_L_THUMB, I_L_INDEX, I_R_PINKY, I_R_THUMB, I_R_INDEX, I_L_HEEL, I_L_FOOT, I_R_HEEL, I_R_FOOT]
for idx in leaf_indices:
    CHILDREN[idx] = idx

def add_virtual_joints(raw_data):
    """
    Calculates virtual joints (Neck, MidHip) from raw 25-joint BlazePose data.
    Input: (F, 25, 7)
    Channels: 0-2 World, 3-4 Screen, 5 Vis, 6 Anchor (0=Start, 1=Cont)
    Output: (F, 27, 7)
    """
    # 1. Get Source Joints
    l_hip = raw_data[:, I_L_HIP, :]
    r_hip = raw_data[:, I_R_HIP, :]
    l_sh  = raw_data[:, I_L_SH, :]
    r_sh  = raw_data[:, I_R_SH, :]

    # 2. Compute Positions (World + Screen + Vis + Anchor)
    # We create empty arrays first
    neck    = np.zeros((raw_data.shape[0], 7), dtype=np.float32)
    mid_hip = np.zeros((raw_data.shape[0], 7), dtype=np.float32)

    # A. Position (Channels 0-4: World XYZ + Screen XY)
    # Average the positions of Left and Right
    mid_hip[:, 0:5] = (l_hip[:, 0:5] + r_hip[:, 0:5]) / 2.0
    neck[:, 0:5]    = (l_sh[:, 0:5]  + r_sh[:, 0:5])  / 2.0

    # B. Visibility (Channel 5)
    # Conservative estimate: Min visibility of parents
    mid_hip[:, 5] = np.minimum(l_hip[:, 5], r_hip[:, 5])
    neck[:, 5]    = np.minimum(l_sh[:, 5],  r_sh[:, 5])

    # C. Anchor Flag (Channel 6) - CORRECTION
    # Do NOT average. Directly copy the flag from an existing joint (e.g., Nose).
    # If Nose is at Frame 0 (Anchor=0), Virtual Joints are also at Frame 0.
    anchor_flag = raw_data[:, I_NOSE, 6]
    mid_hip[:, 6] = anchor_flag
    neck[:, 6]    = anchor_flag

    # 3. Reshape and Append
    # (F, 7) -> (F, 1, 7)
    neck = neck[:, np.newaxis, :]
    mid_hip = mid_hip[:, np.newaxis, :]
    
    return np.concatenate([raw_data, neck, mid_hip], axis=1)


def process_blazepose_frames(raw_data, window_size):
    """
    Core feature extraction pipeline.
    1. Adds virtual joints.
    2. Centers data around Root (MidHip).
    3. Computes relative vectors (Parent/Child).
    4. Handles Leaf Nodes by extending parent vectors.
    """
    # 1. Add Virtual Joints -> (F, 27, 7)
    data_27 = add_virtual_joints(raw_data)
    
    N = window_size
    F, J, _ = data_27.shape
    
    # 2. Padding (Replicate first frame)
    padding_data = np.repeat(data_27[0:1], N-1, axis=0)
    full_data = np.concatenate([padding_data, data_27], axis=0)
    F_pad = full_data.shape[0]

    # 3. Extract Channels
    pos = full_data[:, :, 0:3]       # World XYZ
    screen = full_data[:, :, 3:5]    # Screen XY
    vis = full_data[:, :, 5:6]       # Visibility
    anchor = full_data[:, :, 6:7]    # Anchor Flag
    
    # Identify sequence start (Anchor == 0)
    # This logic applies to ALL joints now, including virtual ones
    is_anchor = (anchor[:, 0, 0] == 0)

    # 4. Normalization & Centering
    screen_norm = (screen * 2.0) - 1.0
    
    # Center Screen Data around Root (MidHip is index 26)
    screen_root = screen_norm[:, I_MIDHIP:I_MIDHIP+1, :]
    screen_centered = screen_norm - screen_root
    
    # Center World Data around Root
    world_root = pos[:, I_MIDHIP:I_MIDHIP+1, :]
    world_centered = pos - world_root

    # 5. Velocity / Deltas
    delta_screen = np.zeros_like(screen_centered)
    delta_screen[1:] = screen_centered[1:] - screen_centered[:-1]
    delta_screen[is_anchor] = 0.0 # Zero velocity at sequence start

    delta_world = np.zeros_like(world_centered)
    delta_world[1:] = world_centered[1:] - world_centered[:-1]
    delta_world[is_anchor] = 0.0 # Zero velocity at sequence start

    # 6. Bone Vectors (Graph Structure)
    # A. Parent Vector: (Self - Parent) 
    parent_vecs = world_centered - world_centered[:, PARENTS]
    
    # B. Child Vector: (Child - Self)
    child_vecs = world_centered[:, CHILDREN] - world_centered

    # --- C. LEAF NODE EXTENSION LOGIC ---
    # Logic: If Child == Self (Leaf), replace Child Vec with Parent Vec.
    # This ensures endpoints have a valid direction (extending the bone).
    leaf_mask = (CHILDREN == np.arange(NUM_JOINTS))
    child_vecs[:, leaf_mask, :] = parent_vecs[:, leaf_mask, :]

    # 7. Stack Features (18 dims)
    # 3(pos) + 3(vel) + 3(parent) + 3(child) + 2(screen) + 2(screen_vel) + 1(vis) + 1(anchor)
    features = np.concatenate([
        world_centered,  # 3
        delta_world,     # 3
        parent_vecs,     # 3
        child_vecs,      # 3 (Now valid for leaves)
        screen_centered, # 2
        delta_screen,    # 2
        vis,             # 1
        anchor           # 1
    ], axis=2)
    
    # 8. Create Sliding Windows
    features_flat = features.reshape(F_pad, -1)
    
    # Clip values to prevent instability
    features_flat = np.clip(features_flat, -10.0, 10.0)
    
    strides = (features_flat.strides[0], features_flat.strides[0], features_flat.strides[1])
    X_windows = np.lib.stride_tricks.as_strided(
        features_flat, shape=(F, N, features_flat.shape[1]), strides=strides
    )
    
    M_masks = np.zeros((F, N), dtype=bool)
    
    return X_windows.copy(), M_masks