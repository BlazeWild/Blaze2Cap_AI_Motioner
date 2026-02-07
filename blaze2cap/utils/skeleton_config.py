# -*- coding: utf-8 -*-
"""
TotalCapture Skeleton Configuration (Re-oriented)
=================================================
22-joint skeleton for TotalCapture dataset.

Coordinate System (User Specified):
- X+: Right
- Y+: Down
- Z+: Forward (-Z towards camera)
"""

import torch
import numpy as np

# 22 joint names (Unchanged)
JOINT_NAMES = [
    "Hips_pos",      # 0
    "Hips_rot",      # 1
    "Spine",         # 2
    "Spine1",        # 3
    "Spine2",        # 4
    "Spine3",        # 5
    "Neck",          # 6
    "Head",          # 7
    "RightShoulder", # 8
    "RightArm",      # 9
    "RightForeArm",  # 10
    "RightHand",     # 11
    "LeftShoulder",  # 12
    "LeftArm",       # 13
    "LeftForeArm",   # 14
    "LeftHand",      # 15
    "RightUpLeg",    # 16
    "RightLeg",      # 17
    "RightFoot",     # 18
    "LeftUpLeg",     # 19
    "LeftLeg",       # 20
    "LeftFoot",      # 21
]

# Parent indices (Unchanged)
PARENTS = [
    -1, 0, 1, 2, 3, 4, 5, 6, 5, 8, 9, 10, 5, 12, 13, 14, 1, 16, 17, 1, 19, 20
]

# Bone offsets re-mapped: (X, Y, Z) -> (X, Z, Y)
# New Interpretation:
# - X: Lateral (Positive = Right)
# - Y: Vertical (Positive = Down)
# - Z: Depth    (Positive = Forward)
OFFSETS_METERS = torch.tensor([
    [0.0, 0.0, 0.0],                     # 0: Hips_pos (Root)
    [0.0, 0.0, 0.0],                     # 1: Hips_rot
    [0.0, -0.0693, 0.0462],              # 2: Spine (Up in Y-, Back/Fwd in Z+)
    [0.0, -0.0909, 0.0160],              # 3: Spine1
    [0.0, -0.0920, 0.0080],              # 4: Spine2
    [0.0, -0.0923, 0.0],                 # 5: Spine3
    [0.0, -0.2424, -0.0346],             # 6: Neck (Large negative Y = Up)
    [0.0, -0.1250, -0.0220],             # 7: Head
    [0.0292, -0.1576, -0.0486],          # 8: RightShoulder
    [0.1449, 0.0, 0.0],                  # 9: RightArm (Positive X = Right)
    [0.2889, 0.0, 0.0],                  # 10: RightForeArm
    [0.2196, 0.0, 0.0],                  # 11: RightHand
    [-0.0292, -0.1576, -0.0486],         # 12: LeftShoulder
    [-0.1449, 0.0, 0.0],                 # 13: LeftArm (Negative X = Left)
    [-0.2889, 0.0, 0.0],                 # 14: LeftForeArm
    [-0.2196, 0.0, 0.0],                 # 15: LeftHand
    [0.0866, 0.0253, 0.0],               # 16: RightUpLeg
    [0.0, 0.3789, 0.0],                  # 17: RightLeg (Positive Y = Down)
    [0.0, 0.3754, 0.0],                  # 18: RightFoot
    [-0.0866, 0.0253, 0.0],              # 19: LeftUpLeg
    [0.0, 0.3789, 0.0],                  # 20: LeftLeg
    [0.0, 0.3754, 0.0],                  # 21: LeftFoot
], dtype=torch.float32)

def get_totalcapture_skeleton():
    return {
        'parents': PARENTS,
        'offsets': OFFSETS_METERS,
        'joint_names': JOINT_NAMES
    }

def get_raw_skeleton(sample_data_xyz):
    """
    Calculate bone offsets from a data sample (e.g. Frame 0 of a sequence).
    Args:
        sample_data_xyz: (25, 3) or (1, 25, 3) numpy array or tensor.
                         Expects WORLD coordinates (Meters).
    Returns:
        offsets: (22, 3) tensor of bone offsets relative to parents.
    """
    if isinstance(sample_data_xyz, np.ndarray):
        sample_data_xyz = torch.from_numpy(sample_data_xyz)
    
    if sample_data_xyz.dim() == 3:
        # Use Frame 0
        joints = sample_data_xyz[0] # (25, 3)
    else:
        joints = sample_data_xyz # (25, 3)
        
    # --- MAPPING LOGIC (Deduced from Data Structure) ---
    # Input Data (25 Joints) appears to be disjoint chains:
    # Right Side (Pos X): 15 (Hip) -> 17 -> 19 -> 21
    #                     3 (Shldr) -> 5 -> 7 -> 9
    # Left Side (Neg X):  16 (Hip) -> 18 -> 20 -> 22
    #                     4 (Shldr) -> 6 -> 8 -> 10
    # Face: 0, 1, 2
    
    # Model Skeleton (22 Joints - TotalCapture):
    # Root: MidHip
    # Spine Chain: Interpolated between MidHip and MidShoulder
    
    # helper for indexing
    def get_pos(idx):
        return joints[idx]

    # 1. Virtual Roots
    mid_hip = (get_pos(15) + get_pos(16)) / 2.0
    mid_shldr = (get_pos(3) + get_pos(4)) / 2.0
    
    # 2. Construct Positions for 22 Model Joints
    # Layout:
    # 0: Hips_pos, 1: Hips_rot, 2: Spine, 3: Spine1, 4: Spine2, 5: Spine3
    # 6: Neck, 7: Head
    # 8-11: Right Arm, 12-15: Left Arm
    # 16-18: Right Leg, 19-21: Left Leg
    
    model_pos = torch.zeros((22, 3), device=joints.device)
    
    # Spine Chain
    model_pos[0] = mid_hip # Hips_pos
    model_pos[1] = mid_hip # Hips_rot
    
    # Interpolate Spine (Approximation)
    # Spine (2) ~ 25% up
    model_pos[2] = mid_hip + (mid_shldr - mid_hip) * 0.25
    # Spine1 (3) ~ 50% up
    model_pos[3] = mid_hip + (mid_shldr - mid_hip) * 0.50
    # Spine2 (4) ~ 75% up
    model_pos[4] = mid_hip + (mid_shldr - mid_hip) * 0.75
    # Spine3 (5) = Mid Shoulder
    model_pos[5] = mid_shldr
    
    # Head/Neck
    # Head (7) -> Input 0 (Nose/Face)
    # Neck (6) -> Interpolate between Spine3 and Head
    head_pos = get_pos(0)
    model_pos[7] = head_pos
    model_pos[6] = mid_shldr + (head_pos - mid_shldr) * 0.50
    
    # Arms (Right = 3->5->7->9, Left = 4->6->8->10)
    # Right
    model_pos[8] = get_pos(3)  # RShoulder
    model_pos[9] = get_pos(5)  # RArm (Elbow)
    model_pos[10] = get_pos(7) # RForeArm (Wrist)
    model_pos[11] = get_pos(9) # RHand (Hand)
    
    # Left
    model_pos[12] = get_pos(4) # LShoulder
    model_pos[13] = get_pos(6) # LArm
    model_pos[14] = get_pos(8) # LForeArm
    model_pos[15] = get_pos(10) # LHand
    
    # Legs (Right = 15->17->19, Left = 16->18->20)
    # Right
    model_pos[16] = get_pos(15) # RUpLeg (HipJoint)
    model_pos[17] = get_pos(17) # RLeg (Knee)
    model_pos[18] = get_pos(19) # RFoot (Ankle)
    
    # Left
    model_pos[19] = get_pos(16) # LUpLeg
    model_pos[20] = get_pos(18) # LLeg
    model_pos[21] = get_pos(20) # LFoot
    
    # 3. Calculate Offsets (Child - Parent)
    offsets = torch.zeros((22, 3), dtype=torch.float32, device=joints.device)
    
    for i in range(22):
        if i == 0:
            offsets[i] = 0.0 # Root
            continue
            
        p = PARENTS[i]
        # P_child = P_parent + Offset
        # Offset = P_child - P_parent
        offsets[i] = model_pos[i] - model_pos[p]
        
    return offsets
