# -*- coding: utf-8 -*-
"""
TotalCapture Skeleton Configuration
===================================
22-joint skeleton for TotalCapture dataset with bone offsets from BVH.

Joint Structure:
- Index 0: Hips - position delta (dx, dy, dz, 0, 0, 0)
- Index 1: Hips - orientation delta (6D)
- Indices 2-21: Local 6D rotations for child joints
"""

import torch

# 22 joint names matching GT order
JOINT_NAMES = [
    "Hips_pos",      # 0: Position delta
    "Hips_rot",      # 1: Orientation delta
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

# Parent indices for FK computation
# -1 means root, otherwise index of parent joint
PARENTS = [
    -1,  # 0: Hips_pos (root)
    0,   # 1: Hips_rot -> Hips_pos (same joint, rotation part)
    1,   # 2: Spine -> Hips_rot
    2,   # 3: Spine1 -> Spine
    3,   # 4: Spine2 -> Spine1
    4,   # 5: Spine3 -> Spine2
    5,   # 6: Neck -> Spine3
    6,   # 7: Head -> Neck
    5,   # 8: RightShoulder -> Spine3
    8,   # 9: RightArm -> RightShoulder
    9,   # 10: RightForeArm -> RightArm
    10,  # 11: RightHand -> RightForeArm
    5,   # 12: LeftShoulder -> Spine3
    12,  # 13: LeftArm -> LeftShoulder
    13,  # 14: LeftForeArm -> LeftArm
    14,  # 15: LeftHand -> LeftForeArm
    1,   # 16: RightUpLeg -> Hips_rot
    16,  # 17: RightLeg -> RightUpLeg
    17,  # 18: RightFoot -> RightLeg
    1,   # 19: LeftUpLeg -> Hips_rot
    19,  # 20: LeftLeg -> LeftUpLeg
    20,  # 21: LeftFoot -> LeftLeg
]

# Bone offsets from BVH (converted from inches to meters: 1 inch = 0.0254m)
# Shape: [22, 3] - XYZ offset from parent joint
OFFSETS_METERS = torch.tensor([
    [0.0, 0.0, 0.0],                     # 0: Hips_pos (root origin)
    [0.0, 0.0, 0.0],                     # 1: Hips_rot (same position as Hips_pos)
    [0.0, 0.0462, -0.0693],              # 2: Spine (0, 1.82, -2.73 inches)
    [0.0, 0.0160, -0.0909],              # 3: Spine1 (0, 0.63, -3.58 inches)
    [0.0, 0.0080, -0.0920],              # 4: Spine2 (0, 0.32, -3.62 inches)
    [0.0, 0.0, -0.0923],                 # 5: Spine3 (0, 0, -3.64 inches)
    [0.0, -0.0346, -0.2424],             # 6: Neck (0, -1.36, -9.54 inches)
    [0.0, -0.0220, -0.1250],             # 7: Head (0, -0.87, -4.92 inches)
    [0.0292, -0.0486, -0.1576],          # 8: RightShoulder (1.15, -1.91, -6.20 inches)
    [0.1449, 0.0, 0.0],                  # 9: RightArm (5.71, 0, 0 inches)
    [0.2889, 0.0, 0.0],                  # 10: RightForeArm (11.37, 0, 0 inches)
    [0.2196, 0.0, 0.0],                  # 11: RightHand (8.65, 0, 0 inches)
    [-0.0292, -0.0486, -0.1576],         # 12: LeftShoulder (-1.15, -1.91, -6.20 inches)
    [-0.1449, 0.0, 0.0],                 # 13: LeftArm (-5.71, 0, 0 inches)
    [-0.2889, 0.0, 0.0],                 # 14: LeftForeArm (-11.37, 0, 0 inches)
    [-0.2196, 0.0, 0.0],                 # 15: LeftHand (-8.65, 0, 0 inches)
    [0.0866, 0.0, 0.0253],               # 16: RightUpLeg (3.41, 0, 0.99 inches)
    [0.0, 0.0, 0.3789],                  # 17: RightLeg (0, 0, 14.92 inches)
    [0.0, 0.0, 0.3754],                  # 18: RightFoot (0, 0, 14.78 inches)
    [-0.0866, 0.0, 0.0253],              # 19: LeftUpLeg (-3.41, 0, 0.99 inches)
    [0.0, 0.0, 0.3789],                  # 20: LeftLeg (0, 0, 14.92 inches)
    [0.0, 0.0, 0.3754],                  # 21: LeftFoot (0, 0, 14.78 inches)
], dtype=torch.float32)


def get_totalcapture_skeleton():
    """
    Returns skeleton configuration for TotalCapture dataset.
    
    Returns:
        dict with 'parents', 'offsets', 'joint_names'
    """
    return {
        'parents': PARENTS,
        'offsets': OFFSETS_METERS,
        'joint_names': JOINT_NAMES
    }
