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

# Bone offsets in BVH coordinate system (meters)
# These are the ORIGINAL BVH offsets converted from inches to meters
# Use these for Forward Kinematics with GT data that uses BVH coordinate rotations
# Conversion: 1 inch = 0.0254 meters
# BVH coordinate system: X=lateral (right+), Y=vertical (up+), Z=depth (back+)

OFFSETS_BVH_METERS = torch.tensor([
    [0.0, 0.0, 0.0],                                    # 0: Hips_pos (Root)
    [0.0, 0.0, 0.0],                                    # 1: Hips_rot
    [0.0, 0.0462, -0.0692],                            # 2: Spine: [0, 1.818, -2.727] * 0.0254
    [0.0, 0.0160, -0.0909],                            # 3: Spine1: [0, 0.631, -3.580] * 0.0254
    [0.0, 0.0080, -0.0920],                            # 4: Spine2: [0, 0.317, -3.622] * 0.0254
    [0.0, 0.0, -0.0923],                               # 5: Spine3: [0, 0, -3.636] * 0.0254
    [0.0, -0.0346, -0.2424],                           # 6: Neck: [0, -1.363, -9.543] * 0.0254
    [0.0, -0.0221, -0.1250],                           # 7: Head: [0, -0.868, -4.923] * 0.0254
    [0.0292, -0.0486, -0.1576],                        # 8: RightShoulder: [1.149, -1.914, -6.203] * 0.0254
    [0.1449, 0.0, 0.0],                                # 9: RightArm: [5.705, 0, 0] * 0.0254
    [0.2889, 0.0, 0.0],                                # 10: RightForeArm: [11.373, 0, 0] * 0.0254
    [0.2196, 0.0, 0.0],                                # 11: RightHand: [8.646, 0, 0] * 0.0254
    [-0.0292, -0.0486, -0.1576],                       # 12: LeftShoulder: [-1.149, -1.914, -6.203] * 0.0254
    [-0.1449, 0.0, 0.0],                               # 13: LeftArm: [-5.705, 0, 0] * 0.0254
    [-0.2889, 0.0, 0.0],                               # 14: LeftForeArm: [-11.373, 0, 0] * 0.0254
    [-0.2196, 0.0, 0.0],                               # 15: LeftHand: [-8.646, 0, 0] * 0.0254
    [0.0866, 0.0, 0.0253],                             # 16: RightUpLeg: [3.408, 0, 0.996] * 0.0254
    [0.0, 0.0, 0.3789],                                # 17: RightLeg: [0, 0, 14.917] * 0.0254
    [0.0, 0.0, 0.3754],                                # 18: RightFoot: [0, 0, 14.781] * 0.0254
    [-0.0866, 0.0, 0.0253],                            # 19: LeftUpLeg: [-3.408, 0, 0.996] * 0.0254
    [0.0, 0.0, 0.3789],                                # 20: LeftLeg: [0, 0, 14.917] * 0.0254
    [0.0, 0.0, 0.3754],                                # 21: LeftFoot: [0, 0, 14.781] * 0.0254
], dtype=torch.float32)



# OFFSETS_BVH_METERS = torch.tensor([
#     [0.0, 0.0, 0.0],                                    # 0: Hips_pos (Root)
#     [0.0, 0.0, 0.0],                                    # 1: Hips_rot
#     [0.0, -0.0693, 0.0462],                            # 2: Spine
#     [0.0, -0.0909, 0.0160],                            # 3: Spine1
#     [0.0, -0.0920, 0.0080],                            # 4: Spine2
#     [0.0, -0.0923, 0.0],                               # 5: Spine3
#     [0.0, -0.2424, -0.0346],                           # 6: Neck
#     [0.0, -0.1250, -0.0221],                           # 7: Head
#     [-0.0292, -0.1576, -0.0486],                       # 8: RightShoulder
#     [-0.1449, 0.0, 0.0],                               # 9: RightArm
#     [-0.2889, 0.0, 0.0],                               # 10: RightForeArm
#     [-0.2196, 0.0, 0.0],                               # 11: RightHand
#     [0.0292, -0.1576, -0.0486],                        # 12: LeftShoulder
#     [0.1449, 0.0, 0.0],                                # 13: LeftArm
#     [0.2889, 0.0, 0.0],                                # 14: LeftForeArm
#     [0.2196, 0.0, 0.0],                                # 15: LeftHand
#     [-0.0866, 0.0253, 0.0],                            # 16: RightUpLeg
#     [0.0, 0.3789, 0.0],                                # 17: RightLeg
#     [0.0, 0.3754, 0.0],                                # 18: RightFoot
#     [0.0866, 0.0253, 0.0],                             # 19: LeftUpLeg
#     [0.0, 0.3789, 0.0],                                # 20: LeftLeg
#     [0.0, 0.3754, 0.0],                                # 21: LeftFoot
# ], dtype=torch.float32)
# --- ALIGNMENT MATRIX ---
# Convert BVH (X right, Y up, Z back) to target (X right, Y down, Z forward)
# X' =  X
# Y' = -Y
# Z' = -Z
R_ALIGN = np.array([
    [ 1,  0,  0],
    [ 0, 1,  0],
    [ 0,  0, 1]
    ])

# Apply Matrix to Offsets
_offsets_np = (R_ALIGN @ OFFSETS_BVH_METERS.numpy().T).T
OFFSETS_METERS = torch.tensor(_offsets_np, dtype=torch.float32)

def get_totalcapture_skeleton():
    return {
        'parents': PARENTS,
        'offsets': OFFSETS_METERS,           
        'offsets_bvh': OFFSETS_BVH_METERS,   
        'joint_names': JOINT_NAMES
    }