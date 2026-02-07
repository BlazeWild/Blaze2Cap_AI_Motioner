# Ground Truth 22 Joint Order

## Overview
Ground truth data has shape: `(frames, 22, 6)`
- 22 joints (indices 0-21)
- 6 channels per joint (6D representation)
- **Used in training**: `gt_22_6_rootdelta_nosync` (indices 0 and 1 are DELTAS)

## Joint Order (Indices 0-21)

### Index 0: Root Position DELTA
- **Joint**: Hips (Root)
- **Data**: Position delta (dx, dy, dz, 0, 0, 0)
- **Frame 0**: (0, 0, 0, 0, 0, 0)
- **Frame N**: position[N] - position[N-1]

### Index 1: Root Orientation DELTA
- **Joint**: Hips (Root)
- **Data**: 6D rotation delta (camera-oriented)
- **Frame 0**: (0, 0, 0, 0, 0, 0)
- **Frame N**: orientation[N] - orientation[N-1]

### Indices 2-21: Child Joints (Local 6D Rotations)
All child joints store local 6D rotations (relative to parent)

| Index | Joint Name | Parent Joint |
|-------|------------|--------------|
| 2 | Spine | Hips |
| 3 | Spine1 | Spine |
| 4 | Spine2 | Spine1 |
| 5 | Spine3 | Spine2 |
| 6 | Neck | Spine3 |
| 7 | Head | Neck |
| 8 | RightShoulder | Spine3 |
| 9 | RightArm | RightShoulder |
| 10 | RightForeArm | RightArm |
| 11 | RightHand | RightForeArm |
| 12 | LeftShoulder | Spine3 |
| 13 | LeftArm | LeftShoulder |
| 14 | LeftForeArm | LeftArm |
| 15 | LeftHand | LeftForeArm |
| 16 | RightUpLeg | Hips |
| 17 | RightLeg | RightUpLeg |
| 18 | RightFoot | RightLeg |
| 19 | LeftUpLeg | Hips |
| 20 | LeftLeg | LeftUpLeg |
| 21 | LeftFoot | LeftLeg |

## Data Formats

### Main Format: Delta GT (gt_22_6_rootdelta_nosync) - USED IN TRAINING
- **Index 0**: Root position delta (dx, dy, dz, 0, 0, 0)
  - Frame 0: (0, 0, 0, 0, 0, 0)
  - Frame N: position[N] - position[N-1]
- **Index 1**: Root orientation 6D delta
  - Frame 0: (0, 0, 0, 0, 0, 0)
  - Frame N: orientation[N] - orientation[N-1]
- **Indices 2-21**: Local 6D rotations (relative to parent)

### Intermediate Format: Standard GT (gt_22_6_nosync) - NOT USED DIRECTLY
- **Index 0**: Root position (x, y, z, 0, 0, 0) - absolute position in camera coords
- **Index 1**: Root orientation 6D - global rotation
- **Indices 2-21**: Local 6D rotations - relative to parent
- **Note**: This is an intermediate step; converted to delta format before training

## Skeleton Hierarchy
```
Hips (Root)
├── Spine
│   └── Spine1
│       └── Spine2
│           └── Spine3
│               ├── Neck
│               │   └── Head
│               ├── RightShoulder
│               │   └── RightArm
│               │       └── RightForeArm
│               │           └── RightHand
│               └── LeftShoulder
│                   └── LeftArm
│                       └── LeftForeArm
│                           └── LeftHand
├── RightUpLeg
│   └── RightLeg
│       └── RightFoot
└── LeftUpLeg
    └── LeftLeg
        └── LeftFoot
```

## 6D Rotation Representation
Each joint's rotation is represented as 6D continuous rotation (first two columns of rotation matrix):
- 6 values: [r1, r2, r3, r4, r5, r6]
- Can be converted back to 3x3 rotation matrix
