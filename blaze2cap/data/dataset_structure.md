# Dataset Structure Documentation

## Overview

This dataset contains temporally augmented BlazePose keypoint data and corresponding ground truth (GT) skeletal motion data from the TotalCapture dataset. The data is organized for training pose-to-motion models that predict 3D skeletal motion from 2D/3D pose landmarks.

## Dataset Location

```
training_dataset_both_in_out/
├── blaze_augmented/    # Input: BlazePose keypoint data
└── gt_augmented/       # Output: Ground truth skeletal motion
```

## Dataset Contents

### Data Provenance

The augmented dataset is derived from:
- **Source**: TotalCapture dataset (5 subjects, multiple actions, 8 camera views)
- **Raw BlazePose**: 33 keypoints × 10 channels at 60fps
- **Raw GT**: BVH skeleton data with 21 joints

### Processing Pipeline Summary

The data underwent the following pipeline (see [DATA_PIPELINE_README.md](../../../training_dataset_both_in_out/DATA_PIPELINE_README.md) for details):

1. **BlazePose Filtering**: 33 keypoints → 25 keypoints (removed face landmarks), 10 channels → 7 channels
2. **GT Generation**: Raw BVH → 22×6 format (root deltas + 20 joint local rotations in 6D)
3. **Synchronization**: Frame alignment, visibility filtering, anchor frame detection
4. **Augmentation**: Temporal subsampling at multiple strides to create training variations

## Folder Structure

Both `blaze_augmented` and `gt_augmented` follow the same hierarchical structure:

```
{blaze|gt}_augmented/
├── S1/                    # Subject 1 (full dataset: 12 actions)
│   ├── acting1/
│   │   ├── cam1/         # Camera 1 view
│   │   │   ├── {prefix}_S1_acting1_cam1_seg0_s1_o0.npy
│   │   │   ├── {prefix}_S1_acting1_cam1_seg0_s2_o0.npy
│   │   │   ├── {prefix}_S1_acting1_cam1_seg0_s2_o1.npy
│   │   │   ├── {prefix}_S1_acting1_cam1_seg0_s3_o0.npy
│   │   │   └── ... (multiple stride/offset combinations)
│   │   ├── cam2/
│   │   ├── ... cam3-8
│   ├── acting2/
│   ├── acting3/
│   ├── freestyle1/
│   ├── freestyle2/
│   ├── freestyle3/
│   ├── rom1/
│   ├── rom2/
│   ├── rom3/
│   ├── walking1/
│   ├── walking2/
│   └── walking3/
├── S2/                    # Subject 2 (same 12 actions)
├── S3/                    # Subject 3 (same 12 actions)
├── S4/                    # Subject 4 (5 actions: acting3, freestyle1, freestyle3, rom3, walking2)
└── S5/                    # Subject 5 (5 actions: acting3, freestyle1, freestyle3, rom3, walking2)
```

**Prefix naming convention:**
- `blaze_augmented/`: Files start with `blaze_`
- `gt_augmented/`: Files start with `gt_`

## File Naming Convention

Each `.npy` file follows this naming pattern:

```
{prefix}_{subject}_{action}_cam{N}_seg{S}_s{stride}_o{offset}.npy
```

**Components:**
- `prefix`: `blaze` or `gt`
- `subject`: S1, S2, S3, S4, S5
- `action`: acting1-3, freestyle1-3, rom1-3, walking1-3
- `cam{N}`: Camera ID (1-8)
- `seg{S}`: Segment index (0, 1, 2, ...) - split by scene cuts/anchor frames
- `s{stride}`: Temporal stride (1=60fps, 2=30fps, 3=20fps equivalent)
- `o{offset}`: Starting offset for subsampling (0 to stride-1)

**Example:**
```
blaze_S1_acting1_cam3_seg0_s2_o1.npy
```
- Subject 1, Acting1, Camera 3, Segment 0
- Stride 2 (30fps equivalent), Offset 1
- Starts from frame 1, takes every 2nd frame

## Data Formats

### BlazePose Augmented (`blaze_augmented/`)

**Shape:** `(frames, 25, 7)`

**25 Keypoints (indices 0-24):**

| New Index | Keypoint Name | Original Index |
| :--- | :--- | :--- |
| 0 | Nose | 0 |
| 1 | Left_ear | 7 |
| 2 | Right_ear | 8 |
| 3 | Left_shoulder | 11 |
| 4 | Right_shoulder | 12 |
| 5 | Left_elbow | 13 |
| 6 | Right_elbow | 14 |
| 7 | Left_wrist | 15 |
| 8 | Right_wrist | 16 |
| 9 | Left_pinky | 17 |
| 10 | Right_pinky | 18 |
| 11 | Left_index | 19 |
| 12 | Right_index | 20 |
| 13 | Left_thumb | 21 |
| 14 | Right_thumb | 22 |
| 15 | Left_hip | 23 |
| 16 | Right_hip | 24 |
| 17 | Left_knee | 25 |
| 18 | Right_knee | 26 |
| 19 | Left_ankle | 27 |
| 20 | Right_ankle | 28 |
| 21 | Left_heel | 29 |
| 22 | Right_heel | 30 |
| 23 | Left_foot_index | 31 |
| 24 | Right_foot_index | 32 |

**after cleaning,27 ones are:**

| Index | Name             | Name of Child                           | Name of Parent |
| ----: | ---------------- | --------------------------------------- | -------------- |
|     0 | Nose             | Left_ear | Right_ear                    | Neck           |
|     1 | Left_ear         | None                                    | Nose           |
|     2 | Right_ear        | None                                    | Nose           |
|     3 | Left_shoulder    | Left_elbow                              | Neck           |
|     4 | Right_shoulder   | Right_elbow                             | Neck           |
|     5 | Left_elbow       | Left_wrist                              | Left_shoulder  |
|     6 | Right_elbow      | Right_wrist                             | Right_shoulder |
|     7 | Left_wrist       | Left_pinky | Left_index | Left_thumb    | Left_elbow     |
|     8 | Right_wrist      | Right_pinky | Right_index | Right_thumb | Right_elbow    |
|     9 | Left_pinky       | None                                    | Left_wrist     |
|    10 | Right_pinky      | None                                    | Right_wrist    |
|    11 | Left_index       | None                                    | Left_wrist     |
|    12 | Right_index      | None                                    | Right_wrist    |
|    13 | Left_thumb       | None                                    | Left_wrist     |
|    14 | Right_thumb      | None                                    | Right_wrist    |
|    15 | Left_hip         | Left_knee                               | MidHip         |
|    16 | Right_hip        | Right_knee                              | MidHip         |
|    17 | Left_knee        | Left_ankle                              | Left_hip       |
|    18 | Right_knee       | Right_ankle                             | Right_hip      |
|    19 | Left_ankle       | Left_heel | Left_foot_index             | Left_knee      |
|    20 | Right_ankle      | Right_heel | Right_foot_index           | Right_knee     |
|    21 | Left_heel        | None                                    | Left_ankle     |
|    22 | Right_heel       | None                                    | Right_ankle    |
|    23 | Left_foot_index  | None                                    | Left_ankle     |
|    24 | Right_foot_index | None                                    | Right_ankle    |
|    25 | Neck             | Nose | Left_shoulder | Right_shoulder   | MidHip         |
|    26 | MidHip           | Neck | Left_hip | Right_hip             | None (Root)    |




**7 Channels (per keypoint):**

| Channel | Content | Description | Range/Units |
|---------|---------|-------------|-------------|
| 0 | world_x | 3D X coordinate in camera space | meters |
| 1 | world_y | 3D Y coordinate in camera space | meters |
| 2 | world_z | 3D Z coordinate in camera space | meters |
| 3 | screen_x | 2D screen X coordinate | [0, 1] normalized |
| 4 | screen_y | 2D screen Y coordinate | [0, 1] normalized |
| 5 | visibility | Keypoint visibility confidence | [0, 1] |
| 6 | anchor_flag | Scene cut marker | 0=anchor, 1=regular |

**Key Properties:**
- First frame of each segment is always an anchor frame (channel 6 = 0)
- Screen coordinates are raw [0-1] range (no delta transformation applied)
- All frames contain visible subject data (filtered during synchronization)

### Ground Truth Augmented (`gt_augmented/`)

**Shape:** `(frames, 22, 6)`

**22 Indices:**

| Index | Joint | Content Type | Description |
|-------|-------|--------------|-------------|
| 0 | Hips (Root) | Position Delta | `[dx, dy, dz, 0, 0, 0]` in meters |
| 1 | Hips (Root) | Rotation Delta | 6D rotation delta from previous frame |
| 2 | Spine | Local Rotation | 6D rotation relative to Hips |
| 3 | Spine1 | Local Rotation | 6D rotation relative to Spine |
| 4 | Spine2 | Local Rotation | 6D rotation relative to Spine1 |
| 5 | Spine3 | Local Rotation | 6D rotation relative to Spine2 |
| 6 | Neck | Local Rotation | 6D rotation relative to Spine3 |
| 7 | Head | Local Rotation | 6D rotation relative to Neck |
| 8 | RightShoulder | Local Rotation | 6D rotation relative to Spine3 |
| 9 | RightArm | Local Rotation | 6D rotation relative to RightShoulder |
| 10 | RightForeArm | Local Rotation | 6D rotation relative to RightArm |
| 11 | RightHand | Local Rotation | 6D rotation relative to RightForeArm |
| 12 | LeftShoulder | Local Rotation | 6D rotation relative to Spine3 |
| 13 | LeftArm | Local Rotation | 6D rotation relative to LeftShoulder |
| 14 | LeftForeArm | Local Rotation | 6D rotation relative to LeftArm |
| 15 | LeftHand | Local Rotation | 6D rotation relative to LeftForeArm |
| 16 | RightUpLeg | Local Rotation | 6D rotation relative to Hips |
| 17 | RightLeg | Local Rotation | 6D rotation relative to RightUpLeg |
| 18 | RightFoot | Local Rotation | 6D rotation relative to RightLeg |
| 19 | LeftUpLeg | Local Rotation | 6D rotation relative to Hips |
| 20 | LeftLeg | Local Rotation | 6D rotation relative to LeftUpLeg |
| 21 | LeftFoot | Local Rotation | 6D rotation relative to LeftLeg |

**6 Channels (per index):**
- **For index 0 (root position):** `[dx, dy, dz, 0, 0, 0]` - translation delta in meters, last 3 are padding
- **For indices 1-21 (rotations):** `[r1, r2, r3, r4, r5, r6]` - 6D rotation representation (first two columns of rotation matrix)

**Key Properties:**
- First frame of each segment: root deltas (indices 0-1) are zero
- Child rotations (indices 2-21) are local rotations relative to parent joint, not deltas
- Deltas are recalculated after subsampling to maintain proper motion flow

## Augmentation Strategy

The augmentation process (see [data_augmentation_gt_blazepose.py](../../../training_dataset_both_in_out/data_augmentation_gt_blazepose.py)) creates multiple temporal variations:

### Segmentation by Anchor Frames
- Original synchronized data is split at anchor frames (scene cuts)
- Each segment is processed independently
- Prevents interpolation/delta accumulation across scene boundaries

### Temporal Subsampling
Three stride values simulate different frame rates:
- **Stride 1**: Original 60fps (baseline)
- **Stride 2**: 30fps equivalent (every 2nd frame)
- **Stride 3**: 20fps equivalent (every 3rd frame)

For each stride, multiple offsets create overlapping sequences:
- Stride 2: offsets [0, 1] → 2 variants per segment
- Stride 3: offsets [0, 1, 2] → 3 variants per segment

**Total augmentation factor:** ~6x per segment (1 + 2 + 3 variants)

### Delta Recalculation for GT

Since GT contains temporal deltas (root position/rotation), simple slicing would create incorrect motion:
- **Problem**: Delta over 1 frame ≠ delta over 2 frames
- **Solution**: 
  1. Reconstruct absolute positions/rotations from cumulative deltas
  2. Subsample absolute trajectory
  3. Recalculate deltas from subsampled frames

BlazePose data is sliced directly as it contains raw positions (no deltas).

## Dataset Statistics

### Total Data Volume
- **Subjects**: 5 (S1-S3 full, S4-S5 partial)
- **Actions per full subject**: 12
- **Cameras per action**: 8
- **Segments per camera**: Variable (1-10+, depending on scene cuts)
- **Augmented variants per segment**: ~6 (stride/offset combinations)

**Estimated total files**: ~20,000-40,000 `.npy` files per dataset folder

### Frame Counts
- Original sequences: 100-3000 frames (varies by action)
- After segmentation: 10-500 frames per segment
- After subsampling: 5-250 frames per variant
- Minimum sequence length: 2 frames (segments < 2 frames are discarded)

## Data Alignment

### Synchronization Guarantees
For any matching pair:
```
blaze_augmented/S1/acting1/cam3/blaze_S1_acting1_cam3_seg0_s2_o1.npy
gt_augmented/S1/acting1/cam3/gt_S1_acting1_cam3_seg0_s2_o1.npy
```

**Guaranteed properties:**
- ✅ Same number of frames
- ✅ Temporal alignment (frame i in BlazePose corresponds to frame i in GT)
- ✅ Same subject/action/camera/segment
- ✅ Same temporal stride and offset
- ✅ Both start with anchor frame (BlazePose flag=0, GT deltas=0)

### Loading Example

```python
import numpy as np

# Load matching pair
blaze = np.load("blaze_augmented/S1/acting1/cam3/blaze_S1_acting1_cam3_seg0_s2_o1.npy")
gt = np.load("gt_augmented/S1/acting1/cam3/gt_S1_acting1_cam3_seg0_s2_o1.npy")

print(f"BlazePose shape: {blaze.shape}")  # (frames, 25, 7)
print(f"GT shape: {gt.shape}")            # (frames, 22, 6)
print(f"Frame count match: {blaze.shape[0] == gt.shape[0]}")  # True

# Verify anchor frame
print(f"First frame is anchor: {blaze[0, 0, 6] == 0}")  # True
print(f"GT root pos delta at anchor: {gt[0, 0, :3]}")   # [0, 0, 0]
```

## Usage Notes

### For Training
1. **Pairing**: Always load matching BlazePose and GT files (same filename pattern)
2. **Anchor handling**: First frame of each sequence is a reset point
3. **Coordinate systems**: 
   - BlazePose world coords are in camera space (not world space)
   - GT is also in camera space (camera-specific files)
4. **Temporal consistency**: Deltas in GT maintain motion continuity within segments

### For Data Loading
- Use filename regex to parse metadata: `r"(blaze|gt)_(S\d)_(\w+)_cam(\d)_seg(\d+)_s(\d+)_o(\d+)\.npy"`
- Filter by subject/action/camera for specific training splits
- Consider stride/offset for curriculum learning (start with stride 1, progress to 2-3)

### Data Splits
The dataset does NOT include pre-defined train/val/test splits. Common strategies:
- **Subject-based**: Train on S1-S3, validate on S4, test on S5
- **Action-based**: Hold out specific actions (e.g., freestyle3)
- **Camera-based**: Train on cam1-6, validate on cam7, test on cam8

## References

For detailed pipeline documentation, see:
- [DATA_PIPELINE_README.md](../../../training_dataset_both_in_out/DATA_PIPELINE_README.md) - Complete data processing pipeline
- [data_augmentation_gt_blazepose.py](../../../training_dataset_both_in_out/data_augmentation_gt_blazepose.py) - Augmentation implementation

---

**Last Updated:** February 2026  
**Dataset Version:** v1.0 (Augmented)
