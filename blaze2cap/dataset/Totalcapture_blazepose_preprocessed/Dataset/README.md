# TotalCapture BlazePose Dataset

This directory should contain the preprocessed TotalCapture dataset with BlazePose landmarks and ground truth motion data.

## 📥 Download Dataset from Hugging Face

The complete dataset (6.89GB, 10,328 files) is hosted on Hugging Face:

**Dataset URL:** https://huggingface.co/datasets/Blazewild/Totalcap-blazepose

### Method 1: Using Hugging Face CLI (Recommended)

```bash
# 1. Install huggingface-hub
pip install huggingface-hub

# 2. Navigate to the project root
cd /path/to/Blaze2Cap

# 3. Download the dataset directly to this location
huggingface-cli download Blazewild/Totalcap-blazepose \
  --repo-type dataset \
  --local-dir blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset
```

### Method 2: Using Python API

```python
from huggingface_hub import snapshot_download

# Download to this directory
snapshot_download(
    repo_id="Blazewild/Totalcap-blazepose",
    repo_type="dataset",
    local_dir="blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset"
)
```

### Method 3: Manual Download from Web

1. Visit: https://huggingface.co/datasets/Blazewild/Totalcap-blazepose/tree/main
2. Download the following folders:
   - `blaze_augmented/` (Input data)
   - `gt_augmented/` (Ground truth)
   - `dataset_map.json` (Train/test split)
3. Place them in this directory (`blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/`)

## 📁 Expected Directory Structure

After downloading, this directory should contain:

```
Dataset/
├── blaze_augmented/          # Input: BlazePose landmarks
│   ├── S1/                   # Subject 1
│   │   ├── acting1/
│   │   │   ├── cam1/
│   │   │   │   ├── blaze_S1_acting1_cam1_seg0_s1_o0.npy
│   │   │   │   ├── blaze_S1_acting1_cam1_seg0_s2_o0.npy
│   │   │   │   └── ...
│   │   │   ├── cam2/
│   │   │   └── ...
│   │   ├── freestyle1/
│   │   └── ...
│   ├── S2/                   # Subject 2
│   ├── S3/                   # Subject 3
│   ├── S4/                   # Subject 4
│   └── S5/                   # Subject 5
│
├── gt_augmented/             # Output: Ground truth motion (6D rotations)
│   ├── S1/
│   │   ├── acting1/
│   │   │   ├── cam1/
│   │   │   │   ├── gt_S1_acting1_cam1_seg0_s1_o0.npy
│   │   │   │   ├── gt_S1_acting1_cam1_seg0_s2_o0.npy
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── ...
│   ├── S2/
│   ├── S3/
│   ├── S4/
│   └── S5/
│
└── dataset_map.json          # Train/test split mapping (5,164 samples)
```

## 📊 Dataset Statistics

- **Total Samples:** 5,164
  - **Training:** 2,775 samples
  - **Testing:** 2,068 samples (excluding cam5)
- **Subjects:** 5 (S1-S5)
- **Actions:** acting1, acting2, acting3, freestyle1, freestyle2, freestyle3, rom1, rom2, rom3, walking1, walking2, walking3
- **Cameras:** 8 per action (cam5 excluded from test set)
- **Augmentation:** Temporal subsampling at 3 strides (s1=60fps, s2=30fps, s3=20fps)

### Input Data (`blaze_augmented/`)
- **Shape:** `(Frames, 25, 18)`
  - 25 BlazePose keypoints
  - 18 features per keypoint:
    - 3D world coordinates (x, y, z)
    - 2D screen coordinates (u, v)
    - Screen deltas (Δu, Δv)
    - Visibility (v)
    - ... (see `dataset_structure.md` for full details)

### Ground Truth (`gt_augmented/`)
- **Shape:** `(Frames, 22, 6)`
  - 22 joints (1 root + 21 body joints)
  - 6D rotation representation per joint

## 🔄 Generate Dataset Mapping

If `dataset_map.json` is missing or you want to regenerate it:

```bash
cd /path/to/Blaze2Cap
python blaze2cap/data/generate_json.py
```

This will scan the dataset and create the train/test split mapping.

## ✅ Verify Installation

Check that the dataset is properly installed:

```bash
# Check directory structure
ls -la blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/

# Should show:
# - blaze_augmented/
# - gt_augmented/
# - dataset_map.json

# Check sample counts
python -c "
import json
with open('blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/dataset_map.json') as f:
    data = json.load(f)
    print(f'Train samples: {len(data[\"train\"])}')
    print(f'Test samples: {len(data[\"test\"])}')
    print(f'Total samples: {len(data[\"train\"]) + len(data[\"test\"])}')
"
```

Expected output:
```
Train samples: 2775
Test samples: 2068
Total samples: 4843
```

## 📝 Notes

- **File Format:** All data files are NumPy `.npy` format
- **Naming Convention:** `{type}_{subject}_{action}_{camera}_seg{segment}_s{stride}_o{offset}.npy`
  - Example: `blaze_S1_acting1_cam1_seg0_s2_o1.npy`
- **Excluded:** Camera 5 (cam5) is excluded from the test set due to data quality issues
- **Total Size:** ~6.89GB

## 🔗 Related Files

- **Dataset Structure Documentation:** `../dataset_structure.md`
- **Data Loader:** `../../data/data_loader.py`
- **Dataset Generator:** `../../data/generate_json.py`

## 📧 Issues

If you encounter any issues downloading or using the dataset:
1. Check the Hugging Face dataset page: https://huggingface.co/datasets/Blazewild/Totalcap-blazepose
2. Open an issue on GitHub: https://github.com/BlazeWild/Blaze2Cap/issues
