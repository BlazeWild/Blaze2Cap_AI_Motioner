import os
import json
import numpy as np
import torch
from torch.utils import data

# ==========================================
# 1. SHARED CONFIGURATION (Module Level)
# ==========================================
# Move hierarchy definitions here so they are accessible to the function below
PARENTS = np.arange(25)
CHILDREN = np.arange(25)

# Define Hierarchy
CHILDREN[3], CHILDREN[4] = 5, 6
PARENTS[5], PARENTS[6] = 3, 4
CHILDREN[5], CHILDREN[6] = 7, 8
PARENTS[7], PARENTS[8] = 5, 6
CHILDREN[7], CHILDREN[8] = 11, 12
PARENTS[[9, 11, 13]] = 7
PARENTS[[10, 12, 14]] = 8
CHILDREN[15], CHILDREN[16] = 17, 18
PARENTS[17], PARENTS[18] = 15, 16
CHILDREN[17], CHILDREN[18] = 19, 20
PARENTS[19], PARENTS[20] = 17, 18
CHILDREN[19], CHILDREN[20] = 23, 24
PARENTS[[21, 23]] = 19
PARENTS[[22, 24]] = 20

# ==========================================
# 2. SHARED PROCESSING FUNCTION
# ==========================================
def process_blazepose_frames(raw_data, window_size):
    """
    Standalone function to turn Raw BlazePose (F, 25, 7) into Features (F, Window, 450).
    Can be used by Dataset AND Inference script.
    """
    N = window_size
    F, J, _ = raw_data.shape
    
    # 1. Padding Strategy (Input): Valid Static History
    padding_data = np.repeat(raw_data[0:1], N-1, axis=0) 
    full_data = np.concatenate([padding_data, raw_data], axis=0)
    F_pad = full_data.shape[0]

    # 2. Extract Components
    pos_raw = full_data[:, :, 0:3]
    screen_raw_01 = full_data[:, :, 3:5]
    vis_raw = full_data[:, :, 5:6]
    anchor_raw = full_data[:, :, 6:7] 
    
    is_anchor = (anchor_raw[:, 0, 0] == 0)

    # 3. Transform Screen
    screen_norm = (screen_raw_01 * 2.0) - 1.0

    # 4. Feature: Centered Screen Position
    screen_hip_center = (screen_norm[:, 15] + screen_norm[:, 16]) / 2.0
    screen_centered = screen_norm - screen_hip_center[:, None, :]

    # 5. Feature: Delta Screen
    delta_screen = np.zeros_like(screen_centered)
    delta_screen[1:] = screen_centered[1:] - screen_centered[:-1]
    delta_screen[is_anchor] = 0.0

    # 6. Feature: World Handling
    world_hip_center = (pos_raw[:, 15] + pos_raw[:, 16]) / 2.0
    world_centered = pos_raw - world_hip_center[:, None, :]
    
    delta_world = np.zeros_like(world_centered)
    delta_world[1:] = world_centered[1:] - world_centered[:-1]
    delta_world[is_anchor] = 0.0

    # 7. Feature: Vectors (Bone/Parent)
    # Uses module-level CHILDREN/PARENTS constants
    bone_vecs = world_centered[:, CHILDREN] - world_centered
    leaf_mask = (CHILDREN == np.arange(25))
    bone_vecs[:, leaf_mask] = 0 
    
    parent_vecs = world_centered - world_centered[:, PARENTS]

    # 8. Stack Features
    features = np.concatenate([
        world_centered, delta_world, bone_vecs, parent_vecs,
        screen_centered, delta_screen, vis_raw, anchor_raw
    ], axis=2) 
    
    # Flatten
    D = 18 * 25
    features_flat = features.reshape(F_pad, D)
    
    # Clip (Important for stability)
    features_flat = np.clip(features_flat, -10.0, 10.0)
    
    # Windowing
    strides = (features_flat.strides[0], features_flat.strides[0], features_flat.strides[1])
    X_windows = np.lib.stride_tricks.as_strided(
        features_flat, shape=(F, N, D), strides=strides
    )
    
    # Masking (All False = Valid)
    M_masks = np.zeros((F, N), dtype=bool) 
    
    return X_windows.copy(), M_masks


# ==========================================
# 3. DATASET CLASS (Now simpler!)
# ==========================================
class PoseSequenceDataset(data.Dataset):
    def __init__(self, dataset_root: str, window_size: int, split: str = "train"):
        self.dataset_root = dataset_root
        self.window_size = window_size
        
        json_path = os.path.join(dataset_root, "dataset_map.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON map not found at {json_path}")
        
        with open(json_path, 'r') as f:
            full_data = json.load(f)
            
        self.samples = [item for item in full_data if item.get(f"split_{split}", False)]
        print(f"[{split.upper()}] Loader Initialized. Samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        input_path = os.path.join(self.dataset_root, item["source"])
        target_path = os.path.join(self.dataset_root, item["target"])
        
        input_data = np.load(input_path).astype(np.float32)
        gt_data = np.load(target_path).astype(np.float32)

        input_data = np.nan_to_num(input_data, nan=0.0)
        gt_data = np.nan_to_num(gt_data, nan=0.0)
        
        min_len = min(len(input_data), len(gt_data))
        input_data = input_data[:min_len]
        gt_data = gt_data[:min_len]
        
        # --- CALL THE SHARED FUNCTION ---
        X, M = process_blazepose_frames(input_data, self.window_size)

        # Process GT (Windowing Logic)
        N = self.window_size
        F = min_len
        Y_flat = gt_data.reshape(F, -1)
        padding_gt = np.repeat(Y_flat[0:1], N-1, axis=0)
        full_gt = np.concatenate([padding_gt, Y_flat], axis=0)
        
        strides = (full_gt.strides[0], full_gt.strides[0], full_gt.strides[1])
        Y_windows = np.lib.stride_tricks.as_strided(
            full_gt, shape=(F, N, 132), strides=strides
        )
        Y_windows = np.nan_to_num(Y_windows, nan=0.0)
        Y_windows = np.clip(Y_windows, -2.0, 2.0)
        
        return {
            "source": torch.from_numpy(X.copy()),
            "mask": torch.from_numpy(M.copy()),
            "target": torch.from_numpy(Y_windows.copy())
        }