# data_loader_posonly_angle.py

import os
import json
import numpy as np
import torch
from torch.utils import data

# IMPORT FROM THE UPDATED PROCESSING FILE
from blaze2cap.modules.pose_processing_posonly_angle import process_blazepose_frames

class PoseSequenceDataset(data.Dataset):
    def __init__(self, dataset_root, window_size, split="train", max_windows=None):
        self.dataset_root = dataset_root
        self.window_size = window_size
        self.max_windows = max_windows
        
        json_path = os.path.join(dataset_root, "dataset_map.json")
        with open(json_path, 'r') as f:
            full_data = json.load(f)
            
        self.samples = [item for item in full_data if item.get(f"split_{split}", False)]
        print(f"[{split.upper()}] Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        input_path = os.path.join(self.dataset_root, item["source"])
        target_path = os.path.join(self.dataset_root, item["target"])
        
        # Load Raw
        input_data = np.nan_to_num(np.load(input_path).astype(np.float32))
        gt_raw = np.nan_to_num(np.load(target_path).astype(np.float32))
        
        min_len = min(len(input_data), len(gt_raw))
        input_data = input_data[:min_len]
        gt_raw = gt_raw[:min_len]
        
        # 1. Process Input (27 Joints, 14 Channels)
        X_windows, _ = process_blazepose_frames(input_data, self.window_size)
        
        # 2. Window Target (GT Rotations -> 22 Joints, 6D)
        N = self.window_size
        F = gt_raw.shape[0]
        
        # Padding
        pad_gt = np.repeat(gt_raw[0:1], N-1, axis=0)
        full_gt = np.concatenate([pad_gt, gt_raw], axis=0)
        
        s0, s1, s2 = full_gt.strides
        # (F, N, 22, 6)
        Y_windows = np.lib.stride_tricks.as_strided(
            full_gt, shape=(F, N, 22, 6), strides=(s0, s0, s1, s2)
        )
        
        # 3. Subsampling
        if self.max_windows is not None and len(X_windows) > self.max_windows:
            indices = np.linspace(0, len(X_windows)-1, self.max_windows, dtype=int)
            X_windows = X_windows[indices]
            Y_windows = Y_windows[indices]

        return {
            "source": torch.from_numpy(X_windows.copy()), 
            "target": torch.from_numpy(Y_windows.copy())
        }