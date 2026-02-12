# data_loader.py
import os
import json
import numpy as np
import torch
from torch.utils import data

# --- STANDARD IMPORT ---
# This matches the final 'pose_processing.py' we just created.
from blaze2cap.modules.pose_processing import process_blazepose_frames
class PoseSequenceDataset(data.Dataset):
    def __init__(self, dataset_root, window_size, split="train", max_windows=None):
        self.dataset_root = dataset_root
        self.window_size = window_size
        self.max_windows = max_windows
        
        # Load Dataset Map
        json_path = os.path.join(dataset_root, "dataset_map.json")
        with open(json_path, 'r') as f:
            full_data = json.load(f)
            
        # Filter by split
        self.samples = [item for item in full_data if item.get(f"split_{split}", False)]
        print(f"[{split.upper()}] Loaded {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # Paths
        input_path = os.path.join(self.dataset_root, item["source"])
        target_path = os.path.join(self.dataset_root, item["target"])
        
        # Load Data
        input_data = np.nan_to_num(np.load(input_path).astype(np.float32))
        gt_data = np.nan_to_num(np.load(target_path).astype(np.float32))
        
        # Ensure lengths match exactly
        min_len = min(len(input_data), len(gt_data))
        
        # 1. PROCESS INPUT -> (F, N, 27, 20)
        # Uses new logic (20 features including 6D alignment)
        X_windows, M_masks = process_blazepose_frames(input_data[:min_len], self.window_size)
        
        # 2. PROCESS TARGET -> (F, N, 21, 6)
        # Original GT (22, 6): Index 0=Pos, 1=Rot, 2-21=Body
        # New Request: Index 0=HipOri(Rot), 1-20=Body. Total 21.
        
        # A. Slice GT to remove Position (Index 0)
        # Shape becomes (min_len, 21, 6)
        Y_sliced = gt_data[:min_len, 1:, :] 
        
        # B. Flatten for stride tricks -> (min_len, 126)
        # 21 joints * 6 dims = 126
        Y_flat = Y_sliced.reshape(min_len, -1) 
        
        N = self.window_size
        
        # C. Padding
        pad_gt = np.repeat(Y_flat[0:1], N-1, axis=0)
        full_gt = np.concatenate([pad_gt, Y_flat], axis=0)
        
        # D. Stride Tricks -> (F, N, 126)
        s0, s1 = full_gt.strides
        Y_windows_flat = np.lib.stride_tricks.as_strided(
            full_gt, shape=(min_len, N, 126), strides=(s0, s0, s1)
        )
        
        # E. Reshape -> (F, N, 21, 6)
        Y_windows = Y_windows_flat.reshape(min_len, N, 21, 6)
        
        # 3. SUBSAMPLING
        if self.max_windows is not None and len(X_windows) > self.max_windows:
            indices = np.linspace(0, len(X_windows)-1, self.max_windows, dtype=int)
            X_windows = X_windows[indices]
            M_masks = M_masks[indices]
            Y_windows = Y_windows[indices]

        return {
            "source": torch.from_numpy(X_windows.copy()), 
            "mask": torch.from_numpy(M_masks.copy()), 
            "target": torch.from_numpy(Y_windows.copy())
        }