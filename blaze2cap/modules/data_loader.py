import os
import json
import numpy as np
import torch
from torch.utils import data

# --- STANDARD IMPORT ---
from blaze2cap.modules.pose_processing import process_blazepose_frames

class PoseSequenceDataset(data.Dataset):
    def __init__(self, dataset_root, window_size, split="train", max_windows=None):
        self.dataset_root = dataset_root
        self.window_size = window_size
        self.max_windows = max_windows
        
        json_path = os.path.join(dataset_root, "dataset_map.json")
        with open(json_path, 'r') as f:
            full_data = json.load(f)
            
        self.samples = [item for item in full_data if item.get(f"split_{split}", False)]
        print(f"[{split.upper()}] Loaded {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        input_path = os.path.join(self.dataset_root, item["source"])
        target_path = os.path.join(self.dataset_root, item["target"])
        
        # Load
        input_data = np.nan_to_num(np.load(input_path).astype(np.float32))
        gt_data = np.nan_to_num(np.load(target_path).astype(np.float32))
        
        min_len = min(len(input_data), len(gt_data))
        
        # 1. Process Input -> Returns (F, N, 27, 19)
        # Uses the code from pose_processing.py above
        X_windows, M_masks = process_blazepose_frames(input_data[:min_len], self.window_size)
        
        # 2. Process Target -> Returns (F, N, 22, 6)
        N = self.window_size
        
        # FIX: Flatten to 2D for stride tricks
        Y_flat = gt_data[:min_len].reshape(min_len, -1) # (F, 132)
        
        # Pad
        pad_gt = np.repeat(Y_flat[0:1], N-1, axis=0)
        full_gt = np.concatenate([pad_gt, Y_flat], axis=0)
        
        # Stride on 2D
        s0, s1 = full_gt.strides
        Y_windows_flat = np.lib.stride_tricks.as_strided(
            full_gt, shape=(min_len, N, 132), strides=(s0, s0, s1)
        )
        
        # Reshape back to (F, N, 22, 6)
        Y_windows = Y_windows_flat.reshape(min_len, N, 22, 6)
        
        # 3. Subsample
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