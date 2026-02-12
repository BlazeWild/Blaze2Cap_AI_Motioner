# old_data_loader.py

import os
import json
import numpy as np
import torch
from torch.utils import data

# IMPORT FROM THE NEW FILE
from blaze2cap.modules.pose_processing_posonly import process_blazepose_frames

class PoseSequenceDataset(data.Dataset):
    def __init__(self, dataset_root, window_size, split="train", max_windows=None):
        self.dataset_root = dataset_root
        self.window_size = window_size
        self.max_windows = max_windows
        
        json_path = os.path.join(dataset_root, "dataset_map.json")
        with open(json_path, 'r') as f:
            full_data = json.load(f)
            
        # Filter samples
        self.samples = []
        candidates = [item for item in full_data if item.get(f"split_{split}", False)]
        print(f"[{split.upper()}] Found {len(candidates)} candidate samples. Checking file existence...")
        
        from tqdm import tqdm
        invalid_count = 0
        for item in tqdm(candidates, desc=f"Checking {split} files", unit="files"):
            if self._is_valid(item):
                self.samples.append(item)
            else:
                invalid_count += 1
                
        print(f"[{split.upper()}] Ready: {len(self.samples)} valid samples (filtered {invalid_count} invalid/missing).")

    def _is_valid(self, item):
        src_path = os.path.join(self.dataset_root, item["source"])
        tgt_path = os.path.join(self.dataset_root, item["target"])
        
        if not os.path.exists(src_path) or not os.path.exists(tgt_path):
            return False
        
        # Quick check: file size must be > 128 bytes (NPY header)
        if os.path.getsize(src_path) < 128 or os.path.getsize(tgt_path) < 128:
            return False
            
        try:
            # Quick check using mmap without loading data
            src_data = np.load(src_path, mmap_mode='r')
            if src_data.shape[0] == 0:
                return False
                
            tgt_data = np.load(tgt_path, mmap_mode='r')
            if tgt_data.shape[0] == 0:
                return False
                
            return True
        except Exception:
            return False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        input_path = os.path.join(self.dataset_root, item["source"])
        target_path = os.path.join(self.dataset_root, item["target"])
        
        input_data = np.nan_to_num(np.load(input_path).astype(np.float32))
        gt_data = np.nan_to_num(np.load(target_path).astype(np.float32))
        
        min_len = min(len(input_data), len(gt_data))
        input_data = input_data[:min_len]
        gt_data = gt_data[:min_len]
        
        # Call the logic from the other file
        X, M = process_blazepose_frames(input_data, self.window_size)
        
        N = self.window_size
        Y_flat = gt_data.reshape(min_len, -1)
        
        pad_gt = np.repeat(Y_flat[0:1], N-1, axis=0)
        full_gt = np.concatenate([pad_gt, Y_flat], axis=0)
        
        strides_gt = (full_gt.strides[0], full_gt.strides[0], full_gt.strides[1])
        Y = np.lib.stride_tricks.as_strided(
            full_gt, shape=(min_len, N, full_gt.shape[1]), strides=strides_gt
        )
        
        if self.max_windows is not None and len(X) > self.max_windows:
            indices = np.linspace(0, len(X)-1, self.max_windows, dtype=int)
            X = X[indices]
            M = M[indices]
            Y = Y[indices]

        return {
            "source": torch.from_numpy(X.copy()), 
            "mask": torch.from_numpy(M.copy()), 
            "target": torch.from_numpy(Y.copy())
        }