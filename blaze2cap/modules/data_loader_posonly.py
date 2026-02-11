import os
import json
import numpy as np
import torch
from torch.utils import data
from blaze2cap.modules.pose_processing import process_blazepose_frames
from blaze2cap.utils.skeleton_config import get_totalcapture_skeleton # Assuming this exists

class PoseSequenceDataset(data.Dataset):
    def __init__(self, dataset_root, window_size, split="train", max_windows=None):
        self.dataset_root = dataset_root
        self.window_size = window_size
        self.max_windows = max_windows
        
        # Load Skeleton Config for FK
        skel_config = get_totalcapture_skeleton()
        self.parents = torch.tensor(skel_config['parents'], dtype=torch.long)
        self.offsets = skel_config['offsets'] # Tensor (22, 3)
        
        # Setup Map
        json_path = os.path.join(dataset_root, "dataset_map.json")
        with open(json_path, 'r') as f:
            full_data = json.load(f)
            
        self.samples = [item for item in full_data if item.get(f"split_{split}", False)]
        print(f"[{split.upper()}] Loaded {len(self.samples)} samples.")

    def _compute_target_positions(self, gt_data_np):
        """
        Converts GT Rotations (Frames, 22, 6) -> Canonical Positions (Frames, 20, 3)
        1. Fix Root to Identity.
        2. Run FK.
        3. Extract Body Joints (2-21).
        """
        # gt_data_np is (F, 22, 6) usually.
        # Channels: 6D rotation.
        
        device = torch.device('cpu')
        offsets = self.offsets.to(device)
        parents = self.parents.to(device)
        
        gt_tensor = torch.from_numpy(gt_data_np) # (F, 22, 6)
        F_frames = gt_tensor.shape[0]
        
        # 1. OVERRIDE ROOT ROTATION (Index 1) -> IDENTITY
        # Identity 6D is [1,0,0, 0,1,0]
        identity_6d = torch.tensor([1,0,0, 0,1,0], dtype=torch.float32)
        gt_tensor[:, 1, :] = identity_6d # Hips_rot
        
        # Root Pos (Index 0) is typically ignored for local pose, set to 0
        root_pos = torch.zeros((F_frames, 3), dtype=torch.float32)

        # 2. CONVERT 6D -> ROTATION MATRICES (3x3)
        # Inline helper for 6D->Mat
        d6 = gt_tensor
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = torch.nn.functional.normalize(a1, dim=-1)
        b2 = a2 - (b1 * torch.sum(b1 * a2, dim=-1, keepdim=True))
        b2 = torch.nn.functional.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        rot_mats = torch.stack((b1, b2, b3), dim=-1) # (F, 22, 3, 3)
        
        # 3. RUN FORWARD KINEMATICS
        # Global storage
        # indices 0,1 are roots in TotalCapture logic. 
        # Start FK from 0.
        
        global_rots = [None] * 22
        global_pos = [None] * 22
        
        # Hips_pos (0)
        global_rots[0] = rot_mats[:, 0] # Usually Identity or global
        global_pos[0] = root_pos
        
        # Hips_rot (1) - Parent is 0
        # R_1 = R_0 * r_1
        global_rots[1] = torch.matmul(global_rots[0], rot_mats[:, 1])
        # P_1 = P_0 + R_0 * offset_1
        off1 = offsets[1].view(1, 3, 1)
        global_pos[1] = global_pos[0] + torch.matmul(global_rots[0], off1).squeeze(-1)
        
        # Loop 2 -> 21
        for i in range(2, 22):
            p_idx = parents[i].item()
            
            # Rotation
            global_rots[i] = torch.matmul(global_rots[p_idx], rot_mats[:, i])
            
            # Position
            off = offsets[i].view(1, 3, 1)
            rotated_off = torch.matmul(global_rots[p_idx], off).squeeze(-1)
            global_pos[i] = global_pos[p_idx] + rotated_off
            
        # Stack all
        full_pos = torch.stack(global_pos, dim=1) # (F, 22, 3)
        
        # 4. EXTRACT BODY JOINTS (2-21)
        # Output shape: (F, 20, 3)
        body_pos = full_pos[:, 2:22, :]
        
        # Optional: Re-center relative to Hips_rot (Index 1) just to be safe?
        # Since we fixed root at (0,0,0), P_1 is at (0,0,0).
        # So body_pos is already local.
        
        return body_pos.numpy()

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # Load Raw
        input_data = np.nan_to_num(np.load(os.path.join(self.dataset_root, item["source"])).astype(np.float32))
        gt_raw = np.nan_to_num(np.load(os.path.join(self.dataset_root, item["target"])).astype(np.float32))
        
        min_len = min(len(input_data), len(gt_raw))
        input_data = input_data[:min_len]
        gt_raw = gt_raw[:min_len]
        
        # A. Process Input (BlazePose -> 19 Joints, 8 Channels, Canonical)
        X, _ = process_blazepose_frames(input_data, self.window_size)
        
        # B. Process Target (GT Rotations -> 20 Joints, 3D Positions, Canonical)
        Y_positions = self._compute_target_positions(gt_raw) # (F, 20, 3)
        
        # Window Target
        N = self.window_size
        Y_flat = Y_positions.reshape(min_len, -1)
        
        pad_gt = np.repeat(Y_flat[0:1], N-1, axis=0)
        full_gt = np.concatenate([pad_gt, Y_flat], axis=0)
        
        strides_gt = (full_gt.strides[0], full_gt.strides[0], full_gt.strides[1])
        Y = np.lib.stride_tricks.as_strided(
            full_gt, shape=(min_len, N, full_gt.shape[1]), strides=strides_gt
        )
        
        # Subsampling
        if self.max_windows is not None and len(X) > self.max_windows:
            indices = np.linspace(0, len(X)-1, self.max_windows, dtype=int)
            X = X[indices]
            Y = Y[indices]

        # Returns:
        # source: (B, 64, 19, 8) -> flattened to (B, 64, 152) in model
        # target: (B, 64, 20, 3) -> flattened to (B, 64, 60) in model
        return {
            "source": torch.from_numpy(X.copy()), 
            "target": torch.from_numpy(Y.copy())
        }