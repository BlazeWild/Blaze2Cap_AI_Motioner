# -*- coding: utf-8 -*-
# @Time    : 2/5/26
# @Project : Real-Time Motion Prediction
# @File    : eval_motion.py

import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MotionEvaluator:
    """
    Evaluates motion prediction using standard metrics:
    1. MPJPE: Mean Per Joint Position Error (in millimeters/units)
    2. MARE: Mean Absolute Rotation Error (Geodesic distance in radians)
    """
    def __init__(self, skeleton_parents, bone_lengths):
        """
        Args:
            skeleton_parents (list): List of parent indices for FK.
            bone_lengths (torch.Tensor): Tensor of bone lengths [25, 3] or [25, 1].
        """
        self.parents = skeleton_parents
        self.bone_lengths = bone_lengths
        self.results = {}

    def compute_metrics(self, pred_full, gt_full):
        """
        Args:
            pred_full: [Batch, Seq, 22, 6] (Full output: Index 0=RootPos, 1=RootRot, 2..21=BodyRot)
            gt_full: [Batch, Seq, 22, 6]
        Returns:
            dict: {"MPJPE": float (in mm), "MARE": float}
        """
        # Slice to get only rotations involved in FK (Joints 1-21)
        # Index 0 is Root Linear Velocity (not a rotation)
        # Index 1 is Root Rotation (First joint in FK chain)
        # Index 2-21 are Body Rotations
        pred_rot_6d = pred_full[:, :, 1:, :] # (B, S, 21, 6)
        gt_rot_6d = gt_full[:, :, 1:, :]     # (B, S, 21, 6)

        # 1. Rotation Error (MARE) - calculated on 6D directly
        mare = torch.mean((pred_rot_6d - gt_rot_6d) ** 2).item()

        # 2. Position Error (MPJPE)
        # We must run Forward Kinematics (FK) to get 3D positions
        pred_pos = self._forward_kinematics(pred_rot_6d)
        gt_pos = self._forward_kinematics(gt_rot_6d)
        
        # Euclidean distance per joint (in meters since offsets are in meters)
        diff = pred_pos - gt_pos # [B, S, 22, 3]
        dist = torch.norm(diff, dim=-1) # [B, S, 22]
        
        # Convert to millimeters for standard MPJPE reporting
        mpjpe_mm = torch.mean(dist).item() * 1000.0

        return {"MPJPE": mpjpe_mm, "MARE": mare}

    def _cont6d_to_mat(self, d6):
        """Standard 6D -> 3x3 Matrix conversion"""
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = torch.nn.functional.normalize(a1, dim=-1)
        b2 = a2 - (b1 * torch.sum(b1 * a2, dim=-1, keepdim=True))
        b2 = torch.nn.functional.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-1)

    def _forward_kinematics(self, body_rot_6d):
        """
        Differentiable Forward Kinematics (Matches loss.py)
        Args:
            body_rot_6d: Tensor [B, S, 21, 6] (Joints 1-21)
        Returns:
            positions: Tensor [B, S, 22, 3] - 3D coordinates in Root-Centered space.
        """
        B, S, J_body, C = body_rot_6d.shape
        # 1. Convert 6D -> Matrix
        rot_mats = self._cont6d_to_mat(body_rot_6d) # (B, S, 21, 3, 3)

        # 2. Build Global Rotations and Positions lists
        # Initialize with Root (Joint 0) - Fixed
        # Global Rot 0: Identity
        root_rot = torch.eye(3, device=body_rot_6d.device).view(1, 1, 3, 3).expand(B, S, 3, 3)
        # Global Pos 0: Zero
        root_pos = torch.zeros((B, S, 3), device=body_rot_6d.device)
        
        global_rots = [root_rot]
        global_pos = [root_pos]
        
        # Iterate Body Joints (1 to 21)
        for i in range(1, 22):
            parent_idx = self.parents[i]
            offset = self.bone_lengths[i].view(1, 1, 3, 1).to(body_rot_6d.device) # (1, 1, 3, 1)
            
            # Get Parent Global Transforms
            parent_rot = global_rots[parent_idx] # (B, S, 3, 3)
            parent_pos = global_pos[parent_idx]  # (B, S, 3)
            
            # Current Local Rotation
            # Map joint index 'i' to body_rot_6d index 'i-1'
            local_rot = rot_mats[:, :, i-1] # (B, S, 3, 3)
            
            # Calculate Global Rotation: R_global = R_parent @ R_local
            curr_rot = torch.matmul(parent_rot, local_rot)
            global_rots.append(curr_rot)
            
            # Calculate Global Position: P_global = P_parent + (R_parent @ Offset)
            rotated_offset = torch.matmul(parent_rot, offset).squeeze(-1) # (B, S, 3)
            curr_pos = parent_pos + rotated_offset
            global_pos.append(curr_pos)
            
        # Stack results
        return torch.stack(global_pos, dim=2) # (B, S, 22, 3)

def evaluate_motion(predictions, targets, skeleton_config):
    """
    Wrapper function similar to 'evaluate' in captioning.
    """
    evaluator = MotionEvaluator(
        skeleton_parents=skeleton_config['parents'], 
        bone_lengths=skeleton_config['offsets']
    )
    
    metrics = evaluator.compute_metrics(predictions, targets)
    return metrics

# How to use this
# In your train.py validation loop, you can call evaluate_motion.

# Critical Note on Bone Lengths: To calculate MPJPE, you must know the bone lengths of your skeleton (e.g., length of the thigh, length of the forearm).

# If your dataset is normalized (all skeletons same height), you can hardcode these offsets in a config file.

# If your dataset handles variable heights, you generally pass the bone offsets as part of the input batch or metadata.