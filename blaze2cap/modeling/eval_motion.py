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

    def compute_metrics(self, pred_rot, gt_rot):
        """
        Args:
            pred_rot: [Batch, Seq, 22, 6] (6D format)
            gt_rot: [Batch, Seq, 22, 6] (6D format)
        Returns:
            dict: {"MPJPE": float (in mm), "MARE": float}
        """
        # 1. Rotation Error (MARE) - calculated on 6D directly or Matrices
        # We approximate it here using MSE of the 6D vector for speed, 
        # or proper Geodesic distance if you convert to matrix.
        mare = torch.mean((pred_rot - gt_rot) ** 2).item()

        # 2. Position Error (MPJPE)
        # We must run Forward Kinematics (FK) to get 3D positions
        pred_pos = self._forward_kinematics(pred_rot)
        gt_pos = self._forward_kinematics(gt_rot)
        
        # Euclidean distance per joint (in meters since offsets are in meters)
        diff = pred_pos - gt_pos # [B, S, 22, 3]
        dist = torch.norm(diff, dim=-1) # [B, S, 22]
        
        # Convert to millimeters for standard MPJPE reporting
        mpjpe_mm = torch.mean(dist).item() * 1000.0

        return {"MPJPE": mpjpe_mm, "MARE": mare}

    def _forward_kinematics(self, rot_6d):
        """
        Simple FK implementation.
        Converts Local 6D Rotations -> Global 3D Positions.
        """
        # Note: This is a simplified FK for evaluation. 
        # In production, you might use a library like pytorch3d or your own optimized FK.
        
        B, S, J, _ = rot_6d.shape
        
        # 1. Convert 6D -> 3D Rotation Matrix [B, S, J, 3, 3]
        rot_mat = self._cont6d_to_mat(rot_6d)
        
        # 2. Iterate hierarchy
        # Global transforms
        glob_pos = torch.zeros(B, S, J, 3, device=rot_6d.device)
        glob_rot = torch.zeros(B, S, J, 3, 3, device=rot_6d.device)
        
        # Root (Index 0)
        glob_pos[:, :, 0] = 0 # Root usually at origin or specific offset
        glob_rot[:, :, 0] = rot_mat[:, :, 0]

        for i in range(1, J):
            parent = self.parents[i]
            # Local position of joint i relative to parent (Bone vector)
            # Usually: Rot_parent * Bone_length
            bone = self.bone_lengths[i].to(rot_6d.device)
            
            # P_global = P_parent + R_parent * P_local
            # We assume bone vectors are aligned with offsets.
            # (Simplified logic - requires your specific bone offsets)
            offset = torch.matmul(glob_rot[:, :, parent], bone.unsqueeze(-1)).squeeze(-1)
            
            glob_pos[:, :, i] = glob_pos[:, :, parent] + offset
            glob_rot[:, :, i] = torch.matmul(glob_rot[:, :, parent], rot_mat[:, :, i])
            
        return glob_pos

    def _cont6d_to_mat(self, d6):
        """
        Converts 6D continuous representation to 3x3 rotation matrix.
        """
        a1 = d6[..., :3]
        a2 = d6[..., 3:]
        
        # Gram-Schmidt normalization
        b1 = torch.nn.functional.normalize(a1, dim=-1)
        b2 = a2 - (b1 * torch.sum(b1 * a2, dim=-1, keepdim=True))
        b2 = torch.nn.functional.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        
        return torch.stack((b1, b2, b3), dim=-1)

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