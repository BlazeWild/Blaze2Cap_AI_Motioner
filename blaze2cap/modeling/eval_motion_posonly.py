# -*- coding: utf-8 -*-
# @Time    : 2/11/26
# @Project : Real-Time Motion Prediction
# @File    : eval_motion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class MotionEvaluator(nn.Module):
    """
    Evaluates Canonical Position Prediction.
    
    1. Receives Predicted Positions (20 joints).
    2. Receives Ground Truth Rotations (22 joints).
    3. Converts GT Rotations -> Canonical Positions using internal FK.
    4. Computes MPJPE in Millimeters.
    """
    def __init__(self, skeleton_parents, bone_lengths):
        """
        Args:
            skeleton_parents (list): List of parent indices.
            bone_lengths (torch.Tensor): Tensor of offsets/bone lengths.
        """
        super().__init__()
        # Register buffers to handle device placement automatically
        self.register_buffer('parents', torch.tensor(skeleton_parents, dtype=torch.long))
        self.register_buffer('offsets', bone_lengths.float())

    def _cont6d_to_mat(self, d6):
        """Converts 6D rotation to 3x3 Matrix."""
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * torch.sum(b1 * a2, dim=-1, keepdim=True))
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-1)

    def _compute_gt_positions(self, gt_rot_6d):
        """
        Converts GT Rotations (B, S, 22, 6) -> Canonical Positions (B, S, 20, 3).
        Forces Root Rotation to Identity to match Model's canonical output.
        """
        B, S, J, C = gt_rot_6d.shape
        device = gt_rot_6d.device
        
        # 1. Convert to Matrices
        rot_mats = self._cont6d_to_mat(gt_rot_6d) # (B, S, 22, 3, 3)
        
        # 2. Force Root Identity (Canonical Pose)
        # Indices 0 (Hips_Pos) and 1 (Hips_Rot) are set to Identity
        eye = torch.eye(3, device=device).view(1, 1, 3, 3).expand(B, S, 3, 3)
        rot_mats[:, :, 0] = eye
        rot_mats[:, :, 1] = eye
        
        # Root Position is fixed at 0,0,0
        root_pos = torch.zeros((B, S, 3), device=device)
        
        # 3. Run FK (Indices 0 to 21)
        global_rots = [None] * 22
        global_pos = [None] * 22
        
        # Init Roots
        global_rots[0] = rot_mats[:, :, 0]
        global_pos[0] = root_pos
        
        # Calc Hips_Rot (Idx 1)
        off1 = self.offsets[1].view(1, 1, 3, 1)
        global_rots[1] = torch.matmul(global_rots[0], rot_mats[:, :, 1])
        global_pos[1] = global_pos[0] + torch.matmul(global_rots[0], off1).squeeze(-1)
        
        # Loop Rest
        for i in range(2, 22):
            p_idx = self.parents[i].item()
            
            # Global Rot
            global_rots[i] = torch.matmul(global_rots[p_idx], rot_mats[:, :, i])
            
            # Global Pos
            off = self.offsets[i].view(1, 1, 3, 1)
            rotated_off = torch.matmul(global_rots[p_idx], off).squeeze(-1)
            global_pos[i] = global_pos[p_idx] + rotated_off
            
        # Stack to (B, S, 22, 3)
        full_pos = torch.stack(global_pos, dim=2)
        
        # 4. Slice to Body Joints (2-21) -> (B, S, 20, 3)
        return full_pos[:, :, 2:22, :]

    def compute_metrics(self, pred_pos, gt_rot):
        """
        Args:
            pred_pos: [Batch, Seq, 20, 3] (Model Output: Positions)
            gt_rot:   [Batch, Seq, 22, 6] (Ground Truth: Rotations)
        Returns:
            dict: {"MPJPE": float (mm), "MARE": 0.0}
        """
        # Ensure evaluator is on correct device
        if self.parents.device != pred_pos.device:
            self.to(pred_pos.device)

        # 1. Convert GT Rotations to Positions
        gt_pos = self._compute_gt_positions(gt_rot) # (B, S, 20, 3)
        
        # 2. Compute MPJPE (Mean Per Joint Position Error)
        # Euclidean distance: ||p - t||
        diff = pred_pos - gt_pos
        dist = torch.norm(diff, dim=-1) # (B, S, 20)
        
        # Mean over Batch, Seq, Joints
        mpjpe_meters = dist.mean().item()
        
        # Convert to Millimeters
        mpjpe_mm = mpjpe_meters * 1000.0

        return {"MPJPE": mpjpe_mm, "MARE": 0.0} # MARE is N/A for position-only models

def evaluate_motion(predictions, targets, skeleton_config):
    """
    Wrapper function for validation loop.
    """
    # Initialize Evaluator (lightweight, creates buffers)
    evaluator = MotionEvaluator(
        skeleton_parents=skeleton_config['parents'], 
        bone_lengths=skeleton_config['offsets']
    )
    
    # Move to device of input
    evaluator.to(predictions.device)
    
    metrics = evaluator.compute_metrics(predictions, targets)
    return metrics