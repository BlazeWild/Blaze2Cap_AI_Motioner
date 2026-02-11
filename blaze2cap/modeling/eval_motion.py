# -*- coding: utf-8 -*-
# @Time    : 2/5/26
# @Project : Real-Time Motion Prediction
# @File    : eval_motion.py

import torch
import numpy as np
import logging
from blaze2cap.modeling.loss import MotionCorrectionLoss

logger = logging.getLogger(__name__)

class MotionEvaluator:
    """
    Evaluates motion prediction using standard metrics from the Loss class.
    Uses the exact same FK logic as training to ensure consistency.
    """
    def __init__(self, skeleton_parents, bone_lengths):
        """
        Args:
            skeleton_parents (list): List of parent indices.
            bone_lengths (torch.Tensor): Tensor of offsets/bone lengths.
        """
        # We instantiate the Loss class to use its FK methods.
        # We don't need weights (lambdas) for evaluation, so we use defaults.
        self.loss_module = MotionCorrectionLoss(
            parents=torch.tensor(skeleton_parents, dtype=torch.long),
            offsets=bone_lengths
        )

    def compute_metrics(self, pred_full, gt_full):
        """
        Args:
            pred_full: [Batch, Seq, 22, 6] (Full output)
            gt_full: [Batch, Seq, 22, 6]
        Returns:
            dict: {"MPJPE": float (mm), "MARE": float}
        """
        # Ensure inputs are on the same device as the loss module buffers
        device = pred_full.device
        if self.loss_module.parents.device != device:
            self.loss_module.to(device)

        # 1. Slice Data (Matches Loss Logic)
        # Index 0: Root Vel (Ignored for Pose/Rot metrics)
        # Index 1: Root Rot Delta
        # Index 2-21: Body Local Rots
        pred_body = pred_full[:, :, 2:, :]
        gt_body = gt_full[:, :, 2:, :]
        
        # 2. Rotation Error (MARE) - Mean Absolute Rotation Error
        # We compare the Full 6D output (Indices 1-21)
        pred_rot_all = pred_full[:, :, 1:, :] 
        gt_rot_all = gt_full[:, :, 1:, :]
        mare = torch.mean(torch.abs(pred_rot_all - gt_rot_all)).item()

        # 3. Position Error (MPJPE)
        # Use the Canonical FK from loss.py (Forces Root to Identity)
        # This measures pure structure error, independent of global rotation.
        pred_pos = self.loss_module._differentiable_fk_canonical(pred_body)
        gt_pos = self.loss_module._differentiable_fk_canonical(gt_body)
        
        # Euclidean distance per joint (in meters)
        diff = pred_pos - gt_pos # [B, S, 22, 3]
        dist = torch.norm(diff, dim=-1) # [B, S, 22]
        
        # Convert to millimeters
        mpjpe_mm = dist.mean().item() * 1000.0

        return {"MPJPE": mpjpe_mm, "MARE": mare}

def evaluate_motion(predictions, targets, skeleton_config):
    """
    Wrapper function for validation loop.
    """
    evaluator = MotionEvaluator(
        skeleton_parents=skeleton_config['parents'], 
        bone_lengths=skeleton_config['offsets']
    )
    
    metrics = evaluator.compute_metrics(predictions, targets)
    return metrics