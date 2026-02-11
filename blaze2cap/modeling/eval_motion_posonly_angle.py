# -*- coding: utf-8 -*-
# @Time    : 2/11/26
# @File    : eval_motion_posonly_angle.py

import torch
import torch.nn as nn
import logging
# Reuse the robust FK logic from the Loss class
from blaze2cap.modules.loss import MotionLoss

logger = logging.getLogger(__name__)

class MotionEvaluator(nn.Module):
    def __init__(self, skeleton_parents, bone_lengths):
        super().__init__()
        # We wrap the Loss module just to access its FK engine
        # Pass dummy lambdas since we don't use the forward()
        self.loss_helper = MotionLoss() 
        # Note: MotionLoss loads its own skeleton config internally using get_totalcapture_skeleton()
        # If you need custom skeletons, you'd modify MotionLoss or pass params there.
        # For now, assuming standard TotalCapture setup.

    def compute_metrics(self, pred_rot_6d, target_rot_6d):
        """
        pred_rot_6d:   (B, S, 20, 6)
        target_rot_6d: (B, S, 22, 6)
        """
        # Ensure helper is on correct device
        if self.loss_helper.parents.device != pred_rot_6d.device:
            self.loss_helper.to(pred_rot_6d.device)
            
        # 1. Slice GT to Body
        gt_body_rot = target_rot_6d[:, :, 2:, :]
        
        # 2. MARE (Mean Absolute Rotation Error)
        # L1 distance between 6D vectors
        mare = torch.mean(torch.abs(pred_rot_6d - gt_body_rot)).item()
        
        # 3. MPJPE (Position Error in mm)
        # Run FK
        pred_pos = self.loss_helper._run_canonical_fk(pred_rot_6d)
        gt_pos = self.loss_helper._run_canonical_fk(gt_body_rot)
        
        # Euclidean Dist
        diff = pred_pos - gt_pos
        dist = torch.norm(diff, dim=-1) # (B, S, 20)
        
        # Convert to mm
        mpjpe_mm = dist.mean().item() * 1000.0
        
        return {"MPJPE": mpjpe_mm, "MARE": mare}

def evaluate_motion(predictions, targets, skeleton_config):
    # skeleton_config is unused here because MotionLoss loads it internally,
    # but kept for API compatibility
    evaluator = MotionEvaluator(None, None)
    evaluator.to(predictions.device)
    return evaluator.compute_metrics(predictions, targets)