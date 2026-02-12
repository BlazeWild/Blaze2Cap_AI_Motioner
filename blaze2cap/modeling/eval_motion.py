# -*- coding: utf-8 -*-
# @Time    : 2/11/26
# @File    : eval_motion_posonly_angle.py

import torch
import torch.nn as nn
import logging

# --- CRITICAL CHANGE HERE ---
# Import from the specific 'posonly_angle' file where you defined the FK logic.
# Ensure the path matches where you saved it (likely 'modeling' or 'modules').
from blaze2cap.modeling.loss import MotionLoss

logger = logging.getLogger(__name__)

class MotionEvaluator(nn.Module):
    def __init__(self, skeleton_parents, bone_lengths):
        super().__init__()
        # We wrap the Loss module just to access its FK engine.
        # Calling it without arguments is fine; we don't need the loss weights 
        # (lambdas) just to use the internal _run_canonical_fk helper.
        self.loss_helper = MotionLoss() 

    def compute_metrics(self, pred_rot_6d, target_rot_6d):
        """
        pred_rot_6d:   (B, S, 20, 6) - Predicted Body Rotations
        target_rot_6d: (B, S, 22, 6) - Full GT (Includes Root)
        """
        # Ensure helper is on correct device
        if self.loss_helper.parents.device != pred_rot_6d.device:
            self.loss_helper.to(pred_rot_6d.device)
            
        # 1. Slice GT to Body (Remove Root Indices 0 & 1)
        # GT is (22 joints), Pred is (20 joints)
        gt_body_rot = target_rot_6d[:, :, 2:, :]
        
        # 2. MARE (Rotation Error)
        # Compare 6D rotation features directly
        mare = torch.mean(torch.abs(pred_rot_6d - gt_body_rot)).item()
        
        # 3. MPJPE (Position Error in mm)
        # Run FK on Preds (Rot -> Pos)
        pred_pos = self.loss_helper._run_canonical_fk(pred_rot_6d)
        
        # Run FK on GT (Rot -> Pos)
        gt_pos = self.loss_helper._run_canonical_fk(gt_body_rot)
        
        # Euclidean Dist
        diff = pred_pos - gt_pos
        dist = torch.norm(diff, dim=-1) # (B, S, 20)
        
        # Convert Meters -> Millimeters
        mpjpe_mm = dist.mean().item() * 1000.0
        
        return {"MPJPE": mpjpe_mm, "MARE": mare}

def evaluate_motion(predictions, targets, skeleton_config):
    # Initialize Evaluator
    evaluator = MotionEvaluator(None, None)
    evaluator.to(predictions.device)
    
    # Compute
    return evaluator.compute_metrics(predictions, targets)