# -*- coding: utf-8 -*-
# @Time    : 2/12/26
# @File    : eval_motion_posonly_angle.py

import torch
import torch.nn as nn
import logging
import numpy as np

# Import the Loss class to reuse its FK and Rotation conversion logic
from blaze2cap.modeling.loss import MotionLoss

logger = logging.getLogger(__name__)

class MotionEvaluator(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize Loss helper to access FK and skeleton data
        self.loss_helper = MotionLoss()
        
        # Indices for Feet in the 22-joint TotalCapture skeleton
        # RightFoot=18, LeftFoot=21
        self.feet_indices = [18, 21] 

    def compute_metrics(self, pred_full, target_full):
        """
        Inputs are (B, S, 22, 6)
        - Index 0: Root Pos Delta
        - Index 1: Root Rot Delta
        - Index 2-21: Body Rotations
        """
        metrics = {}
        
        # Ensure helper is on correct device
        if self.loss_helper.parents.device != pred_full.device:
            self.loss_helper.to(pred_full.device)

        # --- 1. MPJPE (Mean Per Joint Position Error) ---
        # This measures Body Pose quality, independent of Root position.
        # The FK helper automatically zeros out the root to Canonical Frame.
        pred_pos = self.loss_helper._run_canonical_fk(pred_full) # (B, S, 22, 3)
        gt_pos = self.loss_helper._run_canonical_fk(target_full)
        
        # Exclude Root (Index 0) and Virtual Root (Index 1) from average if desired,
        # but usually MPJPE averages all body joints. 
        # We look at indices 2-21 (Actual Body).
        diff = pred_pos[:, :, 2:] - gt_pos[:, :, 2:] # (B, S, 20, 3)
        dist = torch.norm(diff, dim=-1) # (B, S, 20)
        metrics["MPJPE"] = dist.mean().item() * 1000.0 # mm

        # --- 2. ROOT ACCURACY (Trajectory) ---
        # Root Linear Velocity Error (Index 0)
        pred_root_lin = pred_full[:, :, 0, :3]
        gt_root_lin   = target_full[:, :, 0, :3]
        root_pos_err = torch.norm(pred_root_lin - gt_root_lin, dim=-1)
        metrics["Root_Pos_Err"] = root_pos_err.mean().item() * 1000.0 # mm per frame

        # Root Angular Velocity Error (Index 1) - MAE on 6D features
        pred_root_ang = pred_full[:, :, 1, :]
        gt_root_ang   = target_full[:, :, 1, :]
        metrics["Root_Rot_Err"] = torch.mean(torch.abs(pred_root_ang - gt_root_ang)).item()

        # --- 3. PHYSICAL REALISM ---
        
        # A. Jitter (Acceleration)
        # 2nd derivative of position. High values = shaking.
        # We calculate "Jitter Error": |Pred_Acc - GT_Acc| isn't useful. 
        # We want the raw magnitude of Pred Acceleration to see if it's shaking.
        # But usually, we compare to GT to see if we are 'smoother' or 'noisier' than GT.
        # Standard metric: Mean Acceleration Difference
        pred_vel = pred_pos[:, 1:] - pred_pos[:, :-1]
        gt_vel = gt_pos[:, 1:] - gt_pos[:, :-1]
        
        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
        gt_acc = gt_vel[:, 1:] - gt_vel[:, :-1]
        
        accel_dist = torch.norm(pred_acc - gt_acc, dim=-1)
        metrics["Accel_Err"] = accel_dist.mean().item() * 1000.0 # mm/s^2 error

        # B. Foot Skating (Sliding)
        # Definition: How much does the foot move when it should be planted?
        # GT Planted Condition: GT Foot Velocity < 5mm/frame
        gt_feet_vel = gt_vel[:, :, self.feet_indices, :]
        pred_feet_vel = pred_vel[:, :, self.feet_indices, :]
        
        # Boolean Mask (1 = Planted)
        is_planted = (torch.norm(gt_feet_vel, dim=-1) < 0.005)
        
        # Calculate Slide: Magnitude of Pred Velocity * Mask
        # If no frames are planted, return 0 to avoid NaN
        if is_planted.sum() > 0:
            slide_magnitude = torch.norm(pred_feet_vel, dim=-1)[is_planted]
            metrics["Foot_Slide"] = slide_magnitude.mean().item() * 1000.0 # mm of slip
        else:
            metrics["Foot_Slide"] = 0.0

        return metrics

def evaluate_motion(predictions, targets):
    """
    Wrapper function to be called from training loop.
    predictions: (B, S, 22, 6)
    targets: (B, S, 22, 6)
    """
    # Create evaluator on the fly (lightweight)
    evaluator = MotionEvaluator()
    evaluator.to(predictions.device)
    
    with torch.no_grad():
        return evaluator.compute_metrics(predictions, targets)