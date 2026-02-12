# -*- coding: utf-8 -*-
# @Time    : 2/12/26
# @File    : eval_motion.py

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
        # This helper knows how to convert (21, 6) -> (22, 3)
        self.loss_helper = MotionLoss()
        
        # Indices for Feet in the FULL 22-joint TotalCapture skeleton
        # (The FK output restores the full skeleton, so these remain 18 & 21)
        # 18: RightFoot, 21: LeftFoot
        self.feet_indices = [18, 21] 

    def compute_metrics(self, pred_21, target_21):
        """
        Inputs are (B, S, 21, 6)
        - Index 0: Hip Rotation (Was Index 1)
        - Index 1-20: Body Rotations (Was Index 2-21)
        """
        metrics = {}
        
        # Ensure helper is on correct device
        if self.loss_helper.parents.device != pred_21.device:
            self.loss_helper.to(pred_21.device)

        # --- 1. GENERATE 3D SKELETONS (FK) ---
        # The helper maps 21 inputs -> 22 joint positions (adding World Root at 0,0,0)
        # Output shape: (B, S, 22, 3)
        pred_pos = self.loss_helper._run_canonical_fk(pred_21) 
        gt_pos = self.loss_helper._run_canonical_fk(target_21)

        # --- 2. MPJPE (Mean Per Joint Position Error) ---
        # Exclude Index 0 (World Root, which is always 0,0,0)
        # We measure Index 1 (Hips) through Index 21 (Feet)
        diff = pred_pos[:, :, 1:] - gt_pos[:, :, 1:] # (B, S, 21, 3)
        dist = torch.norm(diff, dim=-1) # (B, S, 21)
        metrics["MPJPE"] = dist.mean().item() * 1000.0 # mm

        # --- 3. HIP ORIENTATION ERROR ---
        # Measure error on Index 0 (The Hip/Pelvis Rotation)
        # Use the raw 6D output from the model
        pred_hip_rot = self.loss_helper._cont6d_to_mat(pred_21[:, :, 0, :])
        gt_hip_rot   = self.loss_helper._cont6d_to_mat(target_21[:, :, 0, :])
        
        # Geodesic Distance (Rotation Error in Degrees)
        r_diff = torch.matmul(pred_hip_rot, gt_hip_rot.transpose(-1, -2))
        trace = r_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
        cos_theta = (trace - 1) / 2
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0) 
        angle_err_rad = torch.acos(cos_theta)
        metrics["Root_Rot_Deg"] = torch.mean(torch.abs(angle_err_rad)).item() * (180.0 / np.pi)

        # --- 4. PHYSICAL REALISM ---
        
        # A. Jitter (Acceleration of 3D Joints)
        pred_vel = pred_pos[:, 1:] - pred_pos[:, :-1]
        gt_vel = gt_pos[:, 1:] - gt_pos[:, :-1]
        
        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
        gt_acc = gt_vel[:, 1:] - gt_vel[:, :-1]
        
        accel_dist = torch.norm(pred_acc - gt_acc, dim=-1)
        metrics["Accel_Err"] = accel_dist.mean().item() * 1000.0 

        # B. Foot Skating (Sliding)
        # Check velocity of feet (Indices 18, 21 in the FK skeleton)
        gt_feet_vel = gt_vel[:, :, self.feet_indices, :]
        pred_feet_vel = pred_vel[:, :, self.feet_indices, :]
        
        # Boolean Mask (1 = Planted)
        is_planted = (torch.norm(gt_feet_vel, dim=-1) < 0.005)
        
        if is_planted.sum() > 0:
            slide_magnitude = torch.norm(pred_feet_vel, dim=-1)[is_planted]
            metrics["Foot_Slide"] = slide_magnitude.mean().item() * 1000.0 
        else:
            metrics["Foot_Slide"] = 0.0
            
        # We removed Root_Pos_Err since we don't predict position anymore
        metrics["Root_Pos_Err"] = 0.0 

        return metrics

def evaluate_motion(predictions, targets):
    """
    Wrapper function to be called from training loop.
    predictions: (B, S, 21, 6)
    targets: (B, S, 21, 6)
    """
    # Create evaluator once per call (or move to train.py for efficiency)
    evaluator = MotionEvaluator()
    evaluator.to(predictions.device)
    
    with torch.no_grad():
        return evaluator.compute_metrics(predictions, targets)