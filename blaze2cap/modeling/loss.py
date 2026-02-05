# -*- coding: utf-8 -*-
# @Time : 2026/2/5
# @Author : BlazeWild
# @Project: Real-Time Motion Prediction
# @File : loss.py

__all__=[
    "LossBase",
    "MotionCorrectionLoss",
    "motion_loss_cfg"
]

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra_zen import builds
from torch import Tensor, nn

logger = logging.getLogger(__name__)

class LossBase(nn.Module):
    @abstractmethod
    def forward(self, inputs, targets, mask=None)-> Tensor:
        """Compute loss."""

class MotionCorrectionLoss(LossBase):
    """
    Implementation of the Split-Head Motion Loss defined in Figure 2.
    
    Components:
    1. Rotation Loss (L_rot): MSE between predicted and ground truth 6D vectors.
    2. Smoothness Loss (L_smooth): MSE of velocity differences to penalize jitter.
    
    Formula:
        L_total = L_rot + lambda_smooth * L_smooth
    """
    def __init__(self, lambda_smooth=1.0, lambda_rot=1.0):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_rot = lambda_rot

    def forward(self, inputs, targets, mask=None):
        """
        Args:
            inputs: Tuple (root_out, body_out) from MotionTransformer
                - root_out: [B, S, 2, 6]
                - body_out: [B, S, 20, 6]
            targets: Ground Truth Tensor [B, S, 132] or [B, S, 22, 6]
            mask: Boolean Padding Mask [B, S] (True = Padding/Ignore)
            
        Returns:
            dict with 'loss' (total), 'l_rot', and 'l_smooth'
        """
        # ---1. Data Preparation --
        root_pred, body_pred = inputs

        # Concatenate split heads to form full pose: [B, S, 22,6]
        # 2 root joints + 20 body joints = 22 joints
        pred_full = torch.cat([root_pred, body_pred], dim=2)
        B,S,J,C = pred_full.shape
        
        # Reshape targets to [B, S, 22, 6] if they are flattened
        if targets.dim()==3 and targets.shape[-1]==(J*C):
            targets = targets.view(B,S,J,C)

        # ---2 Mask Handling ---
        # We need a float mask : 1.0 == Valid, 0.0 == Padding
        if mask is not None:
            # If mask is boolean (True = Padding), invert it to get valid frames
            if mask.dtype == torch.bool:
                valid_mask = (~mask).float()
            else:
                valid_mask = mask.float()
            
            # Expand mask for dimensions: [B, S, 1, 1] to broadcast over joints/coords
            weights = valid_mask.view(B, S, 1, 1)
            valid_count = weights.sum() + 1e-8 # Avoid div by zero
        else:
            weights = torch.ones_like(pred_full[..., 0:1])
            valid_count = weights.numel()

        # --- 3. Rotation Loss (L_rot) ---
        # Standard MSE between prediction and GT
        # Formula: || R_hat - R ||^2
        diff_rot = pred_full - targets
        loss_rot_element = diff_rot ** 2
        
        # Apply mask and mean reduction
        loss_rot = (loss_rot_element * weights).sum() / valid_count

        # --- 4. Smoothness Loss (L_smooth) ---
        # Formula: || (R_hat_t - R_hat_t-1) - (R_t - R_t-1) ||^2
        # This forces the *velocity* of the prediction to match the *velocity* of the GT.
        
        # Calculate velocities (Delta between frames)
        # Slice [1:] (current) and [:-1] (previous)
        pred_vel = pred_full[:, 1:] - pred_full[:, :-1]
        target_vel = targets[:, 1:] - targets[:, :-1]
        
        # Calculate MSE of the velocities
        diff_vel = pred_vel - target_vel
        loss_smooth_element = diff_vel ** 2
        
        # Adjust weights for the shortened sequence length (S-1)
        weights_vel = weights[:, 1:] 
        valid_count_vel = weights_vel.sum() + 1e-8
        
        loss_smooth = (loss_smooth_element * weights_vel).sum() / valid_count_vel

        # --- 5. Total Loss ---
        total_loss = (self.lambda_rot * loss_rot) + (self.lambda_smooth * loss_smooth)
        
        return {
            "loss": total_loss,
            "l_rot": loss_rot,
            "l_smooth": loss_smooth
        }

# Build configs for organizing modules with hydra
motion_loss_cfg = builds(MotionCorrectionLoss, populate_full_signature=True)