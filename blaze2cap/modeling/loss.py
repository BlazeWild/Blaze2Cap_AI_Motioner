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
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra_zen import builds
from torch import Tensor

from blaze2cap.utils.skeleton_config import get_totalcapture_skeleton

logger = logging.getLogger(__name__)

class LossBase(nn.Module):
    @abstractmethod
    def forward(self, inputs, targets, mask=None)-> Tensor:
        """Compute loss."""

class MotionCorrectionLoss(LossBase):
    """
    Hybrid Motion Loss with Differentiable MPJPE.
    
    Components:
    1. Root Velocity Loss (MSE): For Index 0 (Root Linear Velocity).
    2. Root Rotation Loss (MSE): For Index 1 (Root Orientation).
    3. Body Rotation Loss (MSE): For Indices 2-21 (Body Joint Rotations).
    4. MPJPE Loss (FK): Forward Kinematics on Body Rotations -> 3D Position Error.
    5. Smoothness Loss (Velocity MSE): Penalizes jitter.
    6. Acceleration Loss (Accel MSE): Penalizes suddern changes in velocity.
    """
    def __init__(self, 
                 skeleton_config: Optional[Dict] = None,
                 lambda_root_vel: float = 1.0,
                 lambda_root_rot: float = 1.0,
                 lambda_pose_rot: float = 1.0, # Standard priority
                 lambda_pose_pos: float = 2.0, # Boosted: Structure accuracy is critical
                 lambda_smooth: float = 10.0,  # Boosted 10x
                 lambda_accel: float = 20.0):  # Boosted 20x
        super().__init__()
        
        # Weights
        self.lambdas = {
            "root_vel": lambda_root_vel,
            "root_rot": lambda_root_rot,
            "pose_rot": lambda_pose_rot,
            "pose_pos": lambda_pose_pos,
            "smooth": lambda_smooth,
            "accel": lambda_accel
        }

        # Skeleton Config for FK
        if skeleton_config is None:
            skeleton_config = get_totalcapture_skeleton()
            
        # Register buffers for FK (persistent, not params)
        self.register_buffer('parents', torch.tensor(skeleton_config['parents'], dtype=torch.long))
        self.register_buffer('offsets', skeleton_config['offsets']) # (22, 3)

    def _cont6d_to_mat(self, d6):
        """Standard 6D -> 3x3 Matrix conversion"""
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * torch.sum(b1 * a2, dim=-1, keepdim=True))
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-1)

    def _differentiable_fk(self, body_rot_6d):
        """
        Differentiable Forward Kinematics for MPJPE Loss.
        Avoids in-place operations to prevent Autograd RuntimeError.
        
        Args:
            body_rot_6d: Tensor [B, S, 21, 6] (Joints 1-21)
                * Index 0 in this tensor corresponds to Joint 1 (Root Rot)
                * Joint 0 (Hips pos) is fixed to (0,0,0)
        
        Returns:
            positions: Tensor [B, S, 22, 3] - 3D coordinates in Root-Centered space.
        """
        B, S, J_body, C = body_rot_6d.shape
        # 1. Convert 6D -> Matrix
        rot_mats = self._cont6d_to_mat(body_rot_6d) # (B, S, 21, 3, 3)

        # 2. Build Global Rotations and Positions lists
        # We process joints in order. Since parents always appear before children in our config,
        # we can build the list sequentially.
        
        # Initialize with Root (Joint 0) - Fixed
        # Global Rot 0: Identity
        root_rot = torch.eye(3, device=body_rot_6d.device).view(1, 1, 3, 3).expand(B, S, 3, 3)
        # Global Pos 0: Zero
        root_pos = torch.zeros((B, S, 3), device=body_rot_6d.device)
        
        global_rots = [root_rot]
        global_pos = [root_pos]
        
        # Iterate Body Joints (1 to 21)
        for i in range(1, 22):
            parent_idx = self.parents[i].item()
            offset = self.offsets[i].view(1, 1, 3, 1) # (1, 1, 3, 1)
            
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

    def forward(self, inputs, targets, mask=None):
        """
        Args:
            inputs: Tuple (root_out, body_out)
                - root_out: [B, S, 2, 6]
                - body_out: [B, S, 20, 6]
            targets: Tensor [B, S, 132] or [B, S, 22, 6] (Ground Truth)
            mask: Tensor [B, S] (Optional)
        """
        # --- 1. Reshape & Concatenate ---
        root_pred, body_pred = inputs
        # Full prediction: [B, S, 22, 6]
        pred_full = torch.cat([root_pred, body_pred], dim=2)
        B, S, J, C = pred_full.shape
        
        # Reshape targets
        if targets.dim() == 3 and targets.shape[-1] == (J * C):
            targets = targets.view(B, S, J, C)
        
        # --- 2. Slice Data (The Critical Split) ---
        # Preds
        pred_root_vel = pred_full[:, :, 0, :3]       # Joint 0 (first 3 channels used)
        pred_root_rot = pred_full[:, :, 1, :]        # Joint 1 (Full 6D)
        pred_body_rot = pred_full[:, :, 2:, :]       # Joints 2-21 (Full 6D)
        
        # Targets
        target_root_vel = targets[:, :, 0, :3]
        target_root_rot = targets[:, :, 1, :]
        target_body_rot = targets[:, :, 2:, :]
        
        # --- 3. Compute Losses ---
        # Note: We use simple mean() as per user request ("The Industry Standard"), 
        # but apply mask if provided to avoid padding noise.
        
        # Mask Preparation
        if mask is not None:
             # If mask is boolean (True = Padding), invert it to get valid frames
            if mask.dtype == torch.bool:
                valid_mask = (~mask).float()
            else:
                valid_mask = mask.float()
            # Expand for broadcasting
            weights = valid_mask.view(B, S, 1) # [B, S, 1]
            valid_count = weights.sum() + 1e-8
            
            # Helper for masked mean
            def masked_mse(pred, tgt, w=weights):
                diff = (pred - tgt).pow(2)
                # Expand weights to match diff
                w_expanded = w.unsqueeze(-1) if diff.dim() > w.dim() else w
                while w_expanded.dim() < diff.dim():
                    w_expanded = w_expanded.unsqueeze(-1)
                return (diff * w_expanded).sum() / (valid_count * diff.shape[-1] * (diff.shape[-2] if diff.dim() > 3 else 1))

        else:
            # Standard Mean
            def masked_mse(pred, tgt):
                return (pred - tgt).pow(2).mean()

        # A. Root Velocity
        l_root_vel = masked_mse(pred_root_vel, target_root_vel)
        
        # B. Root Rotation
        l_root_rot = masked_mse(pred_root_rot, target_root_rot)
        
        # C. Body Rotation
        l_pose_rot = masked_mse(pred_body_rot, target_body_rot)
        
        # D. MPJPE (Forward Kinematics)
        # Combine Root Rot + Body Rot for FK input (Joints 1-21)
        # Shape: [B, S, 21, 6]
        fk_input_pred = torch.cat([pred_root_rot.unsqueeze(2), pred_body_rot], dim=2)
        fk_input_target = torch.cat([target_root_rot.unsqueeze(2), target_body_rot], dim=2)
        
        pos_pred = self._differentiable_fk(fk_input_pred)
        pos_target = self._differentiable_fk(fk_input_target)
        
        l_pose_pos = masked_mse(pos_pred, pos_target) # [B, S, 22, 3]

        # E. Smoothness & Acceleration (Full Output)
        pred_vel = pred_full[:, 1:] - pred_full[:, :-1]
        target_vel = targets[:, 1:] - targets[:, :-1]
        
        if mask is not None:
            w_vel = weights[:, 1:]
            valid_count_vel = w_vel.sum() + 1e-8
            def masked_mse_vel(pred, tgt, w=w_vel, vc=valid_count_vel):
                diff = (pred - tgt).pow(2)
                w_expanded = w
                while w_expanded.dim() < diff.dim():
                    w_expanded = w_expanded.unsqueeze(-1)
                return (diff * w_expanded).sum() / (vc * diff.shape[-1] * diff.shape[-2])
            l_smooth = masked_mse_vel(pred_vel, target_vel)
        else:
            l_smooth = (pred_vel - target_vel).pow(2).mean()

        # F. Acceleration
        pred_accel = pred_vel[:, 1:] - pred_vel[:, :-1]
        target_accel = target_vel[:, 1:] - target_vel[:, :-1]
        
        if mask is not None:
            w_accel = weights[:, 2:]
            valid_count_accel = w_accel.sum() + 1e-8
            def masked_mse_accel(pred, tgt, w=w_accel, vc=valid_count_accel):
                diff = (pred - tgt).pow(2)
                w_expanded = w
                while w_expanded.dim() < diff.dim():
                    w_expanded = w_expanded.unsqueeze(-1)
                return (diff * w_expanded).sum() / (vc * diff.shape[-1] * diff.shape[-2])
            l_accel = masked_mse_accel(pred_accel, target_accel)
        else:
            l_accel = (pred_accel - target_accel).pow(2).mean()

        # --- 5. Total Loss ---
        total_loss = (
            self.lambdas["root_vel"] * l_root_vel +
            self.lambdas["root_rot"] * l_root_rot +
            self.lambdas["pose_rot"] * l_pose_rot +
            self.lambdas["pose_pos"] * l_pose_pos +
            self.lambdas["smooth"]   * l_smooth +
            self.lambdas["accel"]    * l_accel
        )
        
        return {
            "loss": total_loss,
            "l_root_vel": l_root_vel,
            "l_root_rot": l_root_rot,
            "l_pose_rot": l_pose_rot,
            "l_pose_pos": l_pose_pos,
            "l_smooth": l_smooth,
            "l_accel": l_accel
        }

# Build configs for organizing modules with hydra
motion_loss_cfg = builds(MotionCorrectionLoss, populate_full_signature=True)