# -*- coding: utf-8 -*-
# @Time : 2026/2/11
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
    Hybrid Motion Loss for 3D Human Motion Prediction.
    
    Robust to Coordinate Systems (Y-up or Y-down) via Relative Tilt Loss.
    
    Components:
    1. Canonical MPJPE: Checks pose structure independent of global heading.
    2. Relative Tilt Loss: Enforces correct vertical alignment (gravity) by matching GT.
    3. Floor Glue: Penalizes vertical drift (Anti-Rocket).
    4. Foot Contact: Enforces zero-velocity for planted feet (Anti-Slide).
    """
    def __init__(self, 
                 skeleton_config: Optional[Dict] = None,
                 parents: Optional[torch.Tensor] = None,
                 offsets: Optional[torch.Tensor] = None,
                 lambda_root_vel: float = 1.0,   # Base Trajectory
                 lambda_root_rot: float = 2.0,   # Orientation (Boosted)
                 lambda_pose_rot: float = 1.0,   # Local Joint Angles
                 lambda_pose_pos: float = 4.0,   # Canonical MPJPE (Critical for structure)
                 lambda_smooth: float = 10.0,    # Velocity Smoothness
                 lambda_accel: float = 20.0,     # Acceleration Smoothness
                 lambda_contact: float = 0.5,    # Foot Contact (Anti-Slide)
                 lambda_floor: float = 5.0,      # Floor Glue (Anti-Rocket)
                 lambda_tilt: float = 5.0):      # Gravity Consistency
        super().__init__()
        
        self.lambdas = {
            "root_vel": lambda_root_vel,
            "root_rot": lambda_root_rot,
            "pose_rot": lambda_pose_rot,
            "pose_pos": lambda_pose_pos,
            "smooth": lambda_smooth,
            "accel": lambda_accel,
            "contact": lambda_contact,
            "floor": lambda_floor,
            "tilt": lambda_tilt
        }

        # Skeleton Setup
        if parents is not None and offsets is not None:
             self.register_buffer('parents', parents.long())
             self.register_buffer('offsets', offsets.float())
        else:
            if skeleton_config is None: skeleton_config = get_totalcapture_skeleton()
            self.register_buffer('parents', torch.tensor(skeleton_config['parents'], dtype=torch.long))
            self.register_buffer('offsets', skeleton_config['offsets'])

        # Foot Indices (RightFoot=18, LeftFoot=21 in standard 22-joint set)
        self.foot_indices = [18, 21]

    def _cont6d_to_mat(self, d6):
        """Standard 6D -> 3x3 Matrix conversion"""
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1, eps=1e-6)
        b2 = a2 - (b1 * torch.sum(b1 * a2, dim=-1, keepdim=True))
        b2 = F.normalize(b2, dim=-1, eps=1e-6)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-1)

    def _differentiable_fk_canonical(self, body_rot_6d):
        """
        Canonical Forward Kinematics.
        Forces Root Rotation to Identity and Root Position to (0,0,0).
        This isolates the POSE structure from the GLOBAL orientation.
        """
        B, S, J_body, C = body_rot_6d.shape
        rot_mats = self._cont6d_to_mat(body_rot_6d)

        # Force Identity Root (The "Canonical" View)
        root_rot = torch.eye(3, device=body_rot_6d.device).view(1, 1, 3, 3).expand(B, S, 3, 3)
        root_pos = torch.zeros((B, S, 3), device=body_rot_6d.device)
        
        global_rots = [root_rot, root_rot] # Indices 0 & 1 fixed
        global_pos = [root_pos, root_pos]
        
        # FK Loop (Joints 2 to 21)
        for i in range(2, 22):
            parent_idx = self.parents[i].item()
            offset = self.offsets[i].view(1, 1, 3, 1)
            
            parent_rot = global_rots[parent_idx]
            parent_pos = global_pos[parent_idx]
            
            # Local Rot from Input (Index offset by -2)
            local_rot = rot_mats[:, :, i-2] 
            
            curr_rot = torch.matmul(parent_rot, local_rot)
            global_rots.append(curr_rot)
            
            rotated_offset = torch.matmul(parent_rot, offset).squeeze(-1)
            curr_pos = parent_pos + rotated_offset
            global_pos.append(curr_pos)
            
        return torch.stack(global_pos, dim=2)

    def forward(self, inputs, targets, mask=None):
        # 1. Prepare Data
        if isinstance(inputs, tuple):
            pred_full = torch.cat(inputs, dim=2)
        else:
            pred_full = inputs
            
        B, S, J, C = pred_full.shape
        if targets.dim() == 3: targets = targets.view(B, S, J, C)
        
        # 2. Slice Components
        pred_root_vel = pred_full[:, :, 0, :3]
        pred_root_rot = pred_full[:, :, 1, :]
        pred_body_rot = pred_full[:, :, 2:, :]
        
        target_root_vel = targets[:, :, 0, :3]
        target_root_rot = targets[:, :, 1, :]
        target_body_rot = targets[:, :, 2:, :]

        # 3. Helper: Masked MSE
        if mask is not None:
            weights = (~mask).float().view(B, S, 1) if mask.dtype == torch.bool else mask.float().view(B, S, 1)
            valid_count = weights.sum() + 1e-8
            def mse(p, t): return ((p - t).pow(2) * weights).sum() / (valid_count * p.shape[-1])
        else:
            def mse(p, t): return (p - t).pow(2).mean()

        # --- LOSS COMPONENTS ---

        # A. Trajectory & Floor Glue
        l_root_vel = mse(pred_root_vel, target_root_vel)
        
        # Floor Glue: Index 1 is the Vertical Axis (Y) in your plot.
        # This keeps the character from drifting vertically.
        l_floor = mse(pred_root_vel[..., 1:2], target_root_vel[..., 1:2])

        # B. Orientation & Tilt (Relative Gravity)
        l_root_rot = mse(pred_root_rot, target_root_rot)
        
        # Tilt Loss: "Make Predicted Vertical Axis match GT Vertical Axis"
        # Since both are Y-down, this works perfectly without coordinate conversion.
        pred_rot_mat = self._cont6d_to_mat(pred_root_rot)
        target_rot_mat = self._cont6d_to_mat(target_root_rot)
        
        # Local "Up" is [0, 1, 0] relative to hips (Vertical Axis)
        local_up = torch.tensor([0.0, 1.0, 0.0], device=inputs.device).view(1, 1, 3, 1).expand(B, S, -1, -1)
        
        pred_world_up = torch.matmul(pred_rot_mat, local_up)
        target_world_up = torch.matmul(target_rot_mat, local_up)
        l_tilt = mse(pred_world_up, target_world_up)

        # C. Body Pose (Canonical MPJPE)
        l_pose_rot = mse(pred_body_rot, target_body_rot)
        
        pos_pred = self._differentiable_fk_canonical(pred_body_rot)
        pos_target = self._differentiable_fk_canonical(target_body_rot)
        l_pose_pos = mse(pos_pred, pos_target)

        # D. Foot Contact (Anti-Slide)
        # Get Feet Positions from Canonical FK
        feet_pos_pred = pos_pred[:, :, self.foot_indices, :]
        feet_pos_target = pos_target[:, :, self.foot_indices, :]
        
        # Calculate approximate global velocity
        # Global Vel = Root Vel + Local Foot Vel
        feet_vel_loc = torch.zeros_like(feet_pos_pred)
        feet_vel_loc[:, 1:] = feet_pos_pred[:, 1:] - feet_pos_pred[:, :-1]
        
        root_vel_exp = pred_root_vel.unsqueeze(2).expand(-1, -1, 2, -1)
        global_feet_vel = root_vel_exp + feet_vel_loc
        
        # GT Contact Mask logic
        gt_root_vel_exp = target_root_vel.unsqueeze(2).expand(-1, -1, 2, -1)
        gt_feet_vel_loc = torch.zeros_like(feet_pos_target)
        gt_feet_vel_loc[:, 1:] = feet_pos_target[:, 1:] - feet_pos_target[:, :-1]
        gt_global_vel = gt_root_vel_exp + gt_feet_vel_loc
        
        # If GT foot speed < 2mm/frame, it is planted.
        contact_mask = (gt_global_vel.norm(dim=-1) < 0.002).float()
        
        # Penalize movement if planted
        l_contact = (global_feet_vel.norm(dim=-1) * contact_mask).mean()

        # E. Smoothness
        pred_vel = pred_full[:, 1:] - pred_full[:, :-1]
        target_vel = targets[:, 1:] - targets[:, :-1]
        l_smooth = mse(pred_vel, target_vel)

        pred_accel = pred_vel[:, 1:] - pred_vel[:, :-1]
        target_accel = target_vel[:, 1:] - target_vel[:, :-1]
        l_accel = mse(pred_accel, target_accel)

        # --- TOTAL LOSS ---
        total_loss = (
            self.lambdas["root_vel"] * l_root_vel +
            self.lambdas["floor"]    * l_floor +
            self.lambdas["root_rot"] * l_root_rot +
            self.lambdas["tilt"]     * l_tilt +
            self.lambdas["pose_rot"] * l_pose_rot +
            self.lambdas["pose_pos"] * l_pose_pos +
            self.lambdas["contact"]  * l_contact +
            self.lambdas["smooth"]   * l_smooth +
            self.lambdas["accel"]    * l_accel
        )
        
        return {
            "loss": total_loss,
            "l_mpjpe": l_pose_pos,
            "l_tilt": l_tilt,
            "l_contact": l_contact,
            "l_floor": l_floor
        }

motion_loss_cfg = builds(MotionCorrectionLoss, populate_full_signature=True)