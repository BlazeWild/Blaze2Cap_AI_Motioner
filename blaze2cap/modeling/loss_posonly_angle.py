# -*- coding: utf-8 -*-
# @Time : 2026/2/11
# @Author : BlazeWild
# @File : loss_posonly_angle.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from blaze2cap.utils.skeleton_config import get_totalcapture_skeleton

class MotionLoss(nn.Module):
    """
    Hybrid Rotation-Position Loss.
    
    1. Receives Model Output: 20 Body Joint Rotations (6D).
    2. Receives Ground Truth: 22 Joint Rotations (6D).
    3. Computes:
       - Rotation Loss (MSE on 6D features)
       - Position Loss (Run FK on Pred AND Target -> MSE on Positions)
       - Velocity/Smoothness (Computed on the derived Positions)
    """
    def __init__(self, 
                 lambda_rot: float = 1.0,    # Rotation consistency
                 lambda_pos: float = 10.0,   # FK Position accuracy (MPJPE)
                 lambda_vel: float = 1.0,    # Velocity smoothness
                 lambda_smooth: float = 0.1, # Acceleration smoothness
                 lambda_contact: float = 0.5):
        super().__init__()
        
        self.lambdas = {
            "rot": lambda_rot,
            "pos": lambda_pos,
            "vel": lambda_vel,
            "smooth": lambda_smooth,
            "contact": lambda_contact
        }
        
        self.mse = nn.MSELoss()
        
        # --- FK CONFIG ---
        skel = get_totalcapture_skeleton()
        self.register_buffer('parents', torch.tensor(skel['parents'], dtype=torch.long))
        self.register_buffer('offsets', skel['offsets'].float())
        
        # Indices for Feet in the 20-joint body array (Right=16, Left=19)
        self.foot_indices = [16, 19]

    def _cont6d_to_mat(self, d6):
        """Converts 6D rotation to 3x3 Matrix."""
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * torch.sum(b1 * a2, dim=-1, keepdim=True))
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-1)

    def _run_canonical_fk(self, body_rot_6d):
        """
        Converts Body Rotations (B, S, 20, 6) -> Canonical Positions (B, S, 20, 3).
        
        Logic:
        1. Prepend Identity Rotations for Root (Hips_Pos, Hips_Rot).
        2. Reconstruct full 22-joint skeleton.
        3. Return last 20 joints (Body).
        """
        B, S, J, C = body_rot_6d.shape
        device = body_rot_6d.device
        
        # 1. Convert Body 6D -> 3x3
        body_mats = self._cont6d_to_mat(body_rot_6d) # (B, S, 20, 3, 3)
        
        # 2. Prepare Global Buffers (22 joints)
        global_rots = [None] * 22
        global_pos = [None] * 22
        
        # 3. Setup Canonical Roots (Indices 0 & 1) -> Identity
        # This aligns everything to (0,0,0) facing Forward
        eye = torch.eye(3, device=device).view(1, 1, 3, 3).expand(B, S, 3, 3)
        zero_pos = torch.zeros((B, S, 3), device=device)
        
        # Hips_Pos (0)
        global_rots[0] = eye
        global_pos[0] = zero_pos
        
        # Hips_Rot (1) - Fixed to Identity for Canonical
        off1 = self.offsets[1].view(1, 1, 3, 1)
        global_rots[1] = eye # Identity * Identity
        global_pos[1] = global_pos[0] + torch.matmul(global_rots[0], off1).squeeze(-1)
        
        # 4. FK Loop for Body (Indices 2 to 21)
        # body_mats index 0 corresponds to Skeleton index 2
        for i in range(2, 22):
            p_idx = self.parents[i].item()
            local_rot = body_mats[:, :, i-2] # Shift index
            
            # Rot
            global_rots[i] = torch.matmul(global_rots[p_idx], local_rot)
            
            # Pos
            off = self.offsets[i].view(1, 1, 3, 1)
            rotated_off = torch.matmul(global_rots[p_idx], off).squeeze(-1)
            global_pos[i] = global_pos[p_idx] + rotated_off
            
        # Stack 22 joints -> Slice 2:22
        full_pos = torch.stack(global_pos, dim=2)
        return full_pos[:, :, 2:22, :]

    def forward(self, pred_rot_6d, target_rot_6d, mask=None):
        """
        pred_rot_6d:   (B, S, 20, 6) - Model Output
        target_rot_6d: (B, S, 22, 6) - GT from DataLoader
        """
        
        # 1. Slice Target to match Body (20 joints)
        gt_body_rot = target_rot_6d[:, :, 2:, :] # (B, S, 20, 6)
        
        # --- A. ROTATION LOSS (Direct 6D) ---
        l_rot = self.mse(pred_rot_6d, gt_body_rot)
        
        # --- B. POSITION LOSS (Via FK) ---
        # Convert both Pred and GT to positions to check physical validity
        pred_pos = self._run_canonical_fk(pred_rot_6d)
        target_pos = self._run_canonical_fk(gt_body_rot)
        
        l_pos = self.mse(pred_pos, target_pos)
        
        # --- C. DYNAMICS LOSS (On Positions) ---
        # Computing velocity on Positions is usually better than on Rotations
        pred_vel = pred_pos[:, 1:] - pred_pos[:, :-1]
        target_vel = target_pos[:, 1:] - target_pos[:, :-1]
        l_vel = self.mse(pred_vel, target_vel)
        
        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
        target_acc = target_vel[:, 1:] - target_vel[:, :-1]
        l_smooth = self.mse(pred_acc, target_acc)
        
        # --- D. CONTACT LOSS ---
        l_contact = torch.tensor(0.0, device=pred_rot_6d.device)
        if self.lambdas["contact"] > 0:
            gt_feet_vel = target_vel[:, :, self.foot_indices]
            stat_mask = (torch.norm(gt_feet_vel, dim=-1) < 0.005).float()
            pred_feet_vel = pred_vel[:, :, self.foot_indices]
            l_contact = (torch.norm(pred_feet_vel, dim=-1) * stat_mask).mean()
            
        # Total
        total_loss = (
            self.lambdas["rot"] * l_rot +
            self.lambdas["pos"] * l_pos +
            self.lambdas["vel"] * l_vel +
            self.lambdas["smooth"] * l_smooth +
            self.lambdas["contact"] * l_contact
        )
        
        return total_loss, {
            "l_rot": l_rot.item(), 
            "l_mpjpe": l_pos.item()
        }