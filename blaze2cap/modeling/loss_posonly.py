# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from blaze2cap.utils.skeleton_config import get_totalcapture_skeleton

class MotionLoss(nn.Module):
    """
    Geometric Loss with Internal FK.
    
    1. Receives Model Output (Positions).
    2. Receives Ground Truth (Rotations).
    3. Converts GT Rotations -> GT Positions using FK.
    4. Computes Loss (MSE).
    """
    def __init__(self, 
                 lambda_pos: float = 100.0,
                 lambda_vel: float = 10.0,
                 lambda_smooth: float = 1.0,
                 lambda_contact: float = 0.5):
        super().__init__()
        
        self.lambdas = {
            "pos": lambda_pos,
            "vel": lambda_vel,
            "smooth": lambda_smooth,
            "contact": lambda_contact
        }
        
        self.mse = nn.MSELoss()
        
        # --- FK SETUP ---
        skel_config = get_totalcapture_skeleton()
        # Register buffers so they move to GPU automatically
        self.register_buffer('parents', torch.tensor(skel_config['parents'], dtype=torch.long))
        self.register_buffer('offsets', skel_config['offsets'].float())
        
        # Indices for Feet in the *Reduced* 20-joint set
        # RightFoot=16, LeftFoot=19 (after slicing 2-22)
        self.foot_indices = [16, 19]

    def _cont6d_to_mat(self, d6):
        """Converts 6D rotation to 3x3 Matrix."""
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * torch.sum(b1 * a2, dim=-1, keepdim=True))
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-1)

    def _compute_target_positions(self, target_rot_6d):
        """
        Converts GT Rotations (B, S, 22, 6) -> Canonical Positions (B, S, 20, 3)
        """
        B, S, J, C = target_rot_6d.shape
        device = target_rot_6d.device
        
        # 1. Convert to Matrices
        rot_mats = self._cont6d_to_mat(target_rot_6d) # (B, S, 22, 3, 3)
        
        # 2. Force Root Identity (Canonical Pose)
        # We overwrite the rotation matrices for Hips (idx 1) and Hips_Pos (idx 0)
        # to be Identity. This forces the FK to generate positions relative to a 
        # fixed root at (0,0,0).
        eye = torch.eye(3, device=device).view(1, 1, 3, 3).expand(B, S, 3, 3)
        rot_mats[:, :, 0] = eye
        rot_mats[:, :, 1] = eye
        
        # Root Position is 0,0,0
        root_pos = torch.zeros((B, S, 3), device=device)
        
        # 3. Run FK (Indices 0 to 21)
        # We need to store all 22 joint positions to calculate the chain
        global_rots = [None] * 22
        global_pos = [None] * 22
        
        # Init Roots
        global_rots[0] = rot_mats[:, :, 0]
        global_pos[0] = root_pos
        
        # Calc Hips_Rot (Idx 1)
        # P_1 = P_0 + R_0 @ offset_1
        off1 = self.offsets[1].view(1, 1, 3, 1)
        global_rots[1] = torch.matmul(global_rots[0], rot_mats[:, :, 1])
        global_pos[1] = global_pos[0] + torch.matmul(global_rots[0], off1).squeeze(-1)
        
        # Loop Rest
        for i in range(2, 22):
            p_idx = self.parents[i].item()
            
            # Global Rot = Parent_Global_Rot * Local_Rot
            global_rots[i] = torch.matmul(global_rots[p_idx], rot_mats[:, :, i])
            
            # Global Pos = Parent_Global_Pos + Parent_Global_Rot * Offset
            off = self.offsets[i].view(1, 1, 3, 1)
            rotated_off = torch.matmul(global_rots[p_idx], off).squeeze(-1)
            global_pos[i] = global_pos[p_idx] + rotated_off
            
        # Stack to (B, S, 22, 3)
        full_pos = torch.stack(global_pos, dim=2)
        
        # 4. Slice to Body Joints (2-21) -> (B, S, 20, 3)
        return full_pos[:, :, 2:22, :]

    def forward(self, pred_pos, target_rot_6d, mask=None):
        """
        pred_pos:      (B, S, 20, 3) - Predicted Positions
        target_rot_6d: (B, S, 22, 6) - GT Rotations
        """
        
        # 1. Convert GT Rotations to Positions
        target_pos = self._compute_target_positions(target_rot_6d)
        
        # 2. Position Loss
        l_pos = self.mse(pred_pos, target_pos)
        
        # 3. Velocity
        pred_vel = pred_pos[:, 1:] - pred_pos[:, :-1]
        target_vel = target_pos[:, 1:] - target_pos[:, :-1]
        l_vel = self.mse(pred_vel, target_vel)
        
        # 4. Smoothness (Acceleration)
        pred_accel = pred_vel[:, 1:] - pred_vel[:, :-1]
        target_accel = target_vel[:, 1:] - target_vel[:, :-1]
        l_smooth = self.mse(pred_accel, target_accel)
        
        # 5. Contact (Optional)
        if self.lambdas["contact"] > 0:
            gt_feet_vel = target_vel[:, :, self.foot_indices, :]
            pred_feet_vel = pred_vel[:, :, self.foot_indices, :]
            
            # Stationary threshold (e.g., 5mm)
            stationary_mask = (torch.norm(gt_feet_vel, dim=-1) < 0.005).float()
            l_contact = (torch.norm(pred_feet_vel, dim=-1) * stationary_mask).mean()
        else:
            l_contact = torch.tensor(0.0, device=pred_pos.device)
            
        # Total
        total_loss = (
            self.lambdas["pos"] * l_pos +
            self.lambdas["vel"] * l_vel +
            self.lambdas["smooth"] * l_smooth +
            self.lambdas["contact"] * l_contact
        )
        
        return total_loss, {
            "l_mpjpe": l_pos.item(),
            "l_vel": l_vel.item(),
            "l_smooth": l_smooth.item(),
            "l_contact": l_contact.item()
        }