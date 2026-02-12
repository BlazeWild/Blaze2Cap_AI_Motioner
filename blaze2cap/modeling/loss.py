# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from blaze2cap.utils.skeleton_config import get_totalcapture_skeleton

class MotionLoss(nn.Module):
    def __init__(self, 
                 lambda_rot=1.0, 
                 lambda_pos=10.0, 
                 lambda_vel=1.0, 
                 lambda_smooth=0.5, 
                 lambda_contact=2.0, 
                 lambda_floor=2.0, 
                 lambda_tilt=1.0,
                 lambda_root=5.0): # NEW: Weight for Root Velocity
        super().__init__()
        
        self.lambdas = {
            "rot": lambda_rot, "pos": lambda_pos, "vel": lambda_vel,
            "smooth": lambda_smooth, "contact": lambda_contact, 
            "floor": lambda_floor, "tilt": lambda_tilt, "root": lambda_root
        }
        self.mse = nn.MSELoss()
        
        # FK Config
        skel = get_totalcapture_skeleton()
        self.register_buffer('parents', torch.tensor(skel['parents'], dtype=torch.long))
        self.register_buffer('offsets', skel['offsets'].float())
        self.feet_indices = [16, 19] # Body-relative indices
        self.spine_index = 0

    def _cont6d_to_mat(self, d6):
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * torch.sum(b1 * a2, dim=-1, keepdim=True))
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-1)

    def _run_canonical_fk(self, body_rot_6d):
        # ... (Same FK logic as before for Body Only) ...
        # Copied for brevity - ensure it takes (B, S, 20, 6)
        B, S, J, C = body_rot_6d.shape
        device = body_rot_6d.device
        body_mats = self._cont6d_to_mat(body_rot_6d)
        
        global_rots = [None] * 22
        global_pos = [None] * 22
        
        eye = torch.eye(3, device=device).view(1, 1, 3, 3).expand(B, S, 3, 3)
        zeros = torch.zeros((B, S, 3), device=device)
        
        global_rots[0] = eye; global_pos[0] = zeros
        off1 = self.offsets[1].view(1, 1, 3, 1)
        global_rots[1] = eye; global_pos[1] = global_pos[0] + torch.matmul(global_rots[0], off1).squeeze(-1)
        
        for i in range(2, 22):
            p = self.parents[i].item()
            local_rot = body_mats[:, :, i-2]
            global_rots[i] = torch.matmul(global_rots[p], local_rot)
            off = self.offsets[i].view(1, 1, 3, 1)
            global_pos[i] = global_pos[p] + torch.matmul(global_rots[p], off).squeeze(-1)
            
        return torch.stack(global_pos, dim=2)[:, :, 2:22, :]

    def forward(self, pred_full, target_full, mask=None):
        """
        Inputs are now (B, S, 22, 6)
        Index 0: Root Lin Vel (padded)
        Index 1: Root Ang Vel (6D)
        Index 2-21: Body Rot (6D)
        """
        
        # --- SPLIT COMPONENTS ---
        # 1. Root Linear Velocity (Only first 3 dims matter)
        pred_root_lin = pred_full[:, :, 0, :3]
        gt_root_lin   = target_full[:, :, 0, :3]
        
        # 2. Root Angular Velocity (6D)
        pred_root_ang = pred_full[:, :, 1, :]
        gt_root_ang   = target_full[:, :, 1, :]
        
        # 3. Body Rotations (20 joints)
        pred_body = pred_full[:, :, 2:, :]
        gt_body   = target_full[:, :, 2:, :]
        
        # --- LOSSES ---
        
        # A. Root Loss
        l_root_lin = self.mse(pred_root_lin, gt_root_lin)
        l_root_ang = self.mse(pred_root_ang, gt_root_ang)
        l_root = l_root_lin + l_root_ang
        
        # B. Body Rotation Loss
        l_rot = self.mse(pred_body, gt_body)
        
        # C. FK Position Loss
        pred_pos = self._run_canonical_fk(pred_body)
        gt_pos = self._run_canonical_fk(gt_body)
        l_pos = self.mse(pred_pos, gt_pos)
        
        # D. Smoothness/Physics (Same as before)
        pred_vel = pred_pos[:, 1:] - pred_pos[:, :-1]
        gt_vel = gt_pos[:, 1:] - gt_pos[:, :-1]
        l_vel = self.mse(pred_vel, gt_vel)
        
        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
        gt_acc = gt_vel[:, 1:] - gt_vel[:, :-1]
        l_smooth = self.mse(pred_acc, gt_acc)
        
        l_contact = torch.tensor(0.0, device=pred_full.device)
        if self.lambdas["contact"] > 0:
            gt_feet_vel = gt_vel[:, :, self.feet_indices, :]
            pred_feet_vel = pred_vel[:, :, self.feet_indices, :]
            is_planted = (torch.norm(gt_feet_vel, dim=-1) < 0.005).float()
            l_contact = (torch.norm(pred_feet_vel, dim=-1) * is_planted).mean()
            
        l_floor = torch.tensor(0.0, device=pred_full.device)
        if self.lambdas["floor"] > 0:
             pred_feet_y = pred_pos[:, :, self.feet_indices, 1]
             gt_feet_y = gt_pos[:, :, self.feet_indices, 1]
             l_floor = self.mse(pred_feet_y, gt_feet_y)
             
        # E. Tilt
        l_tilt = torch.tensor(0.0, device=pred_full.device)
        if self.lambdas["tilt"] > 0:
            pred_spine = self._cont6d_to_mat(pred_body[:, :, self.spine_index])
            gt_spine = self._cont6d_to_mat(gt_body[:, :, self.spine_index])
            up = torch.tensor([0.,1.,0.], device=pred_full.device).view(1,1,3,1)
            l_tilt = self.mse(torch.matmul(pred_spine, up), torch.matmul(gt_spine, up))

        # --- TOTAL ---
        total_loss = (
            self.lambdas["root"] * l_root +  # NEW
            self.lambdas["rot"] * l_rot +
            self.lambdas["pos"] * l_pos +
            self.lambdas["vel"] * l_vel +
            self.lambdas["smooth"] * l_smooth +
            self.lambdas["contact"] * l_contact +
            self.lambdas["floor"] * l_floor +
            self.lambdas["tilt"] * l_tilt
        )
        
        return total_loss, {
            "l_root": l_root.item(), # Log Root error
            "l_mpjpe": l_pos.item(),
            "l_rot": l_rot.item()
        }