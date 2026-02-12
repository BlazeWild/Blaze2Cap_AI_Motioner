# -*- coding: utf-8 -*-
# @File : loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from blaze2cap.utils.skeleton_config import get_totalcapture_skeleton

class MotionLoss(nn.Module):
    """
    State-of-the-Art Hybrid Loss for 22-Joint Motion Prediction.
    
    Structure:
    - Index 0: Root Position Delta [dx, dy, dz, 0, 0, 0]
    - Index 1: Root Rotation Delta [6D]
    - Index 2-21: Body Local Rotations [6D]
    
    Objectives:
    1. Trajectory Accuracy (Root Loss)
    2. Pose Accuracy (MPJPE via Forward Kinematics)
    3. Physical Realism (Floor, Contact, Smoothness)
    """
    def __init__(self, 
                 lambda_root=5.0,     # High priority on trajectory
                 lambda_rot=1.0,      # Local joint rotation consistency
                 lambda_pos=2.0,      # FK Position accuracy (MPJPE)
                 lambda_vel=1.0,      # Velocity matching
                 lambda_smooth=0.5,   # Acceleration/Jitter penalty
                 lambda_contact=2.0,  # Foot sliding penalty
                 lambda_floor=2.0,    # Floor penetration penalty
                 lambda_tilt=1.0):    # Spine stability
        super().__init__()
        
        self.lambdas = {
            "root": lambda_root, "rot": lambda_rot, "pos": lambda_pos,
            "vel": lambda_vel, "smooth": lambda_smooth, "contact": lambda_contact,
            "floor": lambda_floor, "tilt": lambda_tilt
        }
        self.mse = nn.MSELoss()
        
        # Load Skeleton Topology
        skel = get_totalcapture_skeleton()
        # Ensure these match the 22-joint hierarchy (0=Hips, 2=Spine, 16=RUpLeg, 19=LUpLeg)
        self.register_buffer('parents', torch.tensor(skel['parents'], dtype=torch.long))
        self.register_buffer('offsets', skel['offsets'].float())
        
        # Indices for Feet in the 22-joint array
        # RightFoot=18, LeftFoot=21
        self.feet_indices = [18, 21]
        self.spine_index = 2

    def _cont6d_to_mat(self, d6):
        """Converts 6D rotation representation to 3x3 rotation matrix."""
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * torch.sum(b1 * a2, dim=-1, keepdim=True))
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-1)

    def _run_canonical_fk(self, full_output):
        """
        Computes Forward Kinematics for MPJPE Calculation.
        
        CRITICAL: We calculate 'Canonical' Pose.
        - We FORCE the Root (Index 0) to be at (0,0,0) with Identity Rotation.
        - We apply the Body Rotations (2-21) relative to this fixed root.
        
        This isolates 'Pose Quality' from 'Trajectory Error'.
        """
        B, S, J, C = full_output.shape
        device = full_output.device
        
        # 1. Convert all 6D outputs to Matrices
        # Shape: (B, S, 22, 3, 3)
        rot_mats = self._cont6d_to_mat(full_output) 
        
        # 2. Prepare Global Buffers
        global_rots = [None] * 22
        global_pos = [None] * 22
        
        # 3. ROOT SETUP (Indices 0 & 1)
        # We FORCE Canonical Frame for MPJPE:
        # Root Position (0) = 0,0,0
        # Root Rotation (0) = Identity
        eye = torch.eye(3, device=device).view(1, 1, 3, 3).expand(B, S, 3, 3)
        zeros = torch.zeros((B, S, 3), device=device)
        
        # Hips are Index 0 in hierarchy
        global_rots[0] = eye
        global_pos[0] = zeros
        
        # Index 1 is usually "Hips Rotation" or a virtual root node in some BVH.
        # If your hierarchy says 1 is child of 0:
        p1 = self.parents[1].item() # Should be 0
        global_rots[1] = torch.matmul(global_rots[p1], rot_mats[:, :, 1])
        off1 = self.offsets[1].view(1, 1, 3, 1)
        global_pos[1] = global_pos[p1] + torch.matmul(global_rots[p1], off1).squeeze(-1)

        # 4. BODY FK LOOP (Indices 2 to 21)
        for i in range(2, 22):
            p = self.parents[i].item()
            local_rot = rot_mats[:, :, i]
            
            # Global Rotation = Parent Global * Local
            global_rots[i] = torch.matmul(global_rots[p], local_rot)
            
            # Global Pos = Parent Pos + (Parent Global Rot * Offset)
            off = self.offsets[i].view(1, 1, 3, 1)
            rot_off = torch.matmul(global_rots[p], off).squeeze(-1)
            global_pos[i] = global_pos[p] + rot_off
            
        return torch.stack(global_pos, dim=2) # (B, S, 22, 3)

    def forward(self, pred_full, target_full, mask=None):
        """
        Inputs: (B, S, 22, 6)
        """
        # --- 1. ROOT LOSS (Trajectory) ---
        # Index 0: Position Delta (only first 3 dims matter)
        pred_root_pos_delta = pred_full[:, :, 0, :3]
        gt_root_pos_delta   = target_full[:, :, 0, :3]
        
        # Index 1: Rotation Delta (all 6 dims matter)
        pred_root_rot_delta = pred_full[:, :, 1, :]
        gt_root_rot_delta   = target_full[:, :, 1, :]
        
        l_root_pos = self.mse(pred_root_pos_delta, gt_root_pos_delta)
        l_root_rot = self.mse(pred_root_rot_delta, gt_root_rot_delta)
        
        # Combined Root Loss (Heavily weighted to fix "random walking")
        l_root = l_root_pos + l_root_rot

        # --- 2. BODY ROTATION LOSS (Local Structure) ---
        # Indices 2-21
        l_rot = self.mse(pred_full[:, :, 2:], target_full[:, :, 2:])

        # --- 3. MPJPE (Pose Position Error) ---
        # Run FK to get 3D coordinates (Canonical Frame)
        pred_pos = self._run_canonical_fk(pred_full)
        gt_pos = self._run_canonical_fk(target_full)
        
        l_pos = self.mse(pred_pos, gt_pos)

        # --- 4. DYNAMICS & PHYSICS ---
        # Velocity (Smoothness over time)
        pred_vel = pred_pos[:, 1:] - pred_pos[:, :-1]
        gt_vel = gt_pos[:, 1:] - gt_pos[:, :-1]
        l_vel = self.mse(pred_vel, gt_vel)
        
        # Acceleration (Anti-Jitter)
        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
        gt_acc = gt_vel[:, 1:] - gt_vel[:, :-1]
        l_smooth = self.mse(pred_acc, gt_acc)
        
        # Floor Penetration (Anti-Clipping)
        # Check if feet go below GT feet level (simplistic floor check)
        l_floor = torch.tensor(0.0, device=pred_full.device)
        if self.lambdas["floor"] > 0:
            pred_feet_y = pred_pos[:, :, self.feet_indices, 1] # Y is Up
            # Penalize if pred is significantly lower than -0.05m (approx ground)
            # Or simplified: Match GT floor interaction
            gt_feet_y = gt_pos[:, :, self.feet_indices, 1]
            l_floor = self.mse(pred_feet_y, gt_feet_y)

        # Foot Contact (Anti-Sliding)
        l_contact = torch.tensor(0.0, device=pred_full.device)
        if self.lambdas["contact"] > 0:
            # If GT foot velocity is near zero, Pred foot velocity must be zero
            gt_feet_vel = gt_vel[:, :, self.feet_indices, :]
            pred_feet_vel = pred_vel[:, :, self.feet_indices, :]
            
            # Mask: 1 if foot is planted, 0 if moving
            is_planted = (torch.norm(gt_feet_vel, dim=-1) < 0.005).float()
            
            # Loss = Pred Velocity * Planted Mask
            l_contact = (torch.norm(pred_feet_vel, dim=-1) * is_planted).mean()

        # Spine Tilt (Upright Stability)
        l_tilt = torch.tensor(0.0, device=pred_full.device)
        if self.lambdas["tilt"] > 0:
            pred_spine_rot = self._cont6d_to_mat(pred_full[:, :, self.spine_index])
            gt_spine_rot = self._cont6d_to_mat(target_full[:, :, self.spine_index])
            up_vec = torch.tensor([0., 1., 0.], device=pred_full.device).view(1, 1, 3, 1)
            l_tilt = self.mse(torch.matmul(pred_spine_rot, up_vec), torch.matmul(gt_spine_rot, up_vec))

        # --- TOTAL WEIGHTED LOSS ---
        total_loss = (
            self.lambdas["root"] * l_root +
            self.lambdas["rot"] * l_rot +
            self.lambdas["pos"] * l_pos +
            self.lambdas["vel"] * l_vel +
            self.lambdas["smooth"] * l_smooth +
            self.lambdas["contact"] * l_contact +
            self.lambdas["floor"] * l_floor +
            self.lambdas["tilt"] * l_tilt
        )
        
        # Return Dict for Logging
        return total_loss, {
            "l_root": l_root.item(),
            "l_mpjpe": l_pos.item(),
            "l_rot": l_rot.item(),
            "l_contact": l_contact.item()
        }