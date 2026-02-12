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
    - Index 0: Root Position Delta [dx, dy, dz, 0, 0, 0] (Local Space)
    - Index 1: Root Rotation Delta [6D] (Local Space)
    - Index 2-21: Body Local Rotations [6D] (Parent-Relative)
    """
    def __init__(self, 
                 lambda_root=5.0,     # Trajectory (Critical)
                 lambda_rot=1.0,      # Pose Structure (Rotations)
                 lambda_pos=2.0,      # Pose Shape (MPJPE)
                 lambda_vel=1.0,      # Smoothness
                 lambda_smooth=0.5,   # Anti-Jitter
                 lambda_contact=2.0,  # Anti-Skate
                 lambda_floor=2.0,    # Anti-Clipping
                 lambda_tilt=1.0):    # Upright Constraint
        super().__init__()
        
        self.lambdas = {
            "root": lambda_root, "rot": lambda_rot, "pos": lambda_pos,
            "vel": lambda_vel, "smooth": lambda_smooth, "contact": lambda_contact,
            "floor": lambda_floor, "tilt": lambda_tilt
        }
        self.mse = nn.MSELoss()
        
        # Load Topology
        skel = get_totalcapture_skeleton()
        self.register_buffer('parents', torch.tensor(skel['parents'], dtype=torch.long))
        self.register_buffer('offsets', skel['offsets'].float())
        
        # Feet Indices (TotalCapture 22-joint rig)
        # 18: RightFoot, 21: LeftFoot
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
        Runs Forward Kinematics to get 3D Joint Positions.
        
        STRATEGY: "Canonical Pose"
        We deliberately FORCE the Root to be at (0,0,0) with Identity Rotation.
        
        WHY?
        1. MPJPE must measure BODY SHAPE errors, not Room Position errors.
        2. If we included Root Transforms here, a small rotation error at the hip
           would look like a massive position error at the hand, confusing the gradient.
        3. We learn Root Trajectory separately via 'l_root'.
        """
        B, S, J, C = full_output.shape
        device = full_output.device
        
        # 1. Convert all 6D outputs to Rotation Matrices
        rot_mats = self._cont6d_to_mat(full_output) 
        
        # 2. Prepare Buffers
        global_rots = [None] * 22
        global_pos = [None] * 22
        
        # 3. ROOT SETUP (Force Canonical)
        # Root (0) and Virtual Root (1) are locked to Identity/Zero.
        eye = torch.eye(3, device=device).view(1, 1, 3, 3).expand(B, S, 3, 3)
        zeros = torch.zeros((B, S, 3), device=device)
        
        global_rots[0] = eye
        global_pos[0] = zeros
        
        # Handle Index 1 (often same as root or small offset)
        p1 = self.parents[1].item()
        # Even for Index 1, we ignore the predicted rotation delta here to keep the "Pose" stationary
        global_rots[1] = eye 
        off1 = self.offsets[1].view(1, 1, 3, 1)
        global_pos[1] = global_pos[p1] + torch.matmul(global_rots[p1], off1).squeeze(-1)

        # 4. BODY FK LOOP (Indices 2 to 21)
        # Only these joints reflect the actual body posture
        for i in range(2, 22):
            p = self.parents[i].item()
            local_rot = rot_mats[:, :, i]
            
            # Standard FK Math
            global_rots[i] = torch.matmul(global_rots[p], local_rot)
            
            off = self.offsets[i].view(1, 1, 3, 1)
            rot_off = torch.matmul(global_rots[p], off).squeeze(-1)
            global_pos[i] = global_pos[p] + rot_off
            
        return torch.stack(global_pos, dim=2) # (B, S, 22, 3)

    def forward(self, pred_full, target_full, mask=None):
        """
        Calculates total loss.
        pred_full: (B, S, 22, 6)
        """
        # --- 1. TRAJECTORY LOSS (Direct Regression) ---
        # The model must learn the exact Local Delta values.
        # Index 0: Position Delta [dx, dy, dz]
        l_root_pos = self.mse(pred_full[:, :, 0, :3], target_full[:, :, 0, :3])
        # Index 1: Rotation Delta [6D]
        l_root_rot = self.mse(pred_full[:, :, 1, :], target_full[:, :, 1, :])
        
        l_root = l_root_pos + l_root_rot

        # --- 2. ROTATION CONSISTENCY (Body) ---
        # Direct supervision on 6D features for indices 2-21
        l_rot = self.mse(pred_full[:, :, 2:], target_full[:, :, 2:])

        # --- 3. MPJPE (Pose Shape) ---
        # Convert to 3D points (Canonical Frame) and compare
        pred_pos = self._run_canonical_fk(pred_full)
        gt_pos = self._run_canonical_fk(target_full)
        
        l_pos = self.mse(pred_pos, gt_pos)

        # --- 4. DYNAMICS (Velocity/Accel) ---
        # Ensures smooth animation
        pred_vel = pred_pos[:, 1:] - pred_pos[:, :-1]
        gt_vel = gt_pos[:, 1:] - gt_pos[:, :-1]
        l_vel = self.mse(pred_vel, gt_vel)
        
        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
        gt_acc = gt_vel[:, 1:] - gt_vel[:, :-1]
        l_smooth = self.mse(pred_acc, gt_acc)

        # --- 5. PHYSICS CONSTRAINTS ---
        
        # A. Floor Loss
        # In Canonical Frame (Root=0), feet should match GT height relative to root.
        l_floor = torch.tensor(0.0, device=pred_full.device)
        if self.lambdas["floor"] > 0:
            pred_feet_y = pred_pos[:, :, self.feet_indices, 1] 
            gt_feet_y = gt_pos[:, :, self.feet_indices, 1]
            l_floor = self.mse(pred_feet_y, gt_feet_y)

        # B. Foot Contact (Anti-Skate)
        # If GT says foot is planted (vel=0), Pred foot vel must be 0.
        l_contact = torch.tensor(0.0, device=pred_full.device)
        if self.lambdas["contact"] > 0:
            gt_feet_vel = gt_vel[:, :, self.feet_indices, :]
            pred_feet_vel = pred_vel[:, :, self.feet_indices, :]
            
            # Strictly plant feet if GT speed is < 5mm/frame
            is_planted = (torch.norm(gt_feet_vel, dim=-1) < 0.005).float()
            l_contact = (torch.norm(pred_feet_vel, dim=-1) * is_planted).mean()

        # C. Spine Tilt
        # Keep spine upright if GT is upright
        l_tilt = torch.tensor(0.0, device=pred_full.device)
        if self.lambdas["tilt"] > 0:
            pred_spine = self._cont6d_to_mat(pred_full[:, :, self.spine_index])
            gt_spine = self._cont6d_to_mat(target_full[:, :, self.spine_index])
            up = torch.tensor([0., 1., 0.], device=pred_full.device).view(1, 1, 3, 1)
            l_tilt = self.mse(torch.matmul(pred_spine, up), torch.matmul(gt_spine, up))

        # --- TOTAL ---
        loss = (
            self.lambdas["root"] * l_root +
            self.lambdas["rot"] * l_rot +
            self.lambdas["pos"] * l_pos +
            self.lambdas["vel"] * l_vel +
            self.lambdas["smooth"] * l_smooth +
            self.lambdas["contact"] * l_contact +
            self.lambdas["floor"] * l_floor +
            self.lambdas["tilt"] * l_tilt
        )
        
        return loss, {
            "l_root": l_root.item(),
            "l_mpjpe": l_pos.item(),
            "l_rot": l_rot.item(),
            "l_contact": l_contact.item()
        }