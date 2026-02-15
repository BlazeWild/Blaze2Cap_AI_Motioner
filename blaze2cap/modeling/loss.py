# -*- coding: utf-8 -*-
# @File : loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from blaze2cap.utils.skeleton_config import get_totalcapture_skeleton

class MotionLoss(nn.Module):
    """
    Updated Loss for 21-Joint Motion Prediction (Hip Rotation + Body).
    
    Structure (Input/Target):
    - Shape: (B, S, 21, 6)
    - Index 0: Hip/Pelvis Rotation [6D] (Corresponds to Skeleton Index 1)
    - Index 1-20: Body Rotations [6D] (Corresponds to Skeleton Indices 2-21)
    
    Note: Explicit Root Trajectory (Position Delta) is removed from prediction.
    """
    def __init__(self, 
                 lambda_root=5.0,     # Hip Orientation Weight
                 lambda_rot=1.0,      # Body Pose Structure
                 lambda_pos=5.0,      # FK Position (MPJPE) - Increased for precision
                 lambda_vel=1.0,      # Smoothness
                 # Temporarily Disabled Physics Weights
                 lambda_smooth=0.0,   
                 lambda_contact=0.0,  
                 lambda_floor=0.0,    
                 lambda_tilt=0.0):    
        super().__init__()
        
        self.lambdas = {
            "root": lambda_root, "rot": lambda_rot, "pos": lambda_pos,
            "vel": lambda_vel, "smooth": lambda_smooth, "contact": lambda_contact,
            "floor": lambda_floor, "tilt": lambda_tilt
        }
        self.mse = nn.MSELoss()
        
        # Load Topology (Standard 22-joint TotalCapture)
        # 0: WorldRoot, 1: Hips, 2+: Body
        skel = get_totalcapture_skeleton()
        self.register_buffer('parents', torch.tensor(skel['parents'], dtype=torch.long))
        self.register_buffer('offsets', skel['offsets'].float())
        
        # Feet Indices (Right=18, Left=21)
        self.feet_indices = [18, 21]

    def _cont6d_to_mat(self, d6):
        """Converts 6D rotation representation to 3x3 rotation matrix."""
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * torch.sum(b1 * a2, dim=-1, keepdim=True))
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-1)

    def _run_canonical_fk(self, pred_21_joints):
        """
        Runs Forward Kinematics mapping 21 predicted joints to the 22-joint skeleton.
        """
        B, S, J, C = pred_21_joints.shape # (B, S, 21, 6)
        device = pred_21_joints.device
        
        # 1. Convert Predictions to Matrices
        # rot_mats shape: (B, S, 21, 3, 3)
        rot_mats = self._cont6d_to_mat(pred_21_joints) 
        
        # 2. Prepare Buffers for FULL 22-Joint Skeleton
        global_rots = [None] * 22
        global_pos = [None] * 22
        
        # 3. JOINT 0: WORLD ROOT (Fixed Anchor)
        # We enforce Canonical Frame: Root is at (0,0,0) with Identity Rotation
        eye = torch.eye(3, device=device).view(1, 1, 3, 3).expand(B, S, 3, 3)
        zeros = torch.zeros((B, S, 3), device=device)
        
        global_rots[0] = eye
        global_pos[0] = zeros
        
        # 4. JOINT 1: HIPS (From Prediction Index 0)
        # CRITICAL FIX: We allow this to rotate!
        p1 = self.parents[1].item() # Should be 0
        
        # Map Prediction Index 0 -> Skeleton Index 1
        local_rot_hip = rot_mats[:, :, 0] 
        
        global_rots[1] = torch.matmul(global_rots[p1], local_rot_hip)
        off1 = self.offsets[1].view(1, 1, 3, 1)
        global_pos[1] = global_pos[p1] + torch.matmul(global_rots[p1], off1).squeeze(-1)

        # 5. BODY LOOP (Skeleton Indices 2 to 21)
        # Map Prediction Indices 1..20 -> Skeleton Indices 2..21
        for i in range(2, 22):
            pred_idx = i - 1 # Shift index back by 1
            
            p = self.parents[i].item()
            local_rot = rot_mats[:, :, pred_idx]
            
            # Standard FK
            global_rots[i] = torch.matmul(global_rots[p], local_rot)
            
            off = self.offsets[i].view(1, 1, 3, 1)
            rot_off = torch.matmul(global_rots[p], off).squeeze(-1)
            global_pos[i] = global_pos[p] + rot_off
            
        return torch.stack(global_pos, dim=2) # (B, S, 22, 3)

    def _accumulate_hip_rot(self, hip_deltas_6d):
        """
        Accumulates 6D deltas into a sequence of absolute rotation matrices.
        R_t = R_{t-1} @ Delta_t
        Assumes R_0 is Identity.
        
        hip_deltas_6d: (B, S, 6)
        Returns: (B, S, 3, 3)
        """
        B, S, _ = hip_deltas_6d.shape
        device = hip_deltas_6d.device
        
        # Convert all deltas to matrices first: (B, S, 3, 3)
        delta_mats = self._cont6d_to_mat(hip_deltas_6d)
        
        # Initialize accumulator: (B, 3, 3) identity
        curr_rot = torch.eye(3, device=device).unsqueeze(0).expand(B, 3, 3)
        
        abs_rots = []
        for t in range(S):
            # Delta at time t
            d = delta_mats[:, t] # (B, 3, 3)
            
            # Update: R_new = R_old @ Delta
            curr_rot = torch.matmul(curr_rot, d)
            
            abs_rots.append(curr_rot)
            
        return torch.stack(abs_rots, dim=1) # (B, S, 3, 3)

    def forward(self, pred_full, target_full, mask=None):
        """
        pred_full: (B, S, 21, 6)
        target_full: (B, S, 21, 6)
        """
        # --- 1. HIP ORIENTATION LOSS (Index 0) ---
        # A. Frame-to-Frame Delta Loss
        l_root_delta = self.mse(pred_full[:, :, 0, :], target_full[:, :, 0, :])
        
        # B. Trajectory Accumulation Loss (NEW)
        # Reconstruct absolute orientation sequence from deltas
        pred_hip_abs = self._accumulate_hip_rot(pred_full[:, :, 0, :])
        gt_hip_abs = self._accumulate_hip_rot(target_full[:, :, 0, :])
        
        l_root_accum = self.mse(pred_hip_abs, gt_hip_abs)
        
        # Combine: Weighted sum for Root Loss
        # We give accumulated loss significant weight to kill drift
        l_root = l_root_delta + 1.0 * l_root_accum

        # --- 2. BODY ROTATION LOSS (Indices 1-20) ---
        l_rot = self.mse(pred_full[:, :, 1:], target_full[:, :, 1:])

        # --- 3. MPJPE (Pose Shape) ---
        # FK handles the mapping from 21->22 and centers the root
        pred_pos = self._run_canonical_fk(pred_full)
        gt_pos = self._run_canonical_fk(target_full)
        
        l_pos = self.mse(pred_pos, gt_pos)

        # --- 4. DYNAMICS (Velocity) ---
        # Smoothness of the generated 3D points
        pred_vel = pred_pos[:, 1:] - pred_pos[:, :-1]
        gt_vel = gt_pos[:, 1:] - gt_pos[:, :-1]
        l_vel = self.mse(pred_vel, gt_vel)
        
        # --- DISABLED / COMMENTED OUT LOSSES ---
        l_smooth = torch.tensor(0.0, device=pred_full.device)
        l_contact = torch.tensor(0.0, device=pred_full.device)
        
        # --- TOTAL ---
        loss = (
            self.lambdas["root"] * l_root +
            self.lambdas["rot"] * l_rot +
            self.lambdas["pos"] * l_pos +
            self.lambdas["vel"] * l_vel 
            + self.lambdas["contact"] * l_contact
        )
        
        return loss, {
            "l_root": l_root.item(),
            "l_root_acc": l_root_accum.item(), # Log this!
            "l_mpjpe": l_pos.item(),
            "l_rot": l_rot.item(),
            "l_contact": l_contact.item()
        }