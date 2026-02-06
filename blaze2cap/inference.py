import torch
import numpy as np

class MotionInference:
    """
    Real-time Motion Inference Engine with Zero-Latency Accumulator.
    Implements a rolling buffer strategy to handle history and accumulates
    global position/rotation increments from local deltas.
    """
    def __init__(self, model, device='cpu', window_size=64):
        self.model = model
        self.device = device
        self.window_size = window_size
        self.model.eval()
        
        # Initialize State
        self.buffer = [] # Stores feature vectors
        
        # Global State Accumulators
        self.world_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.world_rot = np.eye(3, dtype=np.float32)

    def _rotation_6d_to_matrix(self, d6):
        """
        Convert 6D rotation representation to 3x3 rotation matrix using Gram-Schmidt.
        Args:
            d6: (6,) numpy array
        Returns:
            R: (3, 3) rotation matrix
        """
        a1 = d6[0:3]
        a2 = d6[3:6]
        
        # Gram-Schmidt orthonormalization
        b1 = a1 / (np.linalg.norm(a1) + 1e-6)
        b2 = a2 - np.dot(b1, a2) * b1
        b2 = b2 / (np.linalg.norm(b2) + 1e-6)
        b3 = np.cross(b1, b2)
        
        R = np.stack([b1, b2, b3], axis=1)
        return R

    def process_frame(self, current_feat):
        """
        Process a single frame of features and update the global state.
        
        Args:
            current_feat: (D,) numpy array of input features (e.g. 450 dim)
            
        Returns:
            world_pos: (3,) Updated global position
            world_rot: (3,3) Updated global rotation matrix
            body_pose: (20, 6) Predicted body local rotations (6D)
        """
        # 1. Buffer Management
        # Cold Start: If buffer is empty, fill it with copies of the first frame.
        if len(self.buffer) == 0:
            self.buffer = [current_feat] * self.window_size
        else:
            # Update: Slide window
            self.buffer.pop(0)
            self.buffer.append(current_feat)
            
        # 2. Prediction
        # Prepare Input: (1, N, D)
        X = np.stack(self.buffer, axis=0) # (N, D)
        X = X[np.newaxis, ...] # (1, N, D)
        
        # To Tensor
        X_tensor = torch.from_numpy(X).float().to(self.device)
        
        # Run Model
        with torch.no_grad():
            # Pass all-valid mask (all False) just in case model expects it
            # Shape (B, S) -> (1, N)
            mask = torch.zeros((1, self.window_size), dtype=torch.bool, device=self.device)
            
            root_out, body_out = self.model(X_tensor, key_padding_mask=mask)
            
            # Get prediction for the LAST timestep
            root_pred = root_out[0, -1].cpu().numpy() # (2, 6)
            body_pred = body_out[0, -1].cpu().numpy() # (20, 6)

        # 3. The Accumulator
        # Extract deltas from prediction
        # Joint 0: Root Position Delta (indices 0-2 of first joint)
        root_pos_delta = root_pred[0, 0:3]
        
        # Joint 1: Root Rotation Delta (indices 0-5 of second joint)
        root_rot_delta_6d = root_pred[1, :]
        delta_rot_mat = self._rotation_6d_to_matrix(root_rot_delta_6d)
        
        # Update Rotation
        # R_new = R_old @ R_delta
        # Re-orthogonalize is implicit because we construct delta_rot_mat from 6D using Gram-Schmidt
        # and matrix multiplication of orthogonal matrices is orthogonal.
        # However, numerical errors might accumulate, but 6D->Mat ensures validity each step.
        self.world_rot = self.world_rot @ delta_rot_mat
        
        # Update Position
        # Global Delta = R_current @ Local_Delta
        # Position += Global Delta
        global_delta = self.world_rot @ root_pos_delta
        self.world_pos += global_delta
        
        return self.world_pos, self.world_rot, body_pred
