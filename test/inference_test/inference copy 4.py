import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

# --- ENVIRONMENT SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
blaze2cap_root = os.path.dirname(os.path.dirname(current_dir))
if blaze2cap_root not in sys.path:
    sys.path.insert(0, blaze2cap_root)

# IMPORT STANDARD MODULES
from blaze2cap.modules.models import MotionTransformer
from blaze2cap.modules.pose_processing import process_blazepose_frames
from blaze2cap.utils.skeleton_config import get_totalcapture_skeleton

# --- CONFIGURATION ---
INPUT_FILE = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/blazepose_final/S1/acting1/cam1/blazepose_S1_acting1_cam1_seg0_s1_o0.npy"
CHECKPOINT_FILE = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/checkpoints/best_model_hip_epoch26.pth"

WINDOW_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- FORWARD KINEMATICS ENGINE ---
class TrajectoryFK(torch.nn.Module):
    """
    Reconstructs 3D Pose from Model Predictions (21 Joints).
    
    Model Output (21, 6):
    - Index 0: Hip/Pelvis Rotation DELTA [6D]
    - Index 1-20: Body Local Rotations [6D]
    
    Logic:
    - Root (Joint 0) is FIXED at (0,0,0) Identity.
    - Hip (Joint 1) Accumulates Deltas: H_t = H_{t-1} @ Delta_t
    """
    def __init__(self):
        super().__init__()
        skel = get_totalcapture_skeleton()
        self.register_buffer('parents', torch.tensor(skel['parents'], dtype=torch.long))
        self.register_buffer('offsets', skel['offsets'].float())

    def cont6d_to_mat(self, d6):
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * torch.sum(b1 * a2, dim=-1, keepdim=True))
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-1)

    def forward(self, pred_series):
        """
        pred_series: (SeqLen, 21, 6)
        Returns: (SeqLen, 22, 3) - Global 3D Positions (Includes Fixed Root)
        """
        L = pred_series.shape[0]
        device = pred_series.device
        
        # 1. Convert all outputs to Matrices
        rot_mats = self.cont6d_to_mat(pred_series) # (L, 21, 3, 3)
        
        all_global_pos = []
        
        # Initialize Hip Accumulator (Identity)
        current_hip_rot = torch.eye(3, device=device)

        # 2. Time Integration Loop
        for t in range(L):
            frame_pos = [None] * 22
            frame_rots = [None] * 22
            
            # --- JOINT 0: WORLD ROOT (FIXED ANCHOR) ---
            # Forced to (0,0,0) Identity
            frame_rots[0] = torch.eye(3, device=device)
            frame_pos[0] = torch.zeros(3, device=device)
            
            # --- JOINT 1: HIPS (Accumulate Delta) ---
            # Index 0 in prediction is the Hip Delta
            delta_hip = rot_mats[t, 0] 
            
            # Update Accumulator: New = Old * Delta
            current_hip_rot = torch.matmul(current_hip_rot, delta_hip)
            
            # Calculate Global Hip Transform
            # Since Parent(1) is 0 (Identity), Global Hip Rot IS current_hip_rot
            frame_rots[1] = current_hip_rot
            
            p1 = self.parents[1].item() # 0
            off1 = self.offsets[1]
            frame_pos[1] = frame_pos[p1] + torch.matmul(frame_rots[p1], off1)

            # --- JOINTS 2-21: BODY (Standard FK) ---
            # Prediction Indices 1-20 map to Skeleton Indices 2-21
            for i in range(2, 22):
                pred_idx = i - 1 
                
                p = self.parents[i].item()
                local_rot = rot_mats[t, pred_idx]
                
                # Global Rot = Parent Global * Local
                frame_rots[i] = torch.matmul(frame_rots[p], local_rot)
                
                # Global Pos = Parent Pos + (Parent Global Rot * Offset)
                off = self.offsets[i]
                frame_pos[i] = frame_pos[p] + torch.matmul(frame_rots[p], off)
            
            all_global_pos.append(torch.stack(frame_pos))

        return torch.stack(all_global_pos) # (L, 22, 3)

# --- VISUALIZER ---
PLOTTING_PARENTS = [0, 1] + [p-2 if p >= 2 else 1 for p in get_totalcapture_skeleton()['parents'][2:]]

class SkeletonVisualizer:
    def __init__(self, predictions):
        self.data = predictions # (Frames, 22, 3)
        self.num_frames = len(predictions)
        self.parents = get_totalcapture_skeleton()['parents']
        
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.25)
        
        self.scats = self.ax.scatter([], [], [], c='r', s=20)
        self.lines = [self.ax.plot([], [], [], 'b-')[0] for _ in range(len(self.parents))]
        
        # Fixed limits 
        limit = 1.2
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_zlim(-limit, limit)
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Z (Depth)')
        self.ax.set_zlabel('Y (Height)')
        self.ax.set_title("Reconstructed Motion (Accumulated Hip)")
        
        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.slider = Slider(ax_slider, 'Frame', 0, self.num_frames - 1, valinit=0, valfmt='%0.0f')
        self.slider.on_changed(self.update)
        self.update(0)

    def update(self, val):
        frame_idx = int(self.slider.val)
        p = self.data[frame_idx]
        
        # Mapping: TotalCapture Y=Up -> Matplotlib Z=Up
        # No Flipping, just standard mapping
        xs = p[:, 0]
        ys = p[:, 1] # Depth -> Y axis on plot
        zs = p[:, 2] # Height -> Z axis on plot
        
        self.scats._offsets3d = (xs, ys, zs)
        
        for i, p_idx in enumerate(self.parents):
            if i == 0: continue 
            self.lines[i].set_data([xs[p_idx], xs[i]], [ys[p_idx], ys[i]])
            self.lines[i].set_3d_properties([zs[p_idx], zs[i]])
            
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()

# --- MAIN ---
def main():
    print(f"--- Inference (21-Joint Output) on {DEVICE} ---")
    
    # 1. Init Model (Updated Config)
    print("1. Init Model...")
    model = MotionTransformer(
        num_joints=27,      
        input_feats=20,     # 20 Features
        num_joints_out=21,  # 21 Output Joints
        d_model=512,
        num_layers=6,
        n_head=8,
        d_ff=1024,
        dropout=0.1
    ).to(DEVICE)

    # 2. Load Weights
    print(f"2. Loading: {os.path.basename(CHECKPOINT_FILE)}")
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # 3. Load & Process Data
    print(f"3. Processing: {os.path.basename(INPUT_FILE)}")
    raw_data = np.nan_to_num(np.load(INPUT_FILE).astype(np.float32))
    
    # Process using Updated 20-Feature Processor
    features, _ = process_blazepose_frames(raw_data, WINDOW_SIZE)
    input_tensor = torch.from_numpy(features).to(DEVICE)
    
    # 4. Inference loop
    print("4. Inference...")
    predictions_raw = []
    
    with torch.no_grad():
        # Sliding window inference
        for i in range(0, len(input_tensor), 1): 
            batch = input_tensor[i : i+1] # (1, 64, 27, 20)
            pred = model(batch) # (1, 64, 21, 6)
            
            # Take the last frame prediction
            pred_last = pred[0, -1, :, :] # (21, 6)
            predictions_raw.append(pred_last)
            
    full_pred_tensor = torch.stack(predictions_raw) # (L, 21, 6)
    
    # 5. Reconstruct Skeleton (FK with Hip Accumulation)
    print("5. Reconstructing Skeleton (FK)...")
    reconstructor = TrajectoryFK().to(DEVICE)
    global_positions = reconstructor(full_pred_tensor) # (L, 22, 3)
    
    global_np = global_positions.cpu().numpy()
    print(f"   Motion generated: {global_np.shape} frames")
    
    # 6. Visualize
    viz = SkeletonVisualizer(global_np)
    viz.show()

if __name__ == "__main__":
    main()