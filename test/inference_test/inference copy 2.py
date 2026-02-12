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

# IMPORT STANDARD MODULES (As per our final cleanup)
from blaze2cap.modules.models import MotionTransformer
from blaze2cap.modules.pose_processing import process_blazepose_frames
from blaze2cap.utils.skeleton_config import get_totalcapture_skeleton

# --- CONFIGURATION ---
# INPUT_FILE = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/blazepose_final/S5/rom3/cam1/blazepose_S5_rom3_cam1_seg0_s1_o0.npy"
INPUT_FILE ="/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/blazepose_final/S1/acting1/cam1/blazepose_S1_acting1_cam1_seg0_s1_o0.npy"
CHECKPOINT_FILE = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/checkpoints/best_modelepoch41all.pth"

WINDOW_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- TRAJECTORY RECONSTRUCTION ENGINE ---
class TrajectoryFK(torch.nn.Module):
    """
    Reconstructs Global Motion from Model Predictions.
    
    Model Output (22, 6):
    - Index 0: Root Position Velocity (Local)
    - Index 1: Root Rotation Velocity (Local)
    - Index 2-21: Body Pose (Local Rotations)
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
        pred_series: (SeqLen, 22, 6) - Stream of predictions
        Returns: (SeqLen, 22, 3) - Global 3D Positions
        """
        L = pred_series.shape[0]
        device = pred_series.device
        
        # 1. Convert all outputs to Matrices
        # Shape: (L, 22, 3, 3)
        rot_mats = self.cont6d_to_mat(pred_series)
        
        # 2. Initialize Global State
        # We start at (0,0,0) facing +Z (Identity)
        current_root_pos = torch.zeros(3, device=device)
        current_root_rot = torch.eye(3, device=device)
        
        all_global_pos = []

        # 3. Time Integration Loop
        for t in range(L):
            # --- A. UPDATE ROOT TRAJECTORY ---
            
            # 1. Get Deltas from Model
            # Index 0: Linear Velocity [vx, vy, vz] (Local to current facing)
            root_vel_local = pred_series[t, 0, :3] 
            
            # Index 1: Angular Velocity (Rotation Matrix)
            root_rot_delta = rot_mats[t, 1] 
            
            # 2. Update Global Rotation
            # New Rot = Old Rot @ Delta Rot
            current_root_rot = torch.matmul(current_root_rot, root_rot_delta)
            
            # 3. Update Global Position
            # New Pos = Old Pos + (Old Rot @ Local Vel)
            # (Move forward in the direction we were facing *before* turning)
            global_vel = torch.matmul(current_root_rot, root_vel_local)
            current_root_pos = current_root_pos + global_vel
            
            # --- B. COMPUTE BODY POSE (FK) ---
            
            frame_pos = [None] * 22
            frame_rots = [None] * 22
            
            # Root (Index 0/1) is now set
            frame_rots[0] = current_root_rot
            frame_pos[0] = current_root_pos
            
            # Index 1 (Hips) is usually same as 0 in TotalCapture or small offset
            # We treat 1 as the start of the chain for simplicity here,
            # using the updated root rotation.
            frame_rots[1] = current_root_rot 
            off1 = self.offsets[1]
            frame_pos[1] = frame_pos[0] + torch.matmul(frame_rots[0], off1)

            # Rest of Body (2-21)
            for i in range(2, 22):
                p = self.parents[i].item()
                local_rot = rot_mats[t, i]
                
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
        # Use TotalCapture parents for plotting structure
        self.parents = get_totalcapture_skeleton()['parents']
        
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.25)
        
        # Plot Objects
        self.scats = self.ax.scatter([], [], [], c='r', s=20)
        self.lines = [self.ax.plot([], [], [], 'b-')[0] for _ in range(len(self.parents))]
        
        # --- FIXED SCALE & LIMITS (Do not update per frame) ---
        # Setting a fixed 4x4 meter stage centered at origin
        # Adjust these values if your character walks further than 2 meters
        limit = 2.0 
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_zlim(-limit, limit)
        
        # Draw a static "Floor" grid for reference
        xx, zz = np.meshgrid(np.linspace(-limit, limit, 10), np.linspace(-limit, limit, 10))
        yy = np.zeros_like(xx)
        self.ax.plot_wireframe(xx, yy, zz, color='gray', alpha=0.2)
        
        self.ax.set_xlabel('X (Meters)')
        self.ax.set_ylabel('Y (Height)')
        self.ax.set_zlabel('Z (Depth)')
        self.ax.set_title("Global Trajectory (Fixed World Frame)")
        
        # Slider
        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.slider = Slider(ax_slider, 'Frame', 0, self.num_frames - 1, valinit=0, valfmt='%0.0f')
        self.slider.on_changed(self.update)
        self.update(0)

    def update(self, val):
        frame_idx = int(self.slider.val)
        current_pose = self.data[frame_idx]
        
        # Extract coordinates
        xs = current_pose[:, 0]
        ys = current_pose[:, 1]
        zs = current_pose[:, 2]
        
        # Update Scatter (Joints)
        self.scats._offsets3d = (xs, ys, zs)
        
        # Update Lines (Bones)
        for i, p_idx in enumerate(self.parents):
            if i == 0: continue # Root has no parent line
            
            # Draw line from Parent -> Child
            self.lines[i].set_data(
                [xs[p_idx], xs[i]], 
                [ys[p_idx], ys[i]]
            )
            self.lines[i].set_3d_properties(
                [zs[p_idx], zs[i]]
            )
            
        # CRITICAL: We do NOT reset set_xlim/ylim/zlim here.
        # This keeps the "camera" static so you see the character walk away.
            
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()

# --- MAIN ---
def main():
    print(f"--- Inference (Root Motion + Pose) on {DEVICE} ---")
    
    # 1. Load Model (19 Feats, 22 Output Joints)
    print("1. Init Model...")
    model = MotionTransformer(
        num_joints=27,      # 27 Input
        input_feats=19,     # 19 Features
        num_joints_out=22,  # 22 Output (Root + Body)
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
    
    # Process using 19-Feature Processor
    features, _ = process_blazepose_frames(raw_data, WINDOW_SIZE)
    input_tensor = torch.from_numpy(features).to(DEVICE)
    
    # 4. Inference loop
    print("4. Inference...")
    predictions_raw = []
    
    with torch.no_grad():
        # Sliding window inference
        for i in range(0, len(input_tensor), 1): # Stride 1 for smoothness
            batch = input_tensor[i : i+1] # (1, 64, 27, 19)
            pred = model(batch) # (1, 64, 22, 6)
            
            # Take the *middle* frame or *last* frame prediction?
            # For trajectory, sequential is best. We take the last frame.
            pred_last = pred[0, -1, :, :] # (22, 6)
            predictions_raw.append(pred_last)
            
    # Stack: (L, 22, 6)
    full_pred_tensor = torch.stack(predictions_raw)
    
    # 5. Reconstruct Global Trajectory
    print("5. Reconstructing Trajectory (FK)...")
    reconstructor = TrajectoryFK().to(DEVICE)
    global_positions = reconstructor(full_pred_tensor) # (L, 22, 3)
    
    global_np = global_positions.cpu().numpy()
    print(f"   Trajectory generated: {global_np.shape} frames")
    
    # 6. Visualize
    viz = SkeletonVisualizer(global_np)
    viz.show()

if __name__ == "__main__":
    main()