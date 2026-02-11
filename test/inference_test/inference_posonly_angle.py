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
# Go up to Blaze2Cap folder
blaze2cap_root = os.path.dirname(os.path.dirname(current_dir))
if blaze2cap_root not in sys.path:
    sys.path.insert(0, blaze2cap_root)

try:
    # IMPORT ROTATION-SPECIFIC MODULES
    from blaze2cap.modules.models_posonly_angle import MotionTransformer
    from blaze2cap.modules.pose_processing_posonly_angle import process_blazepose_frames
    from blaze2cap.utils.skeleton_config import get_totalcapture_skeleton
    print("✅ Modules imported successfully")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

# --- CONFIGURATION ---
INPUT_FILE ="/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/blazepose_final/S5/rom3/cam1/blazepose_S5_rom3_cam1_seg0_s1_o0.npy"
CHECKPOINT_FILE = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/checkpoints/checkpoint_epoch49_poseonly_angle.pth"

WINDOW_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- FORWARD KINEMATICS ENGINE ---
# We need this to convert the Model's Output (Angles) -> Visualizer Input (Positions)
class CanonicalFK(torch.nn.Module):
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

    def forward(self, pred_rot_6d):
        # pred_rot_6d: (B, 20, 6)
        B, J, C = pred_rot_6d.shape
        device = pred_rot_6d.device
        
        # 1. Convert to Matrices (B, 20, 3, 3)
        body_mats = self.cont6d_to_mat(pred_rot_6d)
        
        # 2. Setup Canonical Root (Identity)
        eye = torch.eye(3, device=device).view(1, 3, 3).expand(B, 3, 3)
        zeros = torch.zeros((B, 3), device=device)
        
        global_rots = [None] * 22
        global_pos = [None] * 22
        
        # Roots (Indices 0 & 1)
        global_rots[0] = eye
        global_pos[0] = zeros
        
        off1 = self.offsets[1].view(1, 3, 1)
        global_rots[1] = eye
        global_pos[1] = global_pos[0] + torch.matmul(global_rots[0], off1).squeeze(-1)
        
        # 3. FK Loop (Indices 2 to 21)
        for i in range(2, 22):
            p = self.parents[i].item()
            # body_mats index 0 is skeleton index 2
            local_rot = body_mats[:, i-2] 
            
            global_rots[i] = torch.matmul(global_rots[p], local_rot)
            
            off = self.offsets[i].view(1, 3, 1)
            rot_off = torch.matmul(global_rots[p], off).squeeze(-1)
            global_pos[i] = global_pos[p] + rot_off
            
        # Stack & Slice Body
        full_pos = torch.stack(global_pos, dim=1) # (B, 22, 3)
        return full_pos[:, 2:22, :] # (B, 20, 3)

# --- VISUALIZER SETUP ---
def get_plotting_parents():
    original_parents = get_totalcapture_skeleton()['parents']
    plot_parents = []
    for i in range(2, 22): 
        p_idx = original_parents[i]
        if p_idx < 2:
            plot_parents.append(-1)
        else:
            plot_parents.append(p_idx - 2)
    return plot_parents

PLOTTING_PARENTS = get_plotting_parents()

class SkeletonVisualizer:
    def __init__(self, predictions):
        self.data = predictions
        self.num_frames = len(predictions)
        self.parents = PLOTTING_PARENTS
        
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.25)
        
        self.scats = self.ax.scatter([], [], [], c='r', s=20)
        self.lines = [self.ax.plot([], [], [], 'b-')[0] for _ in range(len(self.parents))]
        
        # Limits
        all_val = self.data.flatten()
        self.ax.set_xlim(-1.0, 1.0)
        self.ax.set_ylim(-1.0, 1.0)
        self.ax.set_zlim(-1.0, 1.0)
        
        self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')
        self.ax.set_title("Inference (Rotations -> FK -> Positions)")
        self.ax.invert_yaxis() 

        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.slider = Slider(ax_slider, 'Frame', 0, self.num_frames - 1, valinit=0, valfmt='%0.0f')
        self.slider.on_changed(self.update)
        self.update(0)

    def update(self, val):
        frame_idx = int(self.slider.val)
        current_pose = self.data[frame_idx]
        xs, ys, zs = current_pose[:, 0], current_pose[:, 1], current_pose[:, 2]
        
        self.scats._offsets3d = (xs, ys, zs)
        for i, p_idx in enumerate(self.parents):
            if p_idx == -1:
                self.lines[i].set_data([0, xs[i]], [0, ys[i]])
                self.lines[i].set_3d_properties([0, zs[i]])
            else:
                self.lines[i].set_data([xs[p_idx], xs[i]], [ys[p_idx], ys[i]])
                self.lines[i].set_3d_properties([zs[p_idx], zs[i]])
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()

# --- MAIN ---
def main():
    print(f"--- Inference (Rotation Model) on {DEVICE} ---")
    
    # 1. Load Model (Configured for Rotation)
    print("1. Init Model...")
    model = MotionTransformer(
        num_joints=27,      # <--- CHANGED: 27 Input Joints
        input_feats=14,     # <--- CHANGED: 14 Features
        output_joints=20,   # 20 Body Joints
        d_model=512,
        num_layers=4,
        n_head=8,
        d_ff=1024,
        dropout=0.1,
        max_len=512
    ).to(DEVICE)

    # 2. Load Weights
    print(f"2. Loading: {os.path.basename(CHECKPOINT_FILE)}")
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 3. Load & Process Data
    print(f"3. Processing: {os.path.basename(INPUT_FILE)}")
    raw_data = np.nan_to_num(np.load(INPUT_FILE).astype(np.float32))
    
    # Process using Angle/PosOnly Processor
    # Returns (Windows, 64, 27, 14)
    features, _ = process_blazepose_frames(raw_data, WINDOW_SIZE)
    input_tensor = torch.from_numpy(features).to(DEVICE)
    
    # 4. Inference & FK
    print("4. Inference & FK...")
    fk_engine = CanonicalFK().to(DEVICE)
    predictions_pos = []
    
    with torch.no_grad():
        for i in range(0, len(input_tensor), 64):
            batch = input_tensor[i : i+64]
            
            # Predict Rotations: (B, 64, 20, 6)
            pred_rot = model(batch)
            
            # Take last frame: (B, 20, 6)
            last_frame_rot = pred_rot[:, -1, :, :]
            
            # Convert to Positions via FK: (B, 20, 3)
            # This is the magic step!
            last_frame_pos = fk_engine(last_frame_rot)
            
            predictions_pos.append(last_frame_pos.cpu().numpy())
            
    full_pred = np.concatenate(predictions_pos, axis=0)
    print(f"   Output Shape: {full_pred.shape}")
    
    # 5. Visualize
    viz = SkeletonVisualizer(full_pred)
    viz.show()

if __name__ == "__main__":
    main()