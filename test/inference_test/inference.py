import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D

# --- 1. SETUP PATHS & IMPORTS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# Current script is at: .../Blaze2Cap/test/inference_test/inference.py
# We need to go up 3 levels to reach ___MOTION_T_LIGHTNING
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    # Add the blaze2cap directories to sys.path
    blaze2cap_modules_dir = os.path.join(project_root, 'Blaze2Cap', 'blaze2cap', 'modules')
    blaze2cap_utils_dir = os.path.join(project_root, 'Blaze2Cap', 'blaze2cap', 'utils')
    if blaze2cap_modules_dir not in sys.path:
        sys.path.insert(0, blaze2cap_modules_dir)
    if blaze2cap_utils_dir not in sys.path:
        sys.path.insert(0, blaze2cap_utils_dir)
    
    # Import directly from .py files
    import models
    import pose_processing
    import skeleton_config
    
    MotionTransformer = models.MotionTransformer
    process_blazepose_frames = pose_processing.process_blazepose_frames
    get_totalcapture_skeleton = skeleton_config.get_totalcapture_skeleton
    
    print("✅ Successfully imported Blaze2Cap modules.")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(f"Project root: {project_root}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# --- 2. CONFIGURATION ---
# UPDATE THESE TO YOUR ACTUAL PATHS
INPUT_NPY = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/blazepose_final/S1/acting1/cam1/blazepose_S1_acting1_cam1_seg0_s1_o0.npy"
CHECKPOINT = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/checkpoints/checkpoint_epoch33.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 64

# --- 3. MATH HELPERS (FK) ---
def cont6d_to_mat(d6):
    """Converts 6D rotation representation to 3x3 Rotation Matrix."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * torch.sum(b1 * a2, dim=-1, keepdim=True))
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)

class SkeletonFK:
    def __init__(self, skeleton_config):
        self.parents = torch.tensor(skeleton_config['parents'], dtype=torch.long)
        self.offsets = skeleton_config['offsets'].clone() # (22, 3)
        
    def forward(self, root_pos, root_rot_6d, body_rot_6d):
        """Reconstructs full 3D skeleton from model predictions."""
        F_frames = root_pos.shape[0]
        device = root_pos.device
        
        # Convert Rotations to Matrices
        root_mat = cont6d_to_mat(root_rot_6d)   # (F, 3, 3)
        body_mats = cont6d_to_mat(body_rot_6d)  # (F, 20, 3, 3)
        
        # Initialize with Root
        # Note: TotalCapture indices 0 & 1 are both root-related
        global_rots = [root_mat, root_mat] 
        global_pos = [root_pos, root_pos]
        
        # Recursive FK Loop (Start from Joint 2)
        # 0=Hips, 1=HipsRot -> 2=Spine...
        for i in range(2, 22):
            parent_idx = self.parents[i].item()
            offset = self.offsets[i].to(device).view(1, 3, 1) # (1, 3, 1)
            
            parent_rot = global_rots[parent_idx] # (F, 3, 3)
            parent_pos = global_pos[parent_idx]  # (F, 3)
            
            # body_mats index 0 corresponds to joint 2
            local_rot = body_mats[:, i-2] 
            
            # Global Rotation
            curr_rot = torch.matmul(parent_rot, local_rot)
            global_rots.append(curr_rot)
            
            # Global Position
            rotated_offset = torch.matmul(parent_rot, offset).squeeze(-1)
            curr_pos = parent_pos + rotated_offset
            global_pos.append(curr_pos)
            
        return torch.stack(global_pos, dim=1) # (Frames, 22, 3)

# --- 4. INTERACTIVE VISUALIZER CLASS ---
class InteractiveSkeletonPlot:
    def __init__(self, xyz_data, parents):
        self.xyz_data = xyz_data # Shape (F, 22, 3)
        self.parents = parents
        self.num_frames = len(xyz_data)
        
        # Create Figure
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.15) # Make room for slider
        
        self.setup_plot()
        self.create_slider()
        
        # Draw Frame 0
        self.update_frame(0)

    def setup_plot(self):
        # Initial empty plots
        self.scatters = self.ax.scatter([], [], [], c='r', marker='o', s=20)
        self.lines = [self.ax.plot([], [], [], 'b-', linewidth=2)[0] for _ in range(len(self.parents))]
        
        # Set Limits (Auto-scaled to data)
        # Note: In matplotlib 3D, Z is usually "Up". 
        # TotalCapture often uses Y as Up/Down. 
        # We mapped (x, z, -y) earlier, so let's check bounds.
        
        all_x = self.xyz_data[:, :, 0]
        all_y = self.xyz_data[:, :, 1]
        all_z = self.xyz_data[:, :, 2]
        
        mid_x, mid_y, mid_z = np.mean(all_x), np.mean(all_y), np.mean(all_z)
        radius = 1.5 # Meters
        
        self.ax.set_xlim(mid_x - radius, mid_x + radius)
        self.ax.set_ylim(mid_y - radius, mid_y + radius)
        self.ax.set_zlim(mid_z - radius, mid_z + radius)
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f"Reconstructed Motion (Epoch 33)")

    def create_slider(self):
        # Create Slider Axes
        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor='lightgoldenrodyellow')
        
        self.slider = Slider(
            ax=ax_slider,
            label='Frame',
            valmin=0,
            valmax=self.num_frames - 1,
            valinit=0,
            valstep=1
        )
        
        # Link update function
        self.slider.on_changed(self.update_frame)

    def update_frame(self, val):
        idx = int(self.slider.val)
        current_pose = self.xyz_data[idx] # (22, 3)
        
        xs = current_pose[:, 0]
        ys = current_pose[:, 1]
        zs = current_pose[:, 2]
        
        # Update Joints
        self.scatters._offsets3d = (xs, ys, zs)
        
        # Update Bones
        for i, parent_idx in enumerate(self.parents):
            if parent_idx == -1 or i == parent_idx: continue
            
            line_xs = [xs[i], xs[parent_idx]]
            line_ys = [ys[i], ys[parent_idx]]
            line_zs = [zs[i], zs[parent_idx]]
            
            self.lines[i].set_data(line_xs, line_ys)
            self.lines[i].set_3d_properties(line_zs)
            
        self.ax.set_title(f"Frame: {idx} / {self.num_frames}")
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()

# --- 5. MAIN LOGIC ---
def main():
    print("--- 1. Loading Model ---")
    model = MotionTransformer(
        num_joints=27, input_feats=18, d_model=512, 
        num_layers=4, n_head=8, d_ff=1024, max_len=512
    ).to(DEVICE)
    
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"--- 2. Processing Data: {os.path.basename(INPUT_NPY)} ---")
    raw_numpy = np.nan_to_num(np.load(INPUT_NPY).astype(np.float32))
    
    # Process using the helper function
    X, _ = process_blazepose_frames(raw_numpy, window_size=WINDOW_SIZE)
    input_tensor = torch.from_numpy(X).to(DEVICE) # (F, 64, 486)
    
    print(f"--- 3. Running Inference on {len(input_tensor)} frames ---")
    all_root_vel = []
    all_root_rot = []
    all_body_rot = []
    
    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(input_tensor), batch_size):
            batch = input_tensor[i : i+batch_size]
            
            # Forward Pass
            root_out, body_out = model(batch, key_padding_mask=None)
            
            # Take last frame of each window
            all_root_vel.append(root_out[:, -1, 0, :3]) 
            all_root_rot.append(root_out[:, -1, 1, :6]) 
            all_body_rot.append(body_out[:, -1, :, :]) 
            
    # Concatenate
    pred_vel = torch.cat(all_root_vel, dim=0)   # (N, 3)
    pred_root_rot = torch.cat(all_root_rot, dim=0) # (N, 6)
    pred_body_rot = torch.cat(all_body_rot, dim=0) # (N, 20, 6)
    
    # Integrate Velocity -> Position
    pred_root_pos = torch.cumsum(pred_vel, dim=0)
    
    # Run FK
    print("--- 4. Reconstructing Skeleton (FK) ---")
    skel_cfg = get_totalcapture_skeleton()
    fk_engine = SkeletonFK(skel_cfg)
    xyz_output = fk_engine.forward(pred_root_pos, pred_root_rot, pred_body_rot)
    
    # Prepare for Visualization
    xyz_np = xyz_output.cpu().numpy()
    
    # IMPORTANT: RE-ORIENT FOR VISUALIZATION
    # TotalCapture often has Y-down. We flip Y and swap axes to make it look upright in Matplotlib.
    # Current: (X, Y, Z). Let's try to map to (X, Z, Y_up) style if needed.
    # For now, just flipping Y usually works for standard Mocap.
    xyz_np[:, :, 1] *= -1 
    
    print("--- 5. Launching Interactive Plot ---")
    viz = InteractiveSkeletonPlot(xyz_np, skel_cfg['parents'])
    viz.show()

if __name__ == "__main__":
    main()