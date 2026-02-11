import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

# --- ENVIRONMENT SETUP ---
# Add the project root to sys.path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up to Blaze2Cap folder
blaze2cap_root = os.path.dirname(os.path.dirname(current_dir))
if blaze2cap_root not in sys.path:
    sys.path.insert(0, blaze2cap_root)

try:
    # Import from blaze2cap package
    from blaze2cap.modules.models_posonly import MotionTransformer
    from blaze2cap.modules.pose_processing_posonly import process_blazepose_frames
    from blaze2cap.utils.skeleton_config import get_totalcapture_skeleton
    
    print("✅ Modules imported successfully")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please check your PYTHONPATH or folder structure.")
    sys.exit(1)

# --- CONFIGURATION ---
# UPDATE THESE PATHS IF NECESSARY
INPUT_FILE = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/blazepose_final/S4/walking2/cam4/blazepose_S4_walking2_cam4_seg0_s1_o0.npy"
CHECKPOINT_FILE = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/checkpoints/best_model.pth"

WINDOW_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- SKELETON DEFINITION FOR PLOTTING ---
# We need to map the 20 output joints (indices 2-21 in TotalCapture) back to a connectivity graph.
# TotalCapture has 22 joints. Model outputs 20 joints (Indices 2 to 21).
# We need to adjust parent indices because our output array is 0-indexed (corresponding to joint 2).

# Original Parents for 22 joints:
# [-1, 0, 1, 2, 3, 4, 5, 6, 5, 8, 9, 10, 5, 12, 13, 14, 1, 16, 17, 1, 19, 20]
# Indices 0 (Hips_Pos) and 1 (Hips_Rot) are NOT in the output.
# Output Index 0 corresponds to TotalCapture Index 2 (Spine).

def get_plotting_parents():
    original_parents = get_totalcapture_skeleton()['parents']
    # Filter parents for joints 2-21
    # If a parent is 0 or 1, it becomes a "root" in our relative plot (connected to origin 0,0,0)
    # We map TC index i -> Output index (i-2)
    
    plot_parents = []
    for i in range(2, 22): # Iterate through output joints
        p_idx = original_parents[i]
        if p_idx < 2:
            plot_parents.append(-1) # Root (connected to origin)
        else:
            plot_parents.append(p_idx - 2)
    return plot_parents

PLOTTING_PARENTS = get_plotting_parents()


# --- VISUALIZER CLASS ---
class SkeletonVisualizer:
    def __init__(self, predictions):
        """
        predictions: (F, 20, 3) numpy array of positions
        """
        self.data = predictions
        self.num_frames = len(predictions)
        self.parents = PLOTTING_PARENTS
        
        # Setup Figure
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.25)
        
        # Plot Elements
        self.scats = self.ax.scatter([], [], [], c='r', s=20)
        self.lines = [self.ax.plot([], [], [], 'b-')[0] for _ in range(len(self.parents))]
        
        # Set Limits
        # TotalCapture is in Meters. Usually +/- 1.5m is enough.
        # Check data range to be safe
        all_x = self.data[:, :, 0].flatten()
        all_y = self.data[:, :, 1].flatten()
        all_z = self.data[:, :, 2].flatten()
        
        self.ax.set_xlim(np.min(all_x)-0.5, np.max(all_x)+0.5)
        self.ax.set_ylim(np.min(all_y)-0.5, np.max(all_y)+0.5)
        self.ax.set_zlim(np.min(all_z)-0.5, np.max(all_z)+0.5)
        
        self.ax.set_xlabel('X (Right)')
        self.ax.set_ylabel('Y (Down)')
        self.ax.set_zlabel('Z (Fwd)')
        self.ax.set_title("Inference Prediction (Canonical Space)")
        
        # Fix View (Optional: Adjust based on your preferred up-axis)
        # TotalCapture Y is down. We might want to invert Y axis for visualization.
        self.ax.invert_yaxis() 

        # Slider
        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.slider = Slider(ax_slider, 'Frame', 0, self.num_frames - 1, valinit=0, valfmt='%0.0f')
        self.slider.on_changed(self.update)
        
        # Draw Initial Frame
        self.update(0)

    def update(self, val):
        frame_idx = int(self.slider.val)
        current_pose = self.data[frame_idx] # (20, 3)
        
        xs = current_pose[:, 0]
        ys = current_pose[:, 1]
        zs = current_pose[:, 2]
        
        # Update Scatter
        self.scats._offsets3d = (xs, ys, zs)
        
        # Update Lines (Bones)
        for i, p_idx in enumerate(self.parents):
            if p_idx == -1:
                # Connect to Origin (0,0,0) if root
                self.lines[i].set_data([0, xs[i]], [0, ys[i]])
                self.lines[i].set_3d_properties([0, zs[i]])
            else:
                self.lines[i].set_data([xs[p_idx], xs[i]], [ys[p_idx], ys[i]])
                self.lines[i].set_3d_properties([zs[p_idx], zs[i]])
                
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()

# --- MAIN INFERENCE PIPELINE ---
def main():
    print(f"--- Running Inference on {DEVICE} ---")
    
    # 1. Load Model
    print("1. Initializing Model...")
    model = MotionTransformer(
        num_joints=19,
        input_feats=8,
        output_joints=20, # Must match model output head
        d_model=512,
        num_layers=4, # Ensure this matches training config
        n_head=8,
        d_ff=1024,
        dropout=0.1,
        max_len=512
    ).to(DEVICE)

    # 2. Load Checkpoint
    print(f"2. Loading Weights: {os.path.basename(CHECKPOINT_FILE)}")
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE, weights_only=False)
    
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    # Filter out potential mismatch keys if strictly needed, but usually exact match is best
    model.load_state_dict(state_dict)
    model.eval()
    
    # 3. Load & Process Data
    print(f"3. Processing Input: {os.path.basename(INPUT_FILE)}")
    raw_data = np.load(INPUT_FILE)  # (F, 25, 7)
    raw_numpy = np.nan_to_num(raw_data.astype(np.float32))
    
    # Preprocess (BlazePose -> Canonical Features)
    # Returns X_windows: (Num_Windows, 64, 19, 8)
    features, masks = process_blazepose_frames(raw_numpy, WINDOW_SIZE)
    
    input_tensor = torch.from_numpy(features).to(DEVICE)
    print(f"   Input Tensor Shape: {input_tensor.shape}")
    
    # 4. Inference Loop
    print("4. Running Model Inference...")
    predictions = []
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, len(input_tensor), batch_size):
            batch = input_tensor[i : i + batch_size]
            
            # Forward Pass
            # Output: (B, 64, 20, 3)
            out_batch = model(batch)
            
            # Take the LAST frame of each window for sequence reconstruction
            # OR take the center frame. Usually last frame for causal/streaming.
            # Shape: (B, 20, 3)
            current_frame_preds = out_batch[:, -1, :, :] 
            predictions.append(current_frame_preds.cpu().numpy())
            
    # Concatenate all batches
    full_pred = np.concatenate(predictions, axis=0) # (Total_Frames, 20, 3)
    print(f"   Prediction Shape: {full_pred.shape}")
    
    # 5. Visualize
    print("5. Launching Visualizer...")
    viz = SkeletonVisualizer(full_pred)
    viz.show()

if __name__ == "__main__":
    main()