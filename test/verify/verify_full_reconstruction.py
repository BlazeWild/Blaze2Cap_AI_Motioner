
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__)) # test/verify
blaze2cap_root = os.path.dirname(os.path.dirname(current_dir)) # Blaze2Cap
if blaze2cap_root not in sys.path:
    sys.path.insert(0, blaze2cap_root)

# Imports
from blaze2cap.utils.skeleton_config import get_totalcapture_skeleton


# --- CONFIGURATION ---
GT_FILE_PATH = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/gt_final/S1/acting1/cam1/gt_S1_acting1_cam1_seg0_s1_o0.npy"

# --- HELPER FUNCTIONS ---

def cont6d_to_mat(d6):
    """
    Converts 6D rotation representation to 3x3 rotation matrix.
    Input: (..., 6)
    Output: (..., 3, 3)
    """
    d6 = torch.tensor(d6) if not isinstance(d6, torch.Tensor) else d6
    
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * torch.sum(b1 * a2, dim=-1, keepdim=True))
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1).numpy()

def reconstruct_full_motion(gt_data):
    """
    Reconstructs full motion using:
    - Index 1: Hip Rotation Delta (Accumulate)
    - Index 2-21: Body Local Rotations (Direct)
    - Index 0: Root Position Delta (IGNORED -> Fixed at 0,0,0)
    """
    frames = gt_data.shape[0]
    
    # 1. Extract & Convert Rotations
    # gt_data shape: (F, 22, 6)
    
    # Hip Delta (Index 1)
    hip_delta_mats = cont6d_to_mat(gt_data[:, 1, :]) # (F, 3, 3)
    
    # Body Rotations (Index 2-21)
    body_local_mats = cont6d_to_mat(gt_data[:, 2:22, :]) # (F, 20, 3, 3)
    
    # 2. Accumulate Hip Rotation
    hip_global_rots = []
    curr_rot = np.eye(3) # Initial Hip Orientation = Identity
    
    for t in range(frames):
        delta = hip_delta_mats[t]
        curr_rot = curr_rot @ delta
        hip_global_rots.append(curr_rot)
        
    hip_global_rots = np.stack(hip_global_rots) # (F, 3, 3)
    
    # 3. Perform FK for Full Skeleton
    skel = get_totalcapture_skeleton()
    parents = skel['parents'] # List
    offsets = skel['offsets'].numpy() # (22, 3)
    
    all_global_pos = []
    
    for t in range(frames):
        frame_pos = np.zeros((22, 3))
        frame_rots = np.zeros((22, 3, 3))
        
        # ROOT (Joint 0) -> Fixed at (0,0,0), I
        frame_pos[0] = [0, 0, 0]
        frame_rots[0] = np.eye(3)
        
        # HIPS (Joint 1) -> Pos relative to Root, Rot from Accumulator
        frame_rots[1] = hip_global_rots[t]
        frame_pos[1] = frame_pos[0] + frame_rots[0] @ offsets[1]
        
        # BODY (Joints 2-21)
        for i in range(2, 22):
            p = parents[i]
            
            # Local Rotation from GT (Index i-2 in body_local_mats)
            local_rot = body_local_mats[t, i-2] 
            
            # Global Rot = ParentGlobal @ Local
            frame_rots[i] = frame_rots[p] @ local_rot
            
            # Global Pos = ParentGlobalPos + ParentGlobalRot @ Offset
            frame_pos[i] = frame_pos[p] + frame_rots[p] @ offsets[i]
            
        all_global_pos.append(frame_pos)
        
    return np.array(all_global_pos) # (F, 22, 3)


# --- VISUALIZATION ---
class SimpleVisualizer:
    def __init__(self, data):
        self.data = data
        self.num_frames = len(data)
        self.skel = get_totalcapture_skeleton()
        self.parents = self.skel['parents']
        
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.25)
        
        self.lines = [self.ax.plot([], [], [], 'b-')[0] for _ in range(len(self.parents))]
        self.scats = self.ax.scatter([], [], [], c='r', s=10)
        
        # Root trace
        self.root_trace, = self.ax.plot([], [], [], 'g-', linewidth=1, alpha=0.5)
        
        # Fixed Scale (-2, 2)
        limit = 2.0
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_zlim(-limit, limit)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title("Full Reconstruction (Hip Delta + Body Rot)")
        
        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.slider = Slider(ax_slider, 'Frame', 0, self.num_frames - 1, valinit=0, valfmt='%0.0f')
        self.slider.on_changed(self.update)
        self.update(0)
        
    def update(self, val):
        idx = int(self.slider.val)
        frame = self.data[idx] # (22, 3)
        
        xs, ys, zs = frame[:, 0], frame[:, 1], frame[:, 2]
        
        self.scats._offsets3d = (xs, ys, zs)
        
        for i, p in enumerate(self.parents):
            if i == 0: continue # Skip root parent pointer
            self.lines[i].set_data([xs[p], xs[i]], [ys[p], ys[i]])
            self.lines[i].set_3d_properties([zs[p], zs[i]])
            
        # Draw trace of Hips (Joint 1)
        # Use last 100 frames
        start = max(0, idx - 100)
        trace = self.data[start:idx+1, 1, :]
        if len(trace) > 0:
            self.root_trace.set_data(trace[:, 0], trace[:, 1])
            self.root_trace.set_3d_properties(trace[:, 2])
            
        self.fig.canvas.draw_idle()
        
    def show(self):
        plt.show()

def main():
    if not os.path.exists(GT_FILE_PATH):
        print(f"Error: File not found: {GT_FILE_PATH}")
        return

    print(f"Loading: {GT_FILE_PATH}")
    gt_data = np.load(GT_FILE_PATH)
    print(f"Shape: {gt_data.shape}") # Expect (F, 22, 6)
    
    print("Reconstructing Full Motion...")
    reconstructed = reconstruct_full_motion(gt_data)
    
    print("Visualizing...")
    viz = SimpleVisualizer(reconstructed)
    viz.show()

if __name__ == "__main__":
    main()
