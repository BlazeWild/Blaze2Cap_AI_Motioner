import torch
import numpy as np
import matplotlib
# Force TkAgg for interactive plots if available
try:
    matplotlib.use('TkAgg')
except ImportError:
    pass
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# Ensure python can find the blaze2cap module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blaze2cap.modules.models import MotionTransformer
# Import from data_loader to ensure consistency if possible, or use the one in modules/dataset
# Since we didn't refactor data_loader to export the function, we'll implement the preprocessing locally
# to GUARANTEE it matches the specific logic we want (Valid Static History).
from blaze2cap.utils.skeleton_config import get_raw_skeleton, get_totalcapture_skeleton

# ==========================================
# CONFIGURATION
# ==========================================
CHECKPOINT_PATH = '/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/checkpoints/checkpoint_epoch57.pth' # Update to latest if needed
TEST_FILE_PATH = '/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/training_dataset_both_in_out/blaze_augmented/S1/acting1/cam1/blaze_S1_acting1_cam1_seg0_s1_o0.npy'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Standard Parents (22 Joint Rig)
PARENTS_22 = [-1, 0, 1, 2, 3, 4, 5, 6, 5, 8, 9, 10, 5, 12, 13, 14, 1, 16, 17, 1, 19, 20]

# Global placeholders for Dynamic Skeleton
OFFSETS = None 

# ==========================================
# MATH UTILS
# ==========================================
def rotation_6d_to_matrix(r6d):
    x_raw, y_raw = r6d[0:3], r6d[3:6]
    # FIX: Add Epsilon to prevent NaNs (matching loss.py fix)
    x = x_raw / (np.linalg.norm(x_raw) + 1e-8)
    z = np.cross(x, y_raw)
    z = z / (np.linalg.norm(z) + 1e-8)
    y = np.cross(z, x)
    return np.column_stack((x, y, z))

def compute_fk(root_pos, all_rots, offsets):
    """
    Forward Kinematics using dynamic offsets.
    input:
        root_pos: (3,)
        all_rots: (22, 3, 3)
        offsets: (22, 3)
    """
    pos = np.zeros((22, 3))
    pos[0] = root_pos
    pos[1] = root_pos
    for i in range(2, 22):
        p = PARENTS_22[i]
        # Offset is in Parent Frame? 
        # loss.py: curr_pos = parent_pos + (parent_rot @ offset)
        # Verify: data_loader bone_vecs = world_child - world_parent
        # get_raw_skeleton extracts vecs.
        # So yes, P_child = P_parent + (R_parent * Offset)
        
        offset_vec = offsets[i] # (3,)
        rotated_offset = all_rots[p] @ offset_vec
        pos[i] = pos[p] + rotated_offset
    return pos

def decode_motion(deltas, offsets):
    """
    Decodes motion from deltas.
    deltas: (Frames, 22, 6)
    offsets: (22, 3)
    """
    n = deltas.shape[0]
    rec_pos = np.zeros((n, 3))
    rec_rot = np.zeros((n, 22, 3, 3))
    curr_pos = np.zeros(3)
    curr_rot = np.eye(3)
    
    rec_pos[0] = curr_pos
    rec_rot[0, 1] = curr_rot
    # Frame 0 body
    for j in range(2, 22):
         rec_rot[0, j] = rotation_6d_to_matrix(deltas[0, j, :])

    for f in range(1, n):
        # 1. Root Update
        v_local = deltas[f, 0, :3]
        R_delta = rotation_6d_to_matrix(deltas[f, 1, :])
        
        # LOCK ROOT FOR VISUALIZATION if desired (y-up vs z-up issues)
        # User requested correct calculation.
        # Assuming model outputs metric velocity in root frame.
        curr_pos = curr_pos + (curr_rot @ v_local) 
        # curr_pos = np.zeros(3) # Force to 0,0,0
        
        curr_rot = curr_rot @ R_delta
        
        rec_pos[f] = curr_pos
        rec_rot[f, 1] = curr_rot
        
        # 2. Body Update
        for j in range(2, 22):
            p = PARENTS_22[j]
            R_local = rotation_6d_to_matrix(deltas[f, j, :])
            rec_rot[f, j] = rec_rot[f, p] @ R_local
            
    return rec_pos, rec_rot

# ==========================================
# PREPROCESSING (Aligned with data_loader)
# ==========================================
def preprocess_input(raw_data, window_size=64):
    """
    Matches data_loader.py _process_vectorized logic exactly.
    Input: (F, 25, 7)
    Output: (F, Window, 450)
    """
    N = window_size
    F, J, _ = raw_data.shape
    
    # 1. Padding Strategy: Valid Static History
    padding_data = np.repeat(raw_data[0:1], N-1, axis=0) 
    full_data = np.concatenate([padding_data, raw_data], axis=0)
    F_pad = full_data.shape[0]

    # 2. Extract Components
    pos_raw = full_data[:, :, 0:3]
    screen_raw_01 = full_data[:, :, 3:5]
    vis_raw = full_data[:, :, 5:6]
    anchor_raw = full_data[:, :, 6:7] 
    
    is_anchor = (anchor_raw[:, 0, 0] == 0)

    # 3. Transform Screen
    screen_norm = (screen_raw_01 * 2.0) - 1.0

    # 4. Feature: Centered Screen Position
    screen_hip_center = (screen_norm[:, 15] + screen_norm[:, 16]) / 2.0
    screen_centered = screen_norm - screen_hip_center[:, None, :]

    # 5. Feature: Delta Screen
    delta_screen = np.zeros_like(screen_centered)
    delta_screen[1:] = screen_centered[1:] - screen_centered[:-1]
    delta_screen[is_anchor] = 0.0

    # 6. Feature: World Handling
    world_hip_center = (pos_raw[:, 15] + pos_raw[:, 16]) / 2.0
    world_centered = pos_raw - world_hip_center[:, None, :]
    
    delta_world = np.zeros_like(world_centered)
    delta_world[1:] = world_centered[1:] - world_centered[:-1]
    delta_world[is_anchor] = 0.0

    # 7. Feature: Vectors (Bone/Parent)
    # Reconstruct Children/Parents arrays locally to be safe
    children = np.arange(25)
    children[3], children[4] = 5, 6
    children[5], children[6] = 7, 8
    children[7], children[8] = 11, 12
    children[15], children[16] = 17, 18
    children[17], children[18] = 19, 20
    children[19], children[20] = 23, 24
    
    parents = np.arange(25)
    parents[5], parents[6] = 3, 4
    parents[7], parents[8] = 5, 6
    parents[[9, 11, 13]] = 7 
    parents[[10, 12, 14]] = 8 
    parents[17], parents[18] = 15, 16
    parents[19], parents[20] = 17, 18
    parents[[21, 23]] = 19 
    parents[[22, 24]] = 20 

    bone_vecs = world_centered[:, children] - world_centered
    leaf_mask = (children == np.arange(25))
    bone_vecs[:, leaf_mask] = 0 
    
    parent_vecs = world_centered - world_centered[:, parents]

    # 8. Stack Features
    features = np.concatenate([
        world_centered, delta_world, bone_vecs, parent_vecs,
        screen_centered, delta_screen, vis_raw, anchor_raw
    ], axis=2) 
    
    # Flatten
    D = 18 * 25
    features_flat = features.reshape(F_pad, D)
    
    # Clip
    features_flat = np.clip(features_flat, -10.0, 10.0)
    
    # Windowing
    strides = (features_flat.strides[0], features_flat.strides[0], features_flat.strides[1])
    X_windows = np.lib.stride_tricks.as_strided(
        features_flat, shape=(F, N, D), strides=strides
    )
    
    return X_windows.copy() # (F, N, D)

# ==========================================
# MAIN
# ==========================================
def main():
    print("Script started...")
    
    # 1. Load Model
    print("Loading Model...")
    model = MotionTransformer(
        num_joints=25, 
        input_feats=18, 
        d_model=256, 
        num_layers=4, 
        n_head=4, 
        d_ff=512, 
        dropout=0.1,
        max_len=512  
    ).to(DEVICE)
    
    # Load Checkpoint
    checkpoint_path = CHECKPOINT_PATH
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        # Try finding ANY checkpoint
        base_dir = os.path.dirname(checkpoint_path)
        
        # If base_dir is empty string (relative path), use current dir?
        # No, base_dir from a full path is usually valid.
        
        if os.path.exists(base_dir):
            files = [f for f in os.listdir(base_dir) if f.endswith('.pth')]
            if files:
                # Sort by name or modification time? Name usually implies epoch.
                # Assuming checkpoint_epochX.pth format.
                files.sort() 
                new_path = os.path.join(base_dir, files[-1])
                print(f"Found alternative: {new_path}")
                checkpoint_path = new_path
            else:
                print("No checkpoints found. Exiting.")
                return
        else:
             print("Checkpoint dir not found.")
             return

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Extract State Dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    try:
        model.load_state_dict(state_dict)
        print("✅ Weights loaded successfully.")
    except RuntimeError as e:
        print(f"❌ Weight loading failed: {e}")
        return
        
    model.eval()
    
    # 2. Load Data & Skeleton
    print(f"Loading: {TEST_FILE_PATH}")
    if not os.path.exists(TEST_FILE_PATH):
        print("File not found.")
        return
        
    raw_numpy = np.load(TEST_FILE_PATH).astype(np.float32) # (F, 25, 7)
    
    # --- DYNAMIC SKELETON EXTRACTION ---
    print("Extracting Skeleton from Data (usedraw)...")
    raw_pos = raw_numpy[:, :, 0:3]
    # Extract offsets using the helper
    # We use Frame 0 (or mean) via the helper
    dynamic_offsets = get_raw_skeleton(raw_pos) # Tensor (22, 3)
    
    # Convert to Numpy for inference script usage
    global OFFSETS
    OFFSETS = dynamic_offsets.cpu().numpy()
    print(f"Skeleton Extracted. Mean Bone Length: {np.linalg.norm(OFFSETS, axis=1).mean():.4f}")
    
    # 3. Process Features
    print("Processing features...")
    # Use the local aligned preprocessor
    features = preprocess_input(raw_numpy, window_size=64)
    
    input_tensor = torch.from_numpy(features).to(DEVICE)
    
    # 4. Inference
    print("Running Inference...")
    predictions = []
    batch_size = 128
    
    with torch.no_grad():
        for i in range(0, len(input_tensor), batch_size):
            batch = input_tensor[i : i+batch_size]
            # CRITICAL: Pass key_padding_mask=None to allow attending to history
            out = model.forward_combined(batch, key_padding_mask=None) 
            predictions.append(out[:, -1, :, :].cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    print(f"Prediction Shape: {predictions.shape}")

    # 5. Decode & Plot
    print(f"Decoding motion using Extracted Skeleton...")
    g_pos, g_rot = decode_motion(predictions, OFFSETS)
    
    print("Launching Plotter...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.2)
    lines = [ax.plot([],[],[], 'b-o', ms=4, mec='k')[0] for _ in range(22)]
    
    # Limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=-45)

    def update(val):
        f = int(slider.val)
        if f >= len(g_pos): f = len(g_pos) - 1
        
        # Use valid offsets
        xyz = compute_fk(g_pos[f], g_rot[f], OFFSETS)
        
        for i, line in enumerate(lines):
            c = i+1
            if c >= 22: continue
            p = PARENTS_22[c]
            line.set_data_3d(
                [xyz[p,0], xyz[c,0]], 
                [xyz[p,1], xyz[c,1]], 
                [xyz[p,2], xyz[c,2]]
            )
        
        # Optional: Center view on root
        # cx, cy, cz = xyz[0]
        # ax.set_xlim(cx-1, cx+1) ...

        fig.canvas.draw_idle()

    slider = Slider(plt.axes([0.2, 0.05, 0.6, 0.03]), 'Frame', 0, len(g_pos)-1, valinit=0, valfmt='%d')
    slider.on_changed(update)
    
    print("Plotting. Close window to finish.")
    plt.show()

# --- ENTRY POINT ---
if __name__ == "__main__":
    main()