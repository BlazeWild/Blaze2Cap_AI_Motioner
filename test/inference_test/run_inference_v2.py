import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- ENVIRONMENT SETUP ---
# Add the project root to sys.path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
# Current script is at: .../Blaze2Cap/test/inference_test/run_inference_v2.py
# We need to go up 3 levels to reach ___MOTION_T_LIGHTNING
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir))) # Should be .../___MOTION_T_LIGHTNING
if project_root not in sys.path:
    sys.path.append(project_root)

# Now we can import modules directly from their .py files to avoid __init__.py conflicts
try:
    # Add the blaze2cap directory to sys.path
    blaze2cap_modules_dir = os.path.join(project_root, 'Blaze2Cap', 'blaze2cap', 'modules')
    if blaze2cap_modules_dir not in sys.path:
        sys.path.insert(0, blaze2cap_modules_dir)
    
    # Import directly from .py files (this will skip all __init__.py files)
    import models
    import pose_processing
    
    MotionTransformer = models.MotionTransformer
    process_blazepose_frames = pose_processing.process_blazepose_frames
    
    print("✅ Modules imported successfully")
    
except ImportError as e:
    print("Error importing modules. Please ensure your PYTHONPATH is set correctly or run this script from the project root.")
    print(f"Current sys.path: {sys.path}")
    print(f"Import error: {e}")
    raise e

# --- CONFIGURATION ---
INPUT_FILE = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/blazepose_final/S1/acting1/cam1/blazepose_S1_acting1_cam1_seg0_s1_o0.npy"
CHECKPOINT_FILE = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/checkpoints/checkpoint_epoch33.pth"
OUTPUT_DIR = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/test/inference_test"
WINDOW_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MAIN PIPELINE ---
def main():
    # 1. Setup Model
    print("--- 1. LOADING MODEL ---")
    model = MotionTransformer(
        num_joints=27, 
        input_feats=18, 
        d_model=512, 
        num_layers=4, 
        n_head=8, 
        d_ff=1024, 
        dropout=0.1,
        max_len=512  # Matching checkpoint (512 vs default 1024)
    ).to(DEVICE)
    
    print(f"Loading weights from: {CHECKPOINT_FILE}")
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        # Assume the checkpoint itself is the state dict
        model.load_state_dict(checkpoint)
        
    model.eval()
    print("✅ Model loaded successfully.")
    
    # 2. Load & Preprocess Data
    print(f"--- 2. LOADING DATA & PREPROCESSING ---")
    print(f"Reading: {INPUT_FILE}")
    raw_numpy = np.load(INPUT_FILE)
    raw_numpy = np.nan_to_num(raw_numpy.astype(np.float32))
    print(f"Raw Input Shape: {raw_numpy.shape}")
    
    print("Running process_blazepose_frames...")
    features, masks = process_blazepose_frames(raw_numpy, window_size=WINDOW_SIZE)
    # features shape: (Frames, Window, Joints*Feats) or similar depending on implementation
    # Based on pose_processing.py: Returns X_windows (F, N, Features)
    
    input_tensor = torch.from_numpy(features).to(DEVICE)
    print(f"Processed Tensor Shape: {input_tensor.shape}")
    
    # 3. Inference
    print("--- 3. RUNNING INFERENCE ---")
    root_predictions = []
    body_predictions = []
    batch_size = 128
    
    with torch.no_grad():
        for i in range(0, len(input_tensor), batch_size):
            batch = input_tensor[i : i+batch_size]
            
            # Model returns (root_out, body_out)
            # root_out: [B, S, 2, 6] (Pos+Rot)
            # body_out: [B, S, 20, 6]
            root_out, body_out = model(batch, key_padding_mask=None)
            
            # Take the last frame from each sequence window (Predictions for the current frame)
            root_predictions.append(root_out[:, -1, :, :].cpu().numpy())
            body_predictions.append(body_out[:, -1, :, :].cpu().numpy())
            
            if i % 1000 == 0:
                print(f"Processed {i}/{len(input_tensor)} frames...")

    # Concatenate batches
    root_predictions = np.concatenate(root_predictions, axis=0)  # [N, 2, 6]
    body_predictions = np.concatenate(body_predictions, axis=0)  # [N, 20, 6]
    
    # 4. Format Output [N, 22, 6]
    print("--- 4. FORMATTING OUTPUT ---")
    # Reshape to [N, 22, 6] format:
    # Joint 0: root position delta (first channel of root_predictions)
    # Joint 1: root rotation delta (second channel of root_predictions)  
    # Joints 2-21: body rotations (20 joints)
    
    root_pos = root_predictions[:, 0, :]  # [N, 6] - position channel
    root_rot = root_predictions[:, 1, :]  # [N, 6] - rotation channel
    
    predictions = np.concatenate([
        root_pos[:, np.newaxis, :],    # [N, 1, 6] - joint 0
        root_rot[:, np.newaxis, :],    # [N, 1, 6] - joint 1
        body_predictions               # [N, 20, 6] - joints 2-21
    ], axis=1)
    
    print(f"Root: {root_predictions.shape}")
    print(f"Body: {body_predictions.shape}")
    print(f"Final Prediction Shape: {predictions.shape}")
    
    # 5. Save Output
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    input_name = os.path.basename(INPUT_FILE).replace('.npy', '')
    output_path = os.path.join(OUTPUT_DIR, f"pred_{input_name}.npy")
    np.save(output_path, predictions)
    print(f"✅ Saved predictions to: {output_path}")

if __name__ == "__main__":
    main()
