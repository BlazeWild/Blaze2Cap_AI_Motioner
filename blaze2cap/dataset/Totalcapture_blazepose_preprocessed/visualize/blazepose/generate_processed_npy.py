import numpy as np
import sys
import os

# Add the project root to path to allow importing blaze2cap
# Current file is in: .../blaze2cap/dataset/Totalcapture_blazepose_preprocessed/visualize/blazepose/
# proper import requires: .../Blaze2Cap/ (parent of the blaze2cap package)
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up 6 levels just to be safe and cover variations, searching for the root containing 'blaze2cap' package
# Level 0: blazepose
# Level 1: visualize
# Level 2: Totalcapture_blazepose_preprocessed
# Level 3: Dataset (Note: path in workspace is "dataset" or "Dataset"? Workspace info says "dataset" and "Totalcapture_blazepose_preprocessed" is inside "dataset"? No, wait.)

# Workspace info:
# Blaze2Cap/ 
#   blaze2cap/
#     ...
# 
# The file path is:
# /home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/blaze2cap/dataset/Totalcapture_blazepose_preprocessed/visualize/blazepose/generate_processed_npy.py

# So:
# 1. blazepose
# 2. visualize
# 3. Totalcapture_blazepose_preprocessed
# 4. dataset
# 5. blaze2cap
# 6. Blaze2Cap (This is what we want)

project_root = os.path.abspath(os.path.join(current_dir, "../../../../../"))
sys.path.append(project_root)

from blaze2cap.modules.pose_processing import process_blazepose_frames

INPUT_FILE = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/blazepose_final/S1/acting1/cam1/blazepose_S1_acting1_cam1_seg0_s1_o0.npy"
OUTPUT_FILE = "processed_features_27_18.npy"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found: {INPUT_FILE}")
        return

    print(f"Loading raw data from: {INPUT_FILE}")
    # Load raw data (Frames, 25, 7)
    raw_data = np.load(INPUT_FILE)
    print(f"Raw data shape: {raw_data.shape}")

    # Process blazepose frames
    # window_size=1 ensures we get per-frame data without temporal windowing context stacking
    print("Processing features (adding virtual joints, centering, calculating deltas)...")
    X_windows, _ = process_blazepose_frames(raw_data, window_size=1)
    
    # X_windows shape comes out as (Frames, 1, Flattened_Features)
    # Flattened_Features = 27 joints * 18 channels = 486
    print(f"Windowed output shape: {X_windows.shape}")
    
    # We want (Frames, 27, 18)
    # First remove the window dimension (index 1)
    features_flat = X_windows[:, 0, :] # Shape: (Frames, 486)
    
    # Reshape to (Frames, 27, 18)
    final_features = features_flat.reshape(features_flat.shape[0], 27, 18)
    
    print(f"Final reshaped output: {final_features.shape}")
    print("  - Feature channels: 3(pos) + 3(vel) + 3(parent_vec) + 3(child_vec) + 2(screen) + 2(screen_vel) + 1(vis) + 1(anchor) = 18")

    # Save
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
    np.save(save_path, final_features)
    print(f"Saved processed file to: {save_path}")

if __name__ == "__main__":
    main()
