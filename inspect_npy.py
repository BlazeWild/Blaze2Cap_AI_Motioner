import numpy as np

blaze_path = '/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/blazepose_augmented/S1/acting1/cam1/blaze_S1_acting1_cam1_seg0_s1_o0.npy'
gt_path = '/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/gt_augmented/S1/acting1/cam1/gt_S1_acting1_cam1_seg0_s1_o0.npy'

print(f"Inspecting BlazePose file: {blaze_path}")
try:
    blaze_data = np.load(blaze_path)
    print(f"Shape: {blaze_data.shape}")
    print(f"Dtype: {blaze_data.dtype}")
    if blaze_data.shape[-1] >= 7:
        print("First frame, channel 7 (index 6) value:")
        print(blaze_data[0, :, 6])
        print("First 5 frames, channel 7 values for first joint:")
        print(blaze_data[:5, 0, 6])
    print(f"Min: {np.min(blaze_data)}, Max: {np.max(blaze_data)}")
    print(f"Any NaNs: {np.isnan(blaze_data).any()}")
except Exception as e:
    print(f"Error loading BlazePose file: {e}")

print("-" * 20)

print(f"Inspecting GT file: {gt_path}")
try:
    gt_data = np.load(gt_path)
    print(f"Shape: {gt_data.shape}")
    print(f"Dtype: {gt_data.dtype}")
    print(f"Min: {np.min(gt_data)}, Max: {np.max(gt_data)}")
    print(f"Any NaNs: {np.isnan(gt_data).any()}")
    print("First frame data (first 5 values):")
    print(gt_data[0].flatten()[:5])
except Exception as e:
    print(f"Error loading GT file: {e}")
