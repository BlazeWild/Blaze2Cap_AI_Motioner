import numpy as np

path = "blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/blazepose_augmented/S1/acting1/cam1/blaze_S1_acting1_cam1_seg0_s1_o0.npy"
data = np.load(path)
frame0 = data[0, :, :3] # (25, 3)

print("--- Frame 0 Positions ---")
indices_of_interest = [0, 1, 2, 3, 4, 15, 16]
for i in indices_of_interest:
    print(f"Joint {i}: {frame0[i]}")

print("\n--- Distances/Directions ---")
# Check 15 vs 16 (Hips)
diff_15_16 = frame0[15] - frame0[16]
print(f"15 - 16 (Hip Vector): {diff_15_16} (Expect X-axis separation)")

# Check 3 vs 4 (Shoulders)
diff_3_4 = frame0[3] - frame0[4]
print(f"3 - 4 (Shldr Vector): {diff_3_4} (Expect X-axis separation)")

# Check Vertical (Shoulders vs Hips)
mid_hip = (frame0[15] + frame0[16]) / 2
mid_shldr = (frame0[3] + frame0[4]) / 2
diff_vert = mid_shldr - mid_hip
print(f"MidShldr - MidHip: {diff_vert} (Expect Negative Y or Positive Y depending on coord sys)")
# TotalCapture Y is Up? Or Down?
# skeleton_config says: Y+ is Down.
# So Shoulders (Above) - Hips (Below) should be Negative Y.

# Check 0, 1, 2 location relative to Spine
print(f"Joint 0 relative to MidHip: {frame0[0] - mid_hip}")
print(f"Joint 1 relative to MidHip: {frame0[1] - mid_hip}")
print(f"Joint 2 relative to MidHip: {frame0[2] - mid_hip}")

