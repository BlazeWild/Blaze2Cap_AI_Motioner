import torch
import numpy as np

# Original Z-Up Offsets (TotalCapture)
OFFSETS_BVH_METERS = np.array([
    [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0462, -0.0692], [0.0, 0.0160, -0.0909],
    [0.0, 0.0080, -0.0920], [0.0, 0.0, -0.0923], [0.0, -0.0346, -0.2424], [0.0, -0.0221, -0.1250],
    [0.0292, -0.0486, -0.1576], [0.1449, 0.0, 0.0], [0.2889, 0.0, 0.0], [0.2196, 0.0, 0.0],
    [-0.0292, -0.0486, -0.1576], [-0.1449, 0.0, 0.0], [-0.2889, 0.0, 0.0], [-0.2196, 0.0, 0.0],
    [0.0866, 0.0, 0.0253], [0.0, 0.0, 0.3789], [0.0, 0.0, 0.3754],
    [-0.0866, 0.0, 0.0253], [0.0, 0.0, 0.3789], [0.0, 0.0, 0.3754]
])

# The Fix Matrix: X->-X (Right), Z->-Y (Down), Y->Z (Depth)
R_align = np.array([
    [-1,  0,  0], 
    [ 0,  0, -1], 
    [ 0,  1,  0] 
])

new_offsets = (R_align @ OFFSETS_BVH_METERS.T).T

print("-" * 60)
print("PASTE THIS INTO skeleton_config.py as OFFSETS_METERS")
print("-" * 60)
print("OFFSETS_METERS = torch.tensor([")
for row in new_offsets:
    print(f"    [{row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}],")
print("], dtype=torch.float32)")