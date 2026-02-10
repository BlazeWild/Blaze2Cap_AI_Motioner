import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 1. Define the hierarchy: {Joint: (Parent, [Offset X, Y, Z])}
# Offsets are taken directly from your BVH HIERARCHY data
hierarchy = {
    'Hips': (None, [0.0, 0.0, 0.0]),
    'Spine': ('Hips', [0.0, 1.817770, -2.726650]),
    'Spine1': ('Spine', [0.0, 0.631300, -3.580300]),
    'Spine2': ('Spine1', [0.0, 0.316860, -3.621700]),
    'Spine3': ('Spine2', [0.0, 0.0, -3.635530]),
    'Neck': ('Spine3', [0.000001, -1.363320, -9.543270]),
    'Head': ('Neck', [0.0, -0.868040, -4.922911]),
    'RightShoulder': ('Spine3', [1.148680, -1.914470, -6.202890]),
    'RightArm': ('RightShoulder', [5.705120, 0.0, 0.0]),
    'RightForeArm': ('RightArm', [11.372740, 0.0, 0.000002]),
    'RightHand': ('RightForeArm', [8.645779, 0.0, 0.0]),
    'LeftShoulder': ('Spine3', [-1.148679, -1.914470, -6.202890]),
    'LeftArm': ('LeftShoulder', [-5.705120, 0.0, 0.0]),
    'LeftForeArm': ('LeftArm', [-11.372740, 0.0, -0.000002]),
    'LeftHand': ('LeftForeArm', [-8.645779, 0.0, 0.0]),
    'RightUpLeg': ('Hips', [3.407760, 0.0, 0.995530]),
    'RightLeg': ('RightUpLeg', [-0.000001, 0.0, 14.916970]),
    'RightFoot': ('RightLeg', [-0.000001, 0.0, 14.781019]),
    'LeftUpLeg': ('Hips', [-3.407760, 0.0, 0.995530]),
    'LeftLeg': ('LeftUpLeg', [-0.000001, 0.0, 14.916970]),
    'LeftFoot': ('LeftLeg', [-0.000001, 0.0, 14.781019]),
}

# 2. Compute Absolute Positions
abs_pos = {}

def get_global_position(name):
    if name in abs_pos:
        return abs_pos[name]
    
    parent, offset = hierarchy[name]
    if parent is None:
        pos = np.array(offset)
    else:
        pos = get_global_position(parent) + np.array(offset)
    
    abs_pos[name] = pos
    return pos

# Calculate for all joints
for joint in hierarchy:
    get_global_position(joint)

# Convert to Meters (1 inch = 0.0254 meters)
coords = np.array([abs_pos[j] for j in hierarchy]) * 0.0254

# 3. Normalize to [-1, 1] range
# Center the skeleton around the origin
centroid = coords.mean(axis=0)
coords_centered = coords - centroid

# Scale so the largest dimension fits within [-1, 1]
max_range = np.abs(coords_centered).max()
coords_norm = coords_centered / max_range

# Map back to a dictionary for plotting bones
norm_pos_dict = {name: coords_norm[i] for i, name in enumerate(hierarchy.keys())}

# 4. Plotting
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw the bones (lines)
for name, (parent, _) in hierarchy.items():
    if parent:
        p1 = norm_pos_dict[parent]
        p2 = norm_pos_dict[name]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                color='royalblue', linewidth=3, alpha=0.7)

# Draw the joints (points)
ax.scatter(coords_norm[:, 0], coords_norm[:, 1], coords_norm[:, 2], 
           color='crimson', s=40, edgecolors='black')

# Set Axis Limits and Labels
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X (Meters)')
ax.set_ylabel('Y (Meters)')
ax.set_zlabel('Z (Meters)')
ax.set_title('Normalized BVH Skeleton (Meters)')

plt.show()