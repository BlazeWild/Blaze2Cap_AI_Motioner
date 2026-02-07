import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 1. Define the hierarchy: {Joint: (Parent, [Offset X, Y, Z])}
# Note: Values are ALREADY in Meters and transformed to [-X, Z, Y]
hierarchy = {
    'Hips': (None, [0.0, 0.0, 0.0]),
    'Spine': ('Hips', [-0.0, -0.069257, 0.046171]),
    'Spine1': ('Spine', [-0.0, -0.090940, 0.016035]),
    'Spine2': ('Spine1', [-0.0, -0.091991, 0.008048]),
    'Spine3': ('Spine2', [-0.0, -0.092342, 0.0]),
    'Neck': ('Spine3', [-0.000000, -0.242399, -0.034628]),
    'Head': ('Neck', [-0.0, -0.125042, -0.022048]),
    'RightShoulder': ('Spine3', [-0.029176, -0.157553, -0.048628]),
    'RightArm': ('RightShoulder', [-0.144910, 0.0, 0.0]),
    'RightForeArm': ('RightArm', [-0.288868, 0.000000, 0.0]),
    'RightHand': ('RightForeArm', [-0.219603, 0.0, 0.0]), # FIXED PARENT
    'LeftShoulder': ('Spine3', [0.029176, -0.157553, -0.048628]),
    'LeftArm': ('LeftShoulder', [0.144910, 0.0, 0.0]),
    'LeftForeArm': ('LeftArm', [0.288868, -0.000000, 0.0]),
    'LeftHand': ('LeftForeArm', [0.219603, 0.0, 0.0]),
    'RightUpLeg': ('Hips', [-0.086557, 0.025286, 0.0]),
    'RightLeg': ('RightUpLeg', [0.000000, 0.378891, 0.0]),
    'RightFoot': ('RightLeg', [0.000000, 0.375438, 0.0]),
    'LeftUpLeg': ('Hips', [0.086557, 0.025286, 0.0]),
    'LeftLeg': ('LeftUpLeg', [0.000000, 0.378891, 0.0]),
    'LeftFoot': ('LeftLeg', [0.000000, 0.375438, 0.0]),
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
        # Recursive call to parent to find its global position
        pos = get_global_position(parent) + np.array(offset)
    
    abs_pos[name] = pos
    return pos

# Calculate for all joints
for joint in hierarchy:
    get_global_position(joint)

# Extract coordinates as a numpy array
coords = np.array([abs_pos[j] for j in hierarchy])

# 3. Normalize to [-1, 1] range for visualization
centroid = coords.mean(axis=0)
coords_centered = coords - centroid

# Scale while maintaining aspect ratio
max_range = np.abs(coords_centered).max()
coords_norm = coords_centered / max_range

# Map back for plotting
norm_pos_dict = {name: coords_norm[i] for i, name in enumerate(hierarchy.keys())}

# 4. Plotting
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Draw the bones
for name, (parent, _) in hierarchy.items():
    if parent:
        p1 = norm_pos_dict[parent]
        p2 = norm_pos_dict[name]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                color='royalblue', linewidth=3, alpha=0.8)

# Draw the joints
ax.scatter(coords_norm[:, 0], coords_norm[:, 1], coords_norm[:, 2], 
           color='crimson', s=50, edgecolors='black')

# Formatting
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X (Negated)')
ax.set_ylabel('Y (Old Z)')
ax.set_zlabel('Z (Old Y)')
ax.set_title('Metric Skeleton: Transformed [-X, Z, Y]')

# Adjust view for a better perspective
ax.view_init(elev=20, azim=45)

plt.show()