import matplotlib.pyplot as plt
import numpy as np

# 1. Raw Data Input (Joint: [X, Y, Z])
# These values are taken from your provided data string
data = {
    'Hips': [0.963698, 33.2809, 5.1053],
    'Spine': [0.435527, 35.5793, 7.38064],
    'Spine1': [0.325905, 39.1323, 8.14296],
    'Spine2': [0.33256, 42.7588, 8.39943],
    'Spine3': [0.442285, 46.3919, 8.32215],
    'Neck': [0.911774, 55.9101, 6.86796],
    'Head': [1.46578, 60.8559, 6.398],
    'RightShoulder': [1.91472, 52.5147, 6.3718],
    'RightArm': [6.12078, 55.8332, 8.33255],
    'RightForeArm': [17.0487, 54.0893, 10.9552],
    'RightHand': [25.6788, 54.1245, 11.4737],
    'LeftShoulder': [-0.379736, 52.6208, 6.32594],
    'LeftArm': [-4.22758, 56.8284, 6.52241],
    'LeftForeArm': [-15.4274, 56.7149, 8.49493],
    'LeftHand': [-24.0023, 57.1511, 9.51012],
    'RightUpLeg': [4.39126, 32.4618, 5.5353],
    'RightLeg': [3.95157, 17.5576, 5.10216],
    'RightFoot': [3.37795, 3.21243, 8.61846],
    'LeftUpLeg': [-2.30312, 32.1524, 4.29398],
    'LeftLeg': [-3.08002, 17.2556, 4.27177],
    'LeftFoot': [-3.69969, 3.0664, 8.36563],
}

# 2. Define the skeleton connections (Bones)
bones = [
    ('Hips', 'Spine'), ('Spine', 'Spine1'), ('Spine1', 'Spine2'), ('Spine2', 'Spine3'),
    ('Spine3', 'Neck'), ('Neck', 'Head'),
    ('Spine3', 'RightShoulder'), ('RightShoulder', 'RightArm'), ('RightArm', 'RightForeArm'), ('RightForeArm', 'RightHand'),
    ('Spine3', 'LeftShoulder'), ('LeftShoulder', 'LeftArm'), ('LeftArm', 'LeftForeArm'), ('LeftForeArm', 'LeftHand'),
    ('Hips', 'RightUpLeg'), ('RightUpLeg', 'RightLeg'), ('RightLeg', 'RightFoot'),
    ('Hips', 'LeftUpLeg'), ('LeftUpLeg', 'LeftLeg'), ('LeftLeg', 'LeftFoot')
]

# Convert to Meters and NumPy array
coords_world = np.array(list(data.values())) * 0.0254
joint_names = list(data.keys())

# Calibration Params
R = np.array([
    [-0.99713, 0.00504186, -0.0755413],
    [0.0221672, -0.93461, -0.354982],
    [-0.0723915, -0.355637, 0.931816]
])
T = np.array([0.820506, 0.59704, 5.33591])

# Transform World -> Camera
# P_cam = R @ P_world + T
coords_cam = []
for p in coords_world:
    p_c = R @ p + T
    coords_cam.append(p_c)
coords_cam = np.array(coords_cam)

# 3. Normalize to [-1, 1] relative to Hips
# Find Hips index
hips_idx = joint_names.index('Hips')
centroid = coords_cam[hips_idx].copy() # Center on Hips
coords_norm = (coords_cam - centroid)

# Scale while keeping aspect ratio
scale_factor = np.abs(coords_norm).max()
coords_norm /= scale_factor

# Create a mapping for easy lookup
pos_dict = {name: coords_norm[i] for i, name in enumerate(joint_names)}

# 4. Plotting
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot Bones
for start_joint, end_joint in bones:
    p1 = pos_dict[start_joint]
    p2 = pos_dict[end_joint]
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'b-', linewidth=2)

# Plot Joints
ax.scatter(coords_norm[:, 0], coords_norm[:, 1], coords_norm[:, 2], c='r', s=30)

# Formatting
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=20, azim=45) # Adjust view angle to see the pose clearly
plt.title("Skeleton Plot (Absolute Coordinates Normalized)")

plt.show()