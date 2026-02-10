import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import math

# --- CONFIGURATION ---
INFERENCE_OUTPUT_DIR = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/test/inference_test"
INPUT_FILENAME = "pred_blazepose_S1_acting1_cam1_seg0_s1_o0.npy" # The file created by the inference script
INPUT_PATH = os.path.join(INFERENCE_OUTPUT_DIR, INPUT_FILENAME)

# --- SKELETON CONFIGURATION (Z-Up) ---
PARENTS = [-1, 0, 1, 2, 3, 4, 5, 6, 5, 8, 9, 10, 5, 12, 13, 14, 1, 16, 17, 1, 19, 20]

OFFSETS = torch.tensor([
    [ 0.0,      0.0,      0.0     ],  # 0: Hips_pos
    [ 0.0,      0.0,      0.0     ],  # 1: Hips_rot
    [ 0.0,     -0.0693,   0.0462  ],  # 2: Spine
    [ 0.0,     -0.0909,   0.0160  ],  # 3: Spine1
    [ 0.0,     -0.0920,   0.0080  ],  # 4: Spine2
    [ 0.0,     -0.0923,  -0.0     ],  # 5: Spine3
    [ 0.0,     -0.2424,  -0.0346  ],  # 6: Neck
    [ 0.0,     -0.1250,  -0.0220  ],  # 7: Head
    [-0.0292,  -0.1576,  -0.0486  ],  # 8: RightShoulder
    [-0.1449,  -0.0,     -0.0     ],  # 9: RightArm
    [-0.2889,  -0.0,     -0.0     ],  # 10: RightForeArm
    [-0.2196,  -0.0,     -0.0     ],  # 11: RightHand
    [ 0.0292,  -0.1576,  -0.0486  ],  # 12: LeftShoulder
    [ 0.1449,  -0.0,     -0.0     ],  # 13: LeftArm
    [ 0.2889,  -0.0,     -0.0     ],  # 14: LeftForeArm
    [ 0.2196,  -0.0,     -0.0     ],  # 15: LeftHand
    [-0.0866,   0.0253,  -0.0     ],  # 16: RightUpLeg
    [ 0.0,      0.3789,  -0.0     ],  # 17: RightLeg
    [ 0.0,      0.3754,  -0.0     ],  # 18: RightFoot
    [ 0.0866,   0.0253,  -0.0     ],  # 19: LeftUpLeg
    [ 0.0,      0.3789,  -0.0     ],  # 20: LeftLeg
    [ 0.0,      0.3754,  -0.0     ],  # 21: LeftFoot
], dtype=torch.float32)

# --- MATH HELPER FUNCTIONS ---
def rotation_matrix_to_euler(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    if sy < 1e-6:
        x, y, z = math.atan2(-R[1, 2], R[1, 1]), math.atan2(-R[2, 0], sy), 0
    else:
        x, y, z = math.atan2(R[2, 1], R[2, 2]), math.atan2(-R[2, 0], sy), math.atan2(R[1, 0], R[0, 0])
    return np.degrees([x, y, z])

def cont6d_to_mat(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = a2 - (b1 * torch.sum(b1 * a2, dim=-1, keepdim=True))
    b2 = F.normalize(b2, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)

def run_fk_raw(data_npy):
    data = torch.from_numpy(data_npy).float()
    F_frames = data.shape[0]
    
    # Optional reshape if flattened
    if data.dim() == 2 and data.shape[-1] == 132: 
        data = data.view(F_frames, 22, 6)

    # --- 1. GET DATA (RAW) ---
    root_vel = data[:, 0, :3] # Index 0: Root Velocity (first 3 channels)
    rot_data = data[:, 1:, :] # Index 1-21: Rotations (Joint 1 + 20 Body Joints)
    
    rot_mats = cont6d_to_mat(rot_data)
    root_deltas = rot_mats[:, 0]  # Corresponds to index 1 of data (Joint 1)
    body_rots = rot_mats[:, 1:]   # Corresponds to index 2-21 of data (Joints 2-21)

    # --- 2. ROOT LOOP (ACCUMULATE MOTION) ---
    curr_pos = torch.zeros(3)
    curr_rot = torch.eye(3)
    
    root_pos_list = []
    root_rot_list = []
    
    print("Computing Forward Kinematics (Root)...")
    for f in range(F_frames):
        # 1. Rotate Velocity from Local to World using current facing
        #    P_new = P_old + (R_old @ V_local)
        step = torch.matmul(curr_rot, root_vel[f])
        curr_pos += step
        
        # 2. Update Rotation
        #    R_new = R_old @ R_delta
        curr_rot = torch.matmul(curr_rot, root_deltas[f])
        
        root_pos_list.append(curr_pos.clone())
        root_rot_list.append(curr_rot.clone())

    # --- 3. BODY LOOP ---
    print("Computing Forward Kinematics (Body)...")
    all_poses = []
    for f in range(F_frames):
        g_pos = [root_pos_list[f], root_pos_list[f]]
        g_rot = [root_rot_list[f], root_rot_list[f]]
        
        for i in range(2, 22):
            pid = PARENTS[i]
            off = OFFSETS[i]
            
            p_rot = g_rot[pid]
            p_pos = g_pos[pid]
            l_rot = body_rots[f, i-2]
            
            curr_rot = torch.matmul(p_rot, l_rot)
            curr_pos = p_pos + torch.matmul(p_rot, off)
            
            g_rot.append(curr_rot)
            g_pos.append(curr_pos)
            
        all_poses.append(torch.stack(g_pos))
        
    return (torch.stack(all_poses).numpy(), 
            np.stack(root_rot_list), 
            root_deltas.numpy(),
            root_vel.numpy())

# --- MAIN PLOTTING LOOP ---
def main():
    print(f"Loading prediction: {INPUT_PATH}")
    if not os.path.exists(INPUT_PATH):
        print("File not found! Run the inference script first.")
        return

    predictions = np.load(INPUT_PATH)
    print(f"Data Shape: {predictions.shape}")
    
    # 1. Run FK
    print("--- COMPUTING FK ---")
    pos, glb_rot, del_rot, vel = run_fk_raw(predictions)
    
    print("Launching Plotter...")
    # Matplotlib Configuration
    try:
        import matplotlib
        matplotlib.use('TkAgg') # Try to use interactive backend
    except:
        pass
        
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Info Text
    info = fig.text(0.02, 0.5, "", fontfamily='monospace', fontsize=10, verticalalignment='center')
    
    # Elements
    scat = ax.scatter([],[],[], c='r', s=15)
    lines = [ax.plot([],[],[], 'b-')[0] for _ in range(21)]
    traj, = ax.plot([],[],[], 'g--', alpha=0.5) 
    
    # Bounds
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-1, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    slider_ax = plt.axes([0.3, 0.05, 0.6, 0.03])
    slider = Slider(slider_ax, 'Frame', 0, len(pos)-1, valinit=0, valstep=1)
    
    def update(val):
        f = int(slider.val)
        p = pos[f]
        
        # Plot Body (Scatter)
        scat._offsets3d = (p[:,0], p[:,1], p[:,2])
        
        # Plot Bones
        for i, line in enumerate(lines):
            # i is 0..20 corresponds to bones connecting to children 1..21
            child_idx = i + 1  
            parent_idx = PARENTS[child_idx]
            
            p1 = p[parent_idx]
            p2 = p[child_idx]
            
            line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
            line.set_3d_properties([p1[2], p2[2]])

        # Plot Trajectory (History of Root)
        path = pos[:f+1, 0, :] 
        traj.set_data(path[:, 0], path[:, 1])
        traj.set_3d_properties(path[:, 2])

        # Update Info
        gr, gp, gy = rotation_matrix_to_euler(glb_rot[f])
        v = vel[f]

        info.set_text(
            f"Frame: {f}\n\n"
            f"ROOT POS (Z-Up):\n"
            f"  X: {p[0,0]:.3f}\n"
            f"  Y: {p[0,1]:.3f}\n"
            f"  Z: {p[0,2]:.3f}\n\n"
            f"ROOT VELOCITY:\n"
            f"  X: {v[0]:.3f}\n"
            f"  Y: {v[1]:.3f}\n"
            f"  Z: {v[2]:.3f}\n\n"
            f"HIP ORI (Global):\n"
            f"  R: {gr:.1f}\n  P: {gp:.1f}\n  Y: {gy:.1f}"
        )
        fig.canvas.draw_idle()
        
    slider.on_changed(update)
    update(0) # Initial draw
    plt.show()

if __name__ == "__main__":
    main()
