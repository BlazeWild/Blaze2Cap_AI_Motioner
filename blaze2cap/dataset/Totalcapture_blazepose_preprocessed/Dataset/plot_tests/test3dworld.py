import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_DIR = '/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/___MOTION_T_LIGHTNING/Blaze2Cap/blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset/blazepose_final'

# --- INDICES (From your table) ---
I_NOSE = 0
I_L_EAR = 1;   I_R_EAR = 2
I_L_SH = 3;    I_R_SH = 4
I_L_ELB = 5;   I_R_ELB = 6
I_L_WRIST = 7; I_R_WRIST = 8
I_L_PINKY = 9; I_R_PINKY = 10
I_L_INDEX = 11; I_R_INDEX = 12
I_L_THUMB = 13; I_R_THUMB = 14

I_L_HIP = 15;  I_R_HIP = 16
I_L_KNEE = 17; I_R_KNEE = 18
I_L_ANKLE = 19; I_R_ANKLE = 20
I_L_HEEL = 21; I_R_HEEL = 22
I_L_FOOT = 23; I_R_FOOT = 24

# Virtual
I_NECK = 25
I_MIDHIP = 26

# --- VISUAL CONNECTIONS ---
BONES_VISUAL = [
    # 1. SPINE (MidHip <-> Neck) - ADDED BACK
    (I_MIDHIP, I_NECK),

    # 2. PELVIS HUB
    (I_MIDHIP, I_L_HIP),   
    (I_MIDHIP, I_R_HIP),   
    
    # 3. SHOULDER HUB
    (I_NECK, I_L_SH),      
    (I_NECK, I_R_SH),      
    
    # 4. ARMS (Left)
    (3, 5), (5, 7),         # Sh -> Elb -> Wrist
    (7, 9), (7, 11), (7, 13), # Wrist -> Pinky/Index/Thumb
    
    # 4. ARMS (Right)
    (4, 6), (6, 8),         # Sh -> Elb -> Wrist
    (8, 10), (8, 12), (8, 14), # Wrist -> Pinky/Index/Thumb
    
    # 5. LEGS (Left)
    (15, 17), (17, 19),     # Hip -> Knee -> Ankle
    (19, 23), (19, 21),     # Ankle -> Foot/Heel

    # 5. LEGS (Right)
    (16, 18), (18, 20),     # Hip -> Knee -> Ankle
    (20, 24), (20, 22),     # Ankle -> Foot/Heel
    
    # 6. FACE (Floating)
    (0, 1), (0, 2), # Nose -> Ears
]

# ==========================================
# 2. DATA LOADING
# ==========================================
data_tree = {}
print(f"Scanning {DATA_DIR}...")

for cam_path in Path(DATA_DIR).rglob("cam*"):
    if not cam_path.is_dir(): continue
    action_path = cam_path.parent
    subject_path = action_path.parent
    sub, act, cam = subject_path.name, action_path.name, cam_path.name
    
    if sub not in data_tree: data_tree[sub] = {}
    if act not in data_tree[sub]: data_tree[sub][act] = {}
    
    files = sorted(list(cam_path.glob("*.npy")))
    if files: data_tree[sub][act][cam] = files

if not data_tree:
    print("No data found!")
    exit()

subjects = sorted(list(data_tree.keys()))
state = {'sub_idx': 0, 'act_idx': 0, 'cam_idx': 0, 'seg_idx': 0, 'data': None}

def get_current_context():
    s_idx = state['sub_idx'] % len(subjects)
    sub = subjects[s_idx]
    actions = sorted(list(data_tree[sub].keys()))
    state['act_idx'] %= len(actions)
    act = actions[state['act_idx']]
    cameras = sorted(list(data_tree[sub][act].keys()))
    state['cam_idx'] %= len(cameras)
    cam = cameras[state['cam_idx']]
    segments = data_tree[sub][act][cam]
    state['seg_idx'] %= len(segments)
    fpath = segments[state['seg_idx']]
    return sub, act, cam, fpath

# ==========================================
# 3. PROCESSING
# ==========================================
def load_and_process(fpath):
    print(f"Loading: {fpath.name}")
    raw = np.load(fpath)
    xyz = raw[:, :, 0:3]
        
    # --- VIRTUAL JOINTS ---
    # Mid-Hip (Mean of 15 & 16)
    l_hip = xyz[:, 15, :]
    r_hip = xyz[:, 16, :]
    mid_hip = (l_hip + r_hip) / 2.0
    
    # Neck (Mean of 3 & 4)
    l_sh = xyz[:, 3, :]
    r_sh = xyz[:, 4, :]
    neck = (l_sh + r_sh) / 2.0
    
    neck = neck[:, np.newaxis, :]
    mid_hip = mid_hip[:, np.newaxis, :]
    return np.concatenate([xyz, neck, mid_hip], axis=1)

_, _, _, p = get_current_context()
state['data'] = load_and_process(p)

# ==========================================
# 4. VISUALIZATION
# ==========================================
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.25)

lines = [ax.plot([],[],[], 'b-', lw=2)[0] for _ in BONES_VISUAL]
scatter = ax.scatter([], [], [], c='r', s=20)

def update_plot(val=None):
    frame_idx = int(slider.val)
    data = state['data']
    if frame_idx >= len(data): frame_idx = len(data) - 1
    
    pose = data[frame_idx]
    
    # --- RAW MAPPING (X->X, Y->Y, Z->Z) ---
    xs = pose[:, 0]
    ys = pose[:, 1] # NO FLIPPING. Raw Y data.
    zs = pose[:, 2] 
    
    scatter._offsets3d = (xs, ys, zs)
    
    for line, (p1, p2) in zip(lines, BONES_VISUAL):
        line.set_data_3d(
            [xs[p1], xs[p2]],
            [ys[p1], ys[p2]],
            [zs[p1], zs[p2]]
        )
        
    sub, act, cam, fpath = get_current_context()
    ax.set_title(f"{sub} | {act} | {cam}\n{fpath.name} (Frame {frame_idx})")
    fig.canvas.draw_idle()

def set_static_limits():
    data = state['data']
    # Calculate limits on raw data
    xs = data[:, :, 0].flatten()
    ys = data[:, :, 1].flatten()
    zs = data[:, :, 2].flatten()
    
    margin = 0.2
    if len(xs) > 0:
        ax.set_xlim(xs.min()-margin, xs.max()+margin)
        ax.set_ylim(ys.min()-margin, ys.max()+margin)
        ax.set_zlim(zs.min()-margin, zs.max()+margin)
    
    # Proper Labels matching Data
    ax.set_xlabel('X')
    ax.set_ylabel('Y (Vertical)')
    ax.set_zlabel('Z (Depth)')

set_static_limits()
# Initial view (User can rotate)
ax.view_init(elev=20, azim=-45)

# ==========================================
# 5. CONTROLS
# ==========================================
ax_slider = plt.axes([0.2, 0.15, 0.6, 0.03])
slider = Slider(ax_slider, 'Frame', 0, len(state['data'])-1, valinit=0, valfmt='%d')
slider.on_changed(update_plot)

def reload_all():
    _, _, _, p = get_current_context()
    state['data'] = load_and_process(p)
    slider.valmax = len(state['data']) - 1
    slider.set_val(0)
    set_static_limits()
    update_plot(0)

def next_subject(event):
    state['sub_idx'] += 1; state['act_idx'] = 0; state['cam_idx'] = 0; state['seg_idx'] = 0
    reload_all()

def next_action(event):
    state['act_idx'] += 1; state['cam_idx'] = 0; state['seg_idx'] = 0
    reload_all()

def next_camera(event):
    state['cam_idx'] += 1; state['seg_idx'] = 0
    reload_all()

def next_segment(event):
    state['seg_idx'] += 1
    reload_all()

y = 0.05
# Persistent Buttons
btn_sub = Button(plt.axes([0.05, y, 0.12, 0.05]), 'Subject')
btn_sub.on_clicked(next_subject)

btn_act = Button(plt.axes([0.20, y, 0.12, 0.05]), 'Action')
btn_act.on_clicked(next_action)

btn_cam = Button(plt.axes([0.35, y, 0.12, 0.05]), 'Camera')
btn_cam.on_clicked(next_camera)

btn_seg = Button(plt.axes([0.50, y, 0.12, 0.05]), 'Segment')
btn_seg.on_clicked(next_segment)

update_plot(0)
plt.show()