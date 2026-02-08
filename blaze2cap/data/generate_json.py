import os
import json
import re

# ================= CONFIGURATION =================
# 1. Get the directory where THIS script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Define paths relative to the script
# Point to dataset location: blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset
dataset_root = os.path.join(script_dir, "..", "dataset", "Totalcapture_blazepose_preprocessed", "Dataset")

# 3. Output file (saved in the same dataset folder)
output_dir = dataset_root
os.makedirs(output_dir, exist_ok=True)
output_json = os.path.join(output_dir, "dataset_map.json")

# 4. Load cam5 exclusion list
# Expected format in txt: "S1" line followed by "acting1", "rom2", etc.
cam5_exclude_file = os.path.join(dataset_root, "..", "garbagecam5.txt")
cam5_exclude_map = {}  # {subject: {set of actions}}

if os.path.exists(cam5_exclude_file):
    print(f"Loading exclusions from {cam5_exclude_file}...")
    with open(cam5_exclude_file, 'r') as f:
        current_subject = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Check for subject line (e.g., "S1", "S5")
            if line.upper().startswith('S') and line[1:].isdigit():
                current_subject = line.upper()
                cam5_exclude_map[current_subject] = set()
            elif current_subject:
                # Add action to the current subject's set
                cam5_exclude_map[current_subject].add(line.lower())

# ================= SPLIT LOGIC =================
# 1. Training Set (Core Knowledge)
# Subjects: S1, S2, S3
# Sequences: ROM 1-3, Walking 1, Freestyle 1, Acting 1
TRAIN_SUBJECTS = {'S1', 'S2', 'S3'}
TRAIN_ACTIONS = {'rom1', 'rom2', 'rom3', 'walking1', 'freestyle1', 'acting1'}

# 2. Validation Set (Progress Check)
# Subjects: S1, S2, S3
# Sequences: Walking 3, Freestyle 2, Acting 2
VAL_SUBJECTS = {'S1', 'S2', 'S3'}
VAL_ACTIONS = {'walking3', 'freestyle2', 'acting2'}

# 3. Test Set (Final Exam)
# Subjects: S1, S2, S3, S4, S5 (Unseen subjects + Seen subjects on new tasks)
# Sequences: Walking 2, Freestyle 3, Acting 3
TEST_SUBJECTS = {'S1', 'S2', 'S3', 'S4', 'S5'}
TEST_ACTIONS = {'walking2', 'freestyle3', 'acting3'}

# =================================================

data_list = []

# Input/Output Folders
blaze_root = os.path.join(dataset_root, 'blazefinal')
gt_root = os.path.join(dataset_root, 'gtfinal')

print(f"Scanning dataset at: {os.path.abspath(dataset_root)}")

if not os.path.exists(blaze_root):
    print(f"ERROR: Could not find folder: {blaze_root}")
    exit()

excluded_count = 0

# Walk through BlazePose folder
for root, dirs, files in os.walk(blaze_root):
    for filename in files:
        if filename.endswith('.npy') and filename.startswith("blaze_"):
            
            # --- 1. Extract metadata from path or filename ---
            # Filename format: blaze_S1_acting1_cam1_seg0...npy
            parts = filename.split('_')
            
            # Safe extraction
            if len(parts) < 4:
                continue
                
            subject = parts[1]  # S1
            action = parts[2]   # acting1
            cam_str = parts[3]  # cam1
            
            # Normalize strings
            subject = subject.upper()
            action = action.lower()
            
            # Extract camera ID
            cam_match = re.search(r'cam(\d+)', cam_str)
            cam_id = cam_match.group(1) if cam_match else None
            
            # --- 2. Apply cam5 exclusion filter ---
            if cam_id == '5':
                if subject in cam5_exclude_map and action in cam5_exclude_map[subject]:
                    excluded_count += 1
                    continue  # Skip this file
            
            # --- 3. Match GT File ---
            # Construct relative path from blaze_root to find equivalent in gt_root
            # Example: root might be .../blazefinal/S1/acting1/cam1
            rel_dir = os.path.relpath(root, blaze_root)
            
            # Target filename: gt_S1_acting1_cam1_seg0...npy
            gt_filename = filename.replace("blaze_", "gt_", 1)
            
            source_full_path = os.path.join(root, filename)
            target_full_path = os.path.join(gt_root, rel_dir, gt_filename)
            
            if not os.path.exists(target_full_path):
                # Fallback: sometimes GT might be flat or organized differently?
                # Assuming rigorous structure matching for now.
                continue

            # --- 4. Assign Splits ---
            is_train = (subject in TRAIN_SUBJECTS and action in TRAIN_ACTIONS)
            is_val = (subject in VAL_SUBJECTS and action in VAL_ACTIONS)
            is_test = (subject in TEST_SUBJECTS and action in TEST_ACTIONS)
            
            # Skip if it doesn't fit any split (e.g. S4 on Train actions)
            if not (is_train or is_val or is_test):
                continue

            # --- 5. Store PORTABLE Relative Paths ---
            # Store paths relative to 'dataset_root' so JSON is portable
            entry = {
                "source": os.path.relpath(source_full_path, dataset_root).replace("\\", "/"),
                "target": os.path.relpath(target_full_path, dataset_root).replace("\\", "/"),
                "subject": subject,
                "action": action,
                "camera": cam_id if cam_id else "unknown",
                "split_train": is_train,
                "split_val": is_val,
                "split_test": is_test
            }
            data_list.append(entry)

# Sort for consistency
data_list.sort(key=lambda x: (
    not x['split_train'],
    not x['split_val'],
    not x['split_test'],
    x['subject'],
    x['action'],
    x['camera'],
    x['source']
))

# Save to JSON
with open(output_json, 'w') as f:
    json.dump(data_list, f, indent=4)

# Print Statistics
train_count = sum(1 for d in data_list if d['split_train'])
val_count = sum(1 for d in data_list if d['split_val'])
test_count = sum(1 for d in data_list if d['split_test'])

print()
print("="*60)
print(f"Success! Generated dataset_map.json")
print("="*60)
print(f"Total valid samples: {len(data_list)}")
print(f"Training samples:   {train_count}")
print(f"Validation samples: {val_count}")
print(f"Testing samples:    {test_count}")
print(f"Excluded cam5:      {excluded_count}")
print()
print(f"Output: {os.path.abspath(output_json)}")
print("="*60)