import os
import json

# ================= CONFIGURATION =================
# 1. Get the directory where THIS script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Define paths relative to the script
# Point to new dataset location: blaze2cap/dataset/Totalcapture_blazepose_preprocessed/Dataset
dataset_root = os.path.join(script_dir, "..", "dataset", "Totalcapture_blazepose_preprocessed", "Dataset")

# 3. Output file (saved in the same dataset folder)
output_dir = os.path.join(script_dir, "..", "dataset", "Totalcapture_blazepose_preprocessed", "Dataset")
os.makedirs(output_dir, exist_ok=True)
output_json = os.path.join(output_dir, "dataset_map.json")

# 4. Load cam5 exclusion list
cam5_exclude_file = os.path.join(dataset_root, "gardbage_cam5.txt")
cam5_exclude = {}  # {subject: [actions]}

if os.path.exists(cam5_exclude_file):
    with open(cam5_exclude_file, 'r') as f:
        current_subject = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('S') and line[1:].isdigit():
                current_subject = line
                cam5_exclude[current_subject] = []
            elif current_subject:
                cam5_exclude[current_subject].append(line.lower())

# 5. Dataset Split Logic
# Train: S1,S2,S3 on ROM1,2,3; Walking1,3; Freestyle1,2; Acting1,2
TRAIN_SEQUENCES = {
    'rom1', 'rom2', 'rom3',
    'walking1', 'walking3',
    'freestyle1', 'freestyle2',
    'acting1', 'acting2'
}
TRAIN_SUBJECTS = {'S1', 'S2', 'S3'}

# Test: S1,S2,S3,S4,S5 on Walking2, Freestyle3, Acting3
TEST_SEQUENCES = {'walking2', 'freestyle3', 'acting3'}
TEST_SUBJECTS = {'S1', 'S2', 'S3', 'S4', 'S5'}
# =================================================

data_list = []

blaze_root = os.path.join(dataset_root, 'blaze_augmented')
gt_root = os.path.join(dataset_root, 'gt_augmented')

print(f"Scanning dataset at: {os.path.relpath(dataset_root, script_dir)}")
print(f"Loaded cam5 exclusions for {len(cam5_exclude)} subjects")

if not os.path.exists(blaze_root):
    print(f"ERROR: Could not find folder: {blaze_root}")
    print("Make sure you run this script from the project root (Blaze2Cap).")
    exit()

excluded_count = 0

# Walk through BlazePose folder
for root, dirs, files in os.walk(blaze_root):
    for filename in files:
        if filename.endswith('.npy') and filename.startswith("blaze_"):
            
            # --- 1. Extract metadata from path and filename ---
            path_parts = root.split(os.sep)
            subject = None
            action = None
            
            # Find subject (S1, S2, etc.)
            for part in path_parts:
                if part.startswith("S") and part[1:].isdigit() and len(part) < 4: 
                    subject = part
                # Find action (acting1, walking2, etc.)
                if any(seq in part.lower() for seq in ['rom', 'walking', 'freestyle', 'acting']):
                    action = part.lower()
            
            if subject is None or action is None:
                continue
            
            # Extract camera ID from filename or path
            cam_id = None
            if 'cam' in filename:
                # Extract cam number (e.g., cam5 from blaze_S1_acting1_cam5_seg0_s1_o0.npy)
                import re
                cam_match = re.search(r'cam(\d+)', filename)
                if cam_match:
                    cam_id = cam_match.group(1)
            
            # --- 2. Apply cam5 exclusion filter ---
            if cam_id == '5':
                if subject in cam5_exclude and action in cam5_exclude[subject]:
                    excluded_count += 1
                    continue  # Skip this file
            
            # --- 3. Match GT File ---
            # Replace "blaze_" with "gt_" for augmented GT files
            suffix = filename.replace("blaze_", "")
            gt_filename = "gt_" + suffix
            
            # Get the internal folder structure (e.g., "S1/acting1/cam5")
            rel_dir = os.path.relpath(root, blaze_root)
            
            source_path = os.path.join(root, filename)
            target_path = os.path.join(gt_root, rel_dir, gt_filename)
            
            if not os.path.exists(target_path):
                # print(f"Warning: Missing GT for {filename}")
                continue

            # --- 4. Determine Splits based on sequence ---
            is_train = (subject in TRAIN_SUBJECTS and action in TRAIN_SEQUENCES)
            is_test = (subject in TEST_SUBJECTS and action in TEST_SEQUENCES)
            
            # --- 5. Store PORTABLE Relative Paths ---
            entry = {
                "source": os.path.relpath(source_path, dataset_root).replace("\\", "/"),
                "target": os.path.relpath(target_path, dataset_root).replace("\\", "/"),
                "subject": subject,
                "action": action,
                "camera": cam_id if cam_id else "unknown",
                "split_train": is_train,
                "split_test": is_test
            }
            data_list.append(entry)

# Sort data_list for consistent ordering
# Sort by: train/test flag, subject, action, camera, then filename
data_list.sort(key=lambda x: (
    not x['split_train'],  # Train first
    not x['split_test'],   # Then test
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
test_count = sum(1 for d in data_list if d['split_test'])

print()
print("="*60)
print(f"Success! Generated dataset_map.json")
print("="*60)
print(f"Total samples: {len(data_list)}")
print(f"Train samples: {train_count}")
print(f"Test samples: {test_count}")
print(f"Excluded cam5 samples: {excluded_count}")
print()
print(f"File location: {os.path.relpath(output_json, script_dir)}")
print("="*60)