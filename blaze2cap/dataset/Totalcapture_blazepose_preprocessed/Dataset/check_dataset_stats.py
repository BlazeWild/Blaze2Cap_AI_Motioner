import os
import numpy as np
from pathlib import Path

# Define Splits
TRAIN_SUBJECTS = {'S1', 'S2', 'S3'}
TRAIN_ACTIONS = {'rom1', 'rom2', 'rom3', 'walking1', 'freestyle1', 'acting1'}

VAL_SUBJECTS = {'S1', 'S2', 'S3'}
VAL_ACTIONS = {'walking3', 'freestyle2', 'acting2'}

TEST_SUBJECTS = {'S1', 'S2', 'S3', 'S4', 'S5'}
TEST_ACTIONS = {'walking2', 'freestyle3', 'acting3'}

def analyze_dataset(dataset_name, root_path):
    root = Path(root_path)
    if not root.exists():
        print(f"Dataset {dataset_name} not found at {root}")
        return

    print(f"\nAnalyzing {dataset_name} at {root}...")
    
    total_frames = 0
    empty_files = []
    
    train_frames = 0
    val_frames = 0
    test_frames = 0
    
    # Track files to ensure we don't count duplicates or miss anything
    count_train = 0
    count_val = 0
    count_test = 0
    count_other = 0
    other_details = set() # Store unique (Subject, Action) tuples for other

    files = list(root.rglob("*.npy"))
    print(f"Found {len(files)} files.")

    for i, file_path in enumerate(files):
        try:
            # Determine frames
            data = np.load(file_path)
            num_frames = data.shape[0]
            
            if num_frames == 0:
                empty_files.append(str(file_path))
            
            total_frames += num_frames
            
            # Determine split
            # Path structure expected: root / Subject / Action / Camera / filename.npy
            # relative path: Subject / Action / Camera / filename.npy
            try:
                rel_parts = file_path.relative_to(root).parts
                if len(rel_parts) >= 2:
                    subject = rel_parts[0]
                    action = rel_parts[1]
                    
                    # Check splits
                    if subject in TRAIN_SUBJECTS and action in TRAIN_ACTIONS:
                        train_frames += num_frames
                        count_train += 1
                    elif subject in VAL_SUBJECTS and action in VAL_ACTIONS:
                        val_frames += num_frames
                        count_val += 1
                    elif subject in TEST_SUBJECTS and action in TEST_ACTIONS:
                        test_frames += num_frames
                        count_test += 1
                    else:
                        count_other += 1
                        other_details.add((subject, action))
            except Exception as e:
                print(f"Error parsing path {file_path}: {e}")

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} files...")

    print("-" * 30)
    print(f"Results for {dataset_name}:")
    print(f"Total Files: {len(files)}")
    print(f"Total Frames: {total_frames}")
    
    if empty_files:
        print(f"WARNING: Found {len(empty_files)} empty files:")
        for f in empty_files[:10]:
            print(f"  {f}")
        if len(empty_files) > 10:
            print(f"  ... and {len(empty_files) - 10} more.")
    else:
        print("No empty files found.")
        
    print("-" * 30)
    print("Split Statistics (Frames):")
    print(f"  Train:      {train_frames:>10} frames (from {count_train} files)")
    print(f"  Validation: {val_frames:>10} frames (from {count_val} files)")
    print(f"  Test:       {test_frames:>10} frames (from {count_test} files)")
    print(f"  Other:      {count_other} files (not matching defined splits)")
    
    if count_other > 0:
        print("\n  'Other' includes the following Subject/Action combinations:")
        for s, a in sorted(list(other_details)):
            print(f"    - Subject: {s}, Action: {a}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    
    gt_path = base_dir / "gtfinal"
    blaze_path = base_dir / "blazefinal"
    
    analyze_dataset("GT Final", gt_path)
    analyze_dataset("Blaze Final", blaze_path)
