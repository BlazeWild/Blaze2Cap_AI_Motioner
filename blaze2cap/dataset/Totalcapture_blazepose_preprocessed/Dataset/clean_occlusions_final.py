import os
import numpy as np
from pathlib import Path
import sys

def process_directory(src_root, dst_root):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    
    if not src_root.exists():
        print(f"Source directory {src_root} does not exist!")
        return

    print(f"Scanning {src_root}...")
    files = list(src_root.rglob("*.npy"))
    total_files = len(files)
    print(f"Found {total_files} .npy files in {src_root}")
    
    processed_count = 0
    
    for i, src_file in enumerate(files):
        rel_path = src_file.relative_to(src_root)
        dst_file = dst_root / rel_path
        
        filename = src_file.name
        
        # Determine crop amount based on s1/s2/s3 in filename
        crop = 0
        if "_s1_" in filename:
            crop = 30
        elif "_s2_" in filename:
            crop = 20
        elif "_s3_" in filename:
            crop = 15
        
        # Proceed regardless of file length
        try:
            data = np.load(src_file)
            
            # Slice the data
            if crop > 0:
                cleaned_data = data[:-crop]
            else:
                cleaned_data = data
                
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(dst_file, cleaned_data)
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {src_file}: {e}")
            
        if (i + 1) % 500 == 0:
            print(f"Progress: scanned {i + 1}/{total_files} files...")

    print(f"Finished {src_root.name} -> {dst_root.name}")
    print(f"  Total Processed & Saved: {processed_count}")

if __name__ == "__main__":
    # Current directory where script is located
    base_dir = Path(__file__).parent.resolve()
    
    # Define source and destination paths
    src_gt = base_dir / "gt_augmented"
    dst_gt = base_dir / "gtfinal"
    
    src_blaze = base_dir / "blazepose_augmented"
    dst_blaze = base_dir / "blazefinal"
    
    print("Starting processing...")
    
    print("-" * 50)
    print("Processing gt_augmented -> gtfinal")
    process_directory(src_gt, dst_gt)
    
    print("-" * 50)
    print("Processing blazepose_augmented -> blazefinal")
    process_directory(src_blaze, dst_blaze)
    
    print("-" * 50)
    print("Done.")
