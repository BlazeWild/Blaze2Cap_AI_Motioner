import os
import json
import numpy as np
from tqdm import tqdm

DATASET_ROOT = r"C:\hav_video_captioning\Blaze2Cap\blaze2cap\dataset\Totalcapture_blazepose_preprocessed\Dataset"
JSON_PATH = os.path.join(DATASET_ROOT, "dataset_map.json")

def check_val_sizes():
    with open(JSON_PATH, 'r') as f:
        data_map = json.load(f)
    
    val_samples = [item for item in data_map if item.get("split_val", False)]
    print(f"Checking {len(val_samples)} validation samples...")
    
    max_len = 0
    max_file = ""
    sizes = []
    
    for item in tqdm(val_samples):
        source_path = os.path.join(DATASET_ROOT, item["source"])
        if not os.path.exists(source_path):
            continue
            
        try:
            # Check length without loading full file if possible, but shape is needed
             # mmap_mode='r' reads header
            data = np.load(source_path, mmap_mode='r')
            length = data.shape[0]
            sizes.append(length)
            
            if length > max_len:
                max_len = length
                max_file = item["source"]
        except Exception:
            pass
            
    if sizes:
        print(f"Max Length: {max_len} frames")
        print(f"Max File: {max_file}")
        print(f"Average Length: {sum(sizes)/len(sizes):.1f}")
        print(f"Percentiles: 90th={np.percentile(sizes, 90)}, 99th={np.percentile(sizes, 99)}")
        
        # Estimate batch size Windows
        # Window size 64. 1 Frame shift approx? Strided windows.
        # If stride is 1 (usually), NumWindows ~= Length.
        # Batch size 4.
        p99 = np.percentile(sizes, 99)
        est_batch_windows = p99 * 4
        print(f"Estimated 99th percentile Batch Windows (Batch=4): {est_batch_windows}")
    else:
        print("No valid validation samples found.")

if __name__ == "__main__":
    check_val_sizes()
