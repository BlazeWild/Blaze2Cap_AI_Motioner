import os
import json
import numpy as np
from tqdm import tqdm

DATASET_ROOT = r"C:\hav_video_captioning\Blaze2Cap\blaze2cap\dataset\Totalcapture_blazepose_preprocessed\Dataset"
JSON_PATH = os.path.join(DATASET_ROOT, "dataset_map.json")

def check_dataset():
    if not os.path.exists(JSON_PATH):
        print(f"Error: dataset_map.json not found at {JSON_PATH}")
        return

    print(f"Loading map from {JSON_PATH}...")
    with open(JSON_PATH, 'r') as f:
        data_map = json.load(f)
    
    print(f"Found {len(data_map)} samples. Checking for empty files...")
    
    empty_count = 0
    missing_count = 0
    
    for item in tqdm(data_map):
        source_rel = item['source']
        target_rel = item['target']
        
        source_path = os.path.join(DATASET_ROOT, source_rel)
        target_path = os.path.join(DATASET_ROOT, target_rel)
        
        # Check Source
        if not os.path.exists(source_path):
            print(f"[MISSING] Source: {source_path}")
            missing_count += 1
        else:
            try:
                data = np.load(source_path)
                if data.size == 0 or data.shape[0] == 0:
                    print(f"[EMPTY] Source ({data.shape}): {source_rel}")
                    empty_count += 1
            except Exception as e:
                print(f"[ERROR] Could not load {source_rel}: {e}")
                empty_count += 1

        # Check Target
        if not os.path.exists(target_path):
            print(f"[MISSING] Target: {target_path}")
            missing_count += 1
        else:
            try:
                data = np.load(target_path)
                if data.size == 0 or data.shape[0] == 0:
                    print(f"[EMPTY] Target ({data.shape}): {target_rel}")
                    empty_count += 1
            except Exception as e:
                print(f"[ERROR] Could not load {target_rel}: {e}")
                empty_count += 1

    print("-" * 30)
    print(f"Scan complete.")
    print(f"Empty/Corrupt Files: {empty_count}")
    print(f"Missing Files: {missing_count}")

if __name__ == "__main__":
    check_dataset()
