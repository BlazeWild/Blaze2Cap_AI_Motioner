import sys
import os
import numpy as np
import traceback

# Add project root to path
sys.path.append(os.getcwd())

try:
    from blaze2cap.modules.pose_processing import process_blazepose_frames
except ImportError:
    print("Could not import blaze2cap. Make sure python path is correct.")
    sys.exit(1)

def test_empty_input():
    print("Testing with 0-length input data...")
    # Simulate empty data (0 frames)
    # shape (Frames, 25, 7)
    empty_data = np.zeros((0, 25, 7), dtype=np.float32)
    
    try:
        process_blazepose_frames(empty_data, window_size=64)
        print("Did not crash (Unexpected).")
    except ValueError as e:
        print(f"Caught Expected ValueError: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"Caught Unexpected Exception: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_empty_input()
