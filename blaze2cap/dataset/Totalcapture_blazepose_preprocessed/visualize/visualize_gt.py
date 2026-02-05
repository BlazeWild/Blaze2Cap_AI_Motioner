"""
Visualize Ground Truth 3D Skeleton Data
Loads .npy files from gt_numpy folder and visualizes skeleton with frame slider
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import random

# Define the 17 keypoints in order (TotalCapture skeleton)
KEYPOINT_NAMES = [
    'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightUpLeg', 'RightLeg', 'RightFoot'
]

# Define skeleton connections (bone pairs) for 17 keypoints
SKELETON_CONNECTIONS = [
    # Spine chain
    (0, 1),   # Hips -> Spine
    (1, 2),   # Spine -> Spine1
    (2, 3),   # Spine1 -> Spine2
    (3, 4),   # Spine2 -> Neck
    (4, 5),   # Neck -> Head
    
    # Right arm
    (4, 6),   # Neck -> RightShoulder
    (6, 7),   # RightShoulder -> RightArm
    (7, 8),   # RightArm -> RightForeArm
    (8, 9),   # RightForeArm -> RightHand
    
    # Left arm
    (4, 10),  # Neck -> LeftShoulder
    (10, 11), # LeftShoulder -> LeftArm
    (11, 12), # LeftArm -> LeftForeArm
    (12, 13), # LeftForeArm -> LeftHand
    
    # Right leg
    (0, 14),  # Hips -> RightUpLeg
    (14, 15), # RightUpLeg -> RightLeg
    (15, 16), # RightLeg -> RightFoot
]


def load_random_gt_file(gt_folder='final_numpy_dataset/gt_numpy'):
    """Load a random .npy file from the gt_numpy folder"""
    # Get all .npy files recursively
    npy_files = glob.glob(os.path.join(gt_folder, '**', '*.npy'), recursive=True)
    
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {gt_folder}")
    
    # Select random file
    selected_file = random.choice(npy_files)
    print(f"Loading: {selected_file}")
    
    # Load data
    data = np.load(selected_file)
    print(f"Data shape: {data.shape}")
    print(f"Expected shape: (num_frames, 17, 3)")
    
    return data, selected_file


def plot_skeleton_3d(ax, keypoints, title=""):
    """Plot skeleton in 3D with connections"""
    ax.clear()
    
    # Extract coordinates
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    z = keypoints[:, 2]
    
    # Plot keypoints
    ax.scatter(x, y, z, c='red', marker='o', s=50, alpha=0.8, label='Keypoints')
    
    # Plot skeleton connections
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            ax.plot(
                [x[start_idx], x[end_idx]],
                [y[start_idx], y[end_idx]],
                [z[start_idx], z[end_idx]],
                'b-', linewidth=2, alpha=0.6
            )
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio for better visualization
    # Get the range of each axis
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add legend
    ax.legend()


def visualize_gt_skeleton(data, filename):
    """
    Visualize ground truth skeleton data with frame slider
    
    Args:
        data: numpy array of shape (num_frames, 17, 3)
        filename: name of the file being visualized
    """
    num_frames = data.shape[0]
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Adjust plot to make room for slider
    plt.subplots_adjust(bottom=0.15)
    
    # Initial frame
    current_frame = 0
    
    # Plot initial skeleton
    plot_skeleton_3d(
        ax, 
        data[current_frame], 
        f"Frame {current_frame + 1}/{num_frames}\n{os.path.basename(filename)}"
    )
    
    # Create slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(
        ax_slider, 
        'Frame', 
        1,  # Start from 1 for display
        num_frames, 
        valinit=1, 
        valstep=1,
        color='lightblue'
    )
    
    # Update function for slider
    def update(val):
        frame_idx = int(slider.val) - 1  # Convert to 0-indexed
        plot_skeleton_3d(
            ax, 
            data[frame_idx], 
            f"Frame {frame_idx + 1}/{num_frames}\n{os.path.basename(filename)}"
        )
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    # Add instructions
    plt.figtext(
        0.5, 0.01, 
        'Use slider to navigate frames | Left mouse: rotate | Right mouse: pan | Scroll: zoom',
        ha='center', 
        fontsize=10, 
        style='italic'
    )
    
    plt.show()


def main():
    """Main function to run the visualization"""
    # Path to gt_numpy folder (relative to script location)
    gt_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'gt_numpy'
    )
    
    # Normalize path
    gt_folder = os.path.normpath(gt_folder)
    
    print(f"Looking for .npy files in: {gt_folder}")
    
    # Check if folder exists
    if not os.path.exists(gt_folder):
        print(f"Error: Folder not found: {gt_folder}")
        print("Please update the gt_folder path in the script.")
        return
    
    # Load random file
    try:
        data, filename = load_random_gt_file(gt_folder)
        
        # Validate data shape
        if len(data.shape) != 3 or data.shape[1] != 17 or data.shape[2] != 3:
            print(f"Warning: Unexpected data shape {data.shape}")
            print("Expected shape: (num_frames, 17, 3)")
            
            # Try to reshape if possible
            if data.size % (17 * 3) == 0:
                num_frames = data.size // (17 * 3)
                data = data.reshape(num_frames, 17, 3)
                print(f"Reshaped to: {data.shape}")
            else:
                print("Cannot reshape data to (num_frames, 17, 3)")
                return
        
        # Visualize
        print(f"\nVisualizing {data.shape[0]} frames...")
        print("Controls:")
        print("  - Use slider to change frames")
        print("  - Left mouse button: rotate view")
        print("  - Right mouse button: pan view")
        print("  - Mouse wheel: zoom in/out")
        
        visualize_gt_skeleton(data, filename)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
