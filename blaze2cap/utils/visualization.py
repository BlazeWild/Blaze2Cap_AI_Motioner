import cv2
import numpy as np
import os

# Define the connections for drawing the skeleton
# Indices based on standard skeleton formats (Modify if TotalCapture differs)
# This generic set works for most 17-keypoint skeletons (COCO format)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Face
    (5, 6), (5, 7), (6, 8), (7, 9),  # Arms
    (8, 10), (9, 11),                # Hands
    (5, 11), (6, 12),                # Body (Torso)
    (11, 12),                        # Hips
    (11, 13), (12, 14),              # Legs
    (13, 15), (14, 16)               # Feet
]

def draw_skeleton_on_canvas(pose, height=512, width=512):
    """
    Draws a single frame of pose data onto a black canvas.
    Args:
        pose: (Joints, 3) or (Joints, 2) numpy array. Assumes values are roughly normalized or in meters.
        height: Canvas height
        width: Canvas width
    Returns:
        image: (H, W, 3) BGR image
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 1. Normalize/Scale Pose to fit Canvas
    # This is a heuristic scaling; you might need to adjust 'scale' and 'shift' 
    # based on your specific data range (e.g., if data is in meters 0.0-2.0)
    scale = 200 
    shift_x = width // 2
    shift_y = height // 2
    
    # Project 3D (x, y, z) -> 2D (x, y) simply by dropping Z for visualization
    # Assuming index 0=X, 1=Y. 
    points_2d = []
    for joint in pose:
        x = int(joint[0] * scale + shift_x)
        y = int(joint[1] * scale + shift_y) # You might need -joint[1] if Y is up
        points_2d.append((x, y))

    # 2. Draw Lines (Bones)
    for p1_idx, p2_idx in SKELETON_CONNECTIONS:
        if p1_idx < len(points_2d) and p2_idx < len(points_2d):
            pt1 = points_2d[p1_idx]
            pt2 = points_2d[p2_idx]
            # Check bounds
            if (0 <= pt1[0] < width and 0 <= pt1[1] < height and 
                0 <= pt2[0] < width and 0 <= pt2[1] < height):
                cv2.line(canvas, pt1, pt2, (0, 255, 0), 2) # Green bones

    # 3. Draw Points (Joints)
    for pt in points_2d:
        if 0 <= pt[0] < width and 0 <= pt[1] < height:
            cv2.circle(canvas, pt, 3, (0, 0, 255), -1) # Red joints

    return canvas

def render_pose_video(save_path, pose_sequence, fps=30):
    """
    Converts a sequence of poses into an MP4 video file.
    Args:
        save_path: Output .mp4 path
        pose_sequence: (Frames, Joints, 3) Numpy array
        fps: Frames per second
    """
    if len(pose_sequence) == 0:
        print("Warning: Empty pose sequence, skipping video generation.")
        return

    height, width = 512, 512
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    print(f"Rendering pose video to {save_path}...")
    
    for i in range(len(pose_sequence)):
        frame_pose = pose_sequence[i]
        
        # Draw the frame
        frame_img = draw_skeleton_on_canvas(frame_pose, height, width)
        
        # Add Frame Number text
        cv2.putText(frame_img, f"Frame: {i}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        
        out.write(frame_img)

    out.release()
    print("Video saved.")

if __name__ == "__main__":
    # --- RELATIVE PATH LOGIC ---
    
    # 1. Get the directory where THIS script (visualization.py) is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Construct path to the dataset relative to 'utils'
    # We go UP one level (..) to 'blaze2cap', then into 'dataset'
    target_rel_path = os.path.join(
        current_dir, 
        "..", "dataset", "Totalcapture_numpy_preprocessed", "gt_numpy", "S1", "acting1", "gt_S1_acting1_cam1.npy"
    )
    
    # Normalize path (fixes slashes for Windows/Linux)
    target_path = os.path.normpath(target_rel_path)

    print(f"Looking for file at: {target_path}")

    if os.path.exists(target_path):
        # Load the data
        data = np.load(target_path)
        
        # Sanity Check: If data is (Frames, 51), reshape to (Frames, 17, 3)
        if len(data.shape) == 2 and data.shape[1] % 3 == 0:
            frames = data.shape[0]
            joints = data.shape[1] // 3
            data = data.reshape(frames, joints, 3)
            
        render_pose_video("viz_S1_acting1.mp4", data, fps=60)
    else:
        print("File not found! Please check your folder structure.")
        print(f"Expected location: {target_path}")