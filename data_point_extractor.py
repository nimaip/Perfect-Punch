import cv2
import mediapipe as mp
import torch
import numpy as np
import os
import csv
import subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def visualize_wrist_trajectory(keypoints, video_path):
    """Visualize the wrist trajectory in 2D (X and Y only) for the current video"""
    # Extract wrist (first 2 coords: x, y)
    wrist_x = keypoints[:, 0]  # x of wrist
    wrist_y = keypoints[:, 1]  # y of wrist

    # Get punch type from video path
    punch_type = os.path.basename(os.path.dirname(video_path))
    video_name = os.path.basename(video_path)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot wrist trajectory
    ax.plot(wrist_x, wrist_y, 'b-o', linewidth=2.5, markersize=6, label='Wrist')

    # Mark start and end
    ax.scatter(wrist_x[0], wrist_y[0],
              c='lime', s=300, marker='*', label='Start',
              edgecolors='darkgreen', linewidths=2, zorder=10)
    ax.scatter(wrist_x[-1], wrist_y[-1],
              c='red', s=300, marker='X', label='End',
              edgecolors='darkred', linewidths=2, zorder=10)

    # Add frame numbers
    for i in range(len(wrist_x)):
        ax.annotate(f'{i}', (wrist_x[i], wrist_y[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.7)

    # Labels
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_title(f'{punch_type}\n{video_name}\n(15 frames)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()  # Block until user closes the window

def convert_mov_to_mp4(mov_path):
    """Convert .mov file to .mp4 using ffmpeg"""
    mp4_path = mov_path.rsplit('.', 1)[0] + '.mp4'

    # Check if mp4 already exists
    if os.path.exists(mp4_path):
        print(f"MP4 already exists: {mp4_path}")
        return mp4_path

    try:
        print(f"Converting {mov_path} to MP4...")
        result = subprocess.run([
            'ffmpeg', '-i', mov_path,
            '-map', '0:v:0',  # map only the first video stream
            '-map', '0:a:0?',  # map the first audio stream if it exists (? makes it optional)
            '-c:v', 'libx264',  # video codec
            '-preset', 'fast',  # encoding speed
            '-crf', '23',  # quality (lower = better, 18-28 is reasonable)
            '-c:a', 'aac',  # audio codec
            '-b:a', '128k',  # audio bitrate
            '-movflags', '+faststart',  # optimize for streaming
            '-y',  # overwrite output file if exists
            mp4_path
        ], check=True, capture_output=True, text=True)
        print(f"Conversion successful: {mp4_path}")
        return mp4_path
    except subprocess.CalledProcessError as e:
        print(f"Error converting {mov_path}:")
        print(f"Exit code: {e.returncode}")
        if e.stderr:
            print(f"Error output: {e.stderr[-500:]}")  # Print last 500 chars of error
        return None
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg first.")
        print("Install with: brew install ffmpeg (macOS)")
        return None

def last_four_frames(video_path, visualize=False):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_at_2s = int(fps * 2)               # frame index at 2 seconds
    start_frame = max(frame_at_2s - 15, 0)   # go back 15 frames, don't go < 0

    keypoints = []

    # Jump to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Read 15 frames (or until we hit the end of the first 2s)
    for _ in range(15):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            if "left" in video_path.lower():
                joints = [
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                ]
            elif "right" in video_path.lower():
                joints = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                ]
            coords = np.array([[j.x, j.y] for j in joints]).flatten()
        else:
            coords = np.zeros(4)

        keypoints.append(coords)

    cap.release()

    keypoints = np.array(keypoints)

    # Visualize if requested
    if visualize and keypoints.size > 0:
        visualize_wrist_trajectory(keypoints, video_path)

    return keypoints


def collect_keypoints(root_dir, visualize=False):
    rows = []
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for video in os.listdir(folder_path):
            video_lower = video.lower()
            video_path = os.path.join(folder_path, video)

            # Skip videos with "saddagatla" in the name
            if "saddagatla" in video_lower or "llin" in video_lower:
                print(f"Skipping {video_path} (contains 'saddagatla' or 'llin')")
                continue

            # Handle .mov files by converting to .mp4
            if video_lower.endswith(".mov"):
                converted_path = convert_mov_to_mp4(video_path)
                if converted_path is None:
                    print(f"Skipping {video_path} due to conversion error")
                    continue
                video_path = converted_path
                video = os.path.basename(converted_path)
            elif not video_lower.endswith(".mp4"):
                continue

            print(f"Processing {video_path}...")
            keypoints = last_four_frames(video_path, visualize=visualize)
            if keypoints.size == 0:
                continue
            for coords in keypoints.tolist():
                rows.append([folder, video] + coords)
    return rows


def main(visualize=False):
    """
    Main function to extract keypoints from videos

    Args:
        visualize: If True, displays a 3D plot of wrist trajectory for each video (default: False)
    """
    keypoint_rows = collect_keypoints("training-videos", visualize=visualize)
    if not keypoint_rows:
        print("No keypoints extracted.")
        return
    with open('data.csv', mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(keypoint_rows)
    print(f"\nExtracted {len(keypoint_rows)} frames from {len(keypoint_rows)//15} videos")


if __name__ == "__main__":
    import sys

    # Check if --visualize flag is passed
    visualize = "--visualize" in sys.argv or "-v" in sys.argv

    if visualize:
        print("Visualization mode enabled! A 3D plot will show for each video.\n")

    main(visualize=visualize)