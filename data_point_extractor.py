import cv2
import mediapipe as mp
import torch
import numpy as np
import os
import csv
import subprocess

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

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

def last_four_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(total_frames - 15, 0)

    keypoints = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for _ in range(15):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            joints = [
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
            ]
            coords = np.array([[j.x, j.y, j.z] for j in joints]).flatten()
        else:
            coords = np.zeros(12)

        keypoints.append(coords)

    cap.release()
    return np.array(keypoints)


def collect_keypoints(root_dir):
    rows = []
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for video in os.listdir(folder_path):
            video_lower = video.lower()
            video_path = os.path.join(folder_path, video)

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
            keypoints = last_four_frames(video_path)
            if keypoints.size == 0:
                continue
            for coords in keypoints.tolist():
                rows.append([folder, video] + coords)
    return rows


def main():
    keypoint_rows = collect_keypoints("training-videos")
    if not keypoint_rows:
        print("No keypoints extracted.")
        return
    with open('data.csv', mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(keypoint_rows)


if __name__ == "__main__":
    main()