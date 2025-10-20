import cv2
import mediapipe as mp
import torch
import numpy as np
import os
import csv

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def last_four_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(total_frames - 4, 0)

    keypoints = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for _ in range(4):
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
            coords = np.array([[j.x, j.y] for j in joints]).flatten()
        else:
            coords = np.zeros(8)

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
            if not video.lower().endswith(".mp4"):
                continue
            video_path = os.path.join(folder_path, video)
            keypoints = last_four_frames(video_path)
            if keypoints.size == 0:
                continue
            rows.extend(keypoints.tolist())
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