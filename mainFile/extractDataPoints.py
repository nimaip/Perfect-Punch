from collections import deque
import mediapipe as mp
import numpy as np
import threading
import time
import torch

mp_pose = mp.solutions.pose

class PoseTracker:
    def __init__(self, max_frames = 15):
        self.pose = mp_pose.Pose(
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5
        )
        self.frame_buffer = deque(maxlen = max_frames)
        self.coord_buffer = deque(maxlen = max_frames)
        self.frame_index = 0
        self.running = False

    def process_frame(self, frame):
        self.frame_index += 1
        frame_rgb = np.copy(frame)
        frame_rgb.flags['WRITEABLE'] = False
        results = self.pose.process(frame_rgb)
        frame_rgb.flags['WRITEABLE'] = True

        landmarks_record = {"frame": self.frame_index, "landmarks": None}

        if results.pose_landmarks:
            selected = [
                mp_pose.PoseLandmark.RIGHT_ELBOW,
                mp_pose.PoseLandmark.RIGHT_WRIST,
                mp_pose.PoseLandmark.LEFT_ELBOW,
                mp_pose.PoseLandmark.LEFT_WRIST,
            ]

            h, w, _ = frame.shape
            coords = {}

            for landmark_id in selected:
                lm = results.pose_landmarks.landmark[landmark_id]
                x, y = int(lm.x * w), int(lm.y * h)
                coords[landmark_id.name] = {"x": x, "y": y}
            
            landmarks_record["landmarks"] = coords

            self.coord_buffer.append(landmarks_record)
            self.frame_buffer.append(np.copy(frame))

    def get_last_fifteen_coords(self):
        return list(self.coord_buffer)
    
    def get_last_frames_tensor(self):
        """Return the last 15 frames as a normalized PyTorch tensor [N,C,H,W]."""
        if len(self.frame_buffer) == 0:
            return None
        frames = list(self.frame_buffer)
        frames = [torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in frames]
        frames_tensor = torch.stack(frames)  # [15, 3, H, W]
        return frames_tensor.unsqueeze(0)
        
    def run(self, get_frame_callable, poll_interval = 0.03):
        self.running = True
        while self.running:
            frame = get_frame_callable()
            if frame is not None:
                self.process_frame(frame)
            time.sleep(poll_interval)

    def stop(self):
        self.running = False