import cv2
import mediapipe as mp
import numpy as np
import random
from target_utils import respawn_target, wrists_hit_circle, choose_punch_type, PUNCH_COLORS
import time
from extractDataPoints import PoseTracker
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from datetime import datetime
import math

class Model(nn.Module):
    def __init__(self, in_features=120, h1=128, h2=64, out_features=3):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x




tracker = PoseTracker(max_frames=15)

model = Model()
model.load_state_dict(torch.load("models/model_state.pt"))
model.eval()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
TARGET_CENTER = None
TARGET_RADIUS = 25 
SPAWN_PROTECT_S = 1
last_spawn_ts = 0.0
circle_spawn_ts = None
protect_release_ts = None
MAX_RUNTIME = 30

reaction_time_punch = {"jab": [], "hook": [], "uppercut": []}
reaction_time_windows = [
    {k: [] for k in reaction_time_punch.keys()},
    {k: [] for k in reaction_time_punch.keys()},
    {k: [] for k in reaction_time_punch.keys()},
]
correct_punches_thrown = 0
punches_thrown = 0
attempts_by_type = {k: 0 for k in reaction_time_punch.keys()}
correct_by_type = {k: 0 for k in reaction_time_punch.keys()}
accuracy_windows = [{"correct": 0, "attempts": 0} for _ in range(3)]
reaction_window_combined = [[] for _ in range(3)]
speed_by_type = {k: [] for k in reaction_time_punch.keys()}
speed_windows = [
    {k: [] for k in reaction_time_punch.keys()},
    {k: [] for k in reaction_time_punch.keys()},
    {k: [] for k in reaction_time_punch.keys()},
]
speed_window_combined = [[] for _ in range(3)]
attempts_by_type = {k: 0 for k in reaction_time_punch.keys()}
correct_by_type = {k: 0 for k in reaction_time_punch.keys()}

start_time = time.time()
cap = cv2.VideoCapture(0)

def get_frame():
    ret, frame = cap.read()
    return frame if ret else None

thread = threading.Thread(target = tracker.run, args = (get_frame,))
thread.start()



with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue  # skip this iteration and try again

        now = time.time()
        elapsed = now - start_time

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False

        results = pose.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w = image.shape[:2]

        try:
            landmarks = results.pose_landmarks.landmark
        except AttributeError:
            landmarks = None

        

        if TARGET_CENTER is None and landmarks is not None:
            TARGET_CENTER = respawn_target(landmarks, w, h, TARGET_RADIUS)
            CURRENT_TYPE = choose_punch_type()
            last_spawn_ts = now
            circle_spawn_ts = last_spawn_ts
            protect_release_ts = circle_spawn_ts + SPAWN_PROTECT_S

        collide = False
        protecting = protect_release_ts is not None and now < protect_release_ts
        if landmarks is not None and TARGET_CENTER is not None and not protecting:
            collide = wrists_hit_circle(landmarks, w, h, TARGET_CENTER, TARGET_RADIUS)
        
        if not protecting and collide:
            reference_time = circle_spawn_ts if circle_spawn_ts is not None else protect_release_ts
            reaction_time = now - reference_time

            reaction_time_punch[CURRENT_TYPE].append(reaction_time)

            window_idx = None
            if elapsed <= 10:
                window_idx = 0
            elif 10 < elapsed <= 20:
                window_idx = 1
            elif 20 < elapsed <= 30:
                window_idx = 2

            if window_idx is not None:
                reaction_time_windows[window_idx][CURRENT_TYPE].append(reaction_time)
                reaction_window_combined[window_idx].append(reaction_time)

            speed_value = None
            raw_coords = tracker.get_last_fifteen_coords()
            if raw_coords:
                first_landmarks = raw_coords[0].get("landmarks")
                last_landmarks = raw_coords[min(14, len(raw_coords) - 1)].get("landmarks")
                if first_landmarks and last_landmarks:
                    distances = []
                    for wrist_name in ("RIGHT_WRIST", "LEFT_WRIST"):
                        if wrist_name in first_landmarks and wrist_name in last_landmarks:
                            dx = last_landmarks[wrist_name]["x"] - first_landmarks[wrist_name]["x"]
                            dy = last_landmarks[wrist_name]["y"] - first_landmarks[wrist_name]["y"]
                            distances.append(math.hypot(dx, dy))
                    if distances:
                        distance_px = max(distances)
                        fps_value = actual_fps if actual_fps and actual_fps > 0 else 30.0
                        frame_count = min(15, len(raw_coords))
                        if frame_count > 1 and fps_value:
                            duration_s = frame_count / fps_value
                            speed_value = distance_px / duration_s
                            speed_by_type[CURRENT_TYPE].append(speed_value)
                            if window_idx is not None:
                                speed_windows[window_idx][CURRENT_TYPE].append(speed_value)
                                speed_window_combined[window_idx].append(speed_value)

            coords = tracker.get_last_normalized_coords()
            if coords:
                # Flatten and normalize like in training
                features = []
                for record in coords:
                    if record["landmarks"]:
                        for key in record["landmarks"].values():
                            features.extend([key["x"], key["y"]])
                
                # Convert to tensor
                x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # shape [1, 120]

                # Make prediction
                with torch.no_grad():
                    output = model(x)
                    pred_class = output.argmax(dim=1).item()  # get predicted class index

                # Map prediction to punch type
                punch_map = {0: "hook", 1: "jab", 2: "uppercut"}
                predicted_punch = punch_map[pred_class]
                attempts_by_type[CURRENT_TYPE] += 1
                punches_thrown += 1
                if window_idx is not None:
                    accuracy_windows[window_idx]["attempts"] += 1
                if predicted_punch == CURRENT_TYPE:
                    correct_punches_thrown += 1
                    correct_by_type[CURRENT_TYPE] += 1
                    if window_idx is not None:
                        accuracy_windows[window_idx]["correct"] += 1
                    
                print(f"Model Output: {output}")
                print(f"Predicted Punch: {predicted_punch}")
            TARGET_CENTER = respawn_target(landmarks, w, h, TARGET_RADIUS)
            CURRENT_TYPE = choose_punch_type()
            print(CURRENT_TYPE, "<-----")
            last_spawn_ts = now
            circle_spawn_ts = last_spawn_ts
            protect_release_ts = circle_spawn_ts + SPAWN_PROTECT_S
        
        if TARGET_CENTER is not None and CURRENT_TYPE is not None:
            color = PUNCH_COLORS.get(CURRENT_TYPE, (0, 0, 255))
            cv2.circle(image, TARGET_CENTER, TARGET_RADIUS, color, -1)

        cv2.imshow("Mediapipe Feed (Press q to quit)", cv2.flip(image,1))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        if elapsed >= MAX_RUNTIME:
            break

all_reaction_times = [t for times in reaction_time_punch.values() for t in times]

def round_or_none(value, digits=2):
    return round(value, digits) if value is not None else None

reaction_types_ms = {}
for punch_type, times in reaction_time_punch.items():
    if times:
        avg_ms = (sum(times) / len(times)) * 1000
        reaction_types_ms[punch_type] = round_or_none(avg_ms)
    else:
        reaction_types_ms[punch_type] = None

reaction_all_points_ms = []
for window_times in reaction_window_combined:
    if window_times:
        avg_ms = (sum(window_times) / len(window_times)) * 1000
        reaction_all_points_ms.append(round_or_none(avg_ms))
    else:
        reaction_all_points_ms.append(None)

if all_reaction_times:
    reaction_average_ms = round_or_none((sum(all_reaction_times) / len(all_reaction_times)) * 1000)
    reaction_best_ms = round_or_none(min(all_reaction_times) * 1000)
    reaction_worst_ms = round_or_none(max(all_reaction_times) * 1000)
else:
    reaction_average_ms = reaction_best_ms = reaction_worst_ms = None

accuracy_types = {}
for punch_type in reaction_time_punch.keys():
    attempts = attempts_by_type[punch_type]
    correct = correct_by_type[punch_type]
    if attempts:
        accuracy_types[punch_type] = round_or_none((correct / attempts) * 100)
    else:
        accuracy_types[punch_type] = None

window_accuracy_points = []
for window in accuracy_windows:
    if window["attempts"]:
        window_accuracy_points.append(round_or_none((window["correct"] / window["attempts"]) * 100))
    else:
        window_accuracy_points.append(None)

if punches_thrown:
    overall_accuracy_percent = round_or_none((correct_punches_thrown / punches_thrown) * 100)
else:
    overall_accuracy_percent = None

valid_accuracy_points = [pt for pt in window_accuracy_points if pt is not None]
if valid_accuracy_points:
    accuracy_best = max(valid_accuracy_points)
    accuracy_worst = min(valid_accuracy_points)
else:
    accuracy_best = accuracy_worst = None

all_speeds = [s for values in speed_by_type.values() for s in values]

speed_types = {}
for punch_type, values in speed_by_type.items():
    if values:
        speed_types[punch_type] = round_or_none(sum(values) / len(values))
    else:
        speed_types[punch_type] = None

speed_all_points = []
for window_values in speed_window_combined:
    if window_values:
        speed_all_points.append(round_or_none(sum(window_values) / len(window_values)))
    else:
        speed_all_points.append(None)

if all_speeds:
    speed_average = round_or_none(sum(all_speeds) / len(all_speeds))
    speed_best = round_or_none(max(all_speeds))
    speed_worst = round_or_none(min(all_speeds))
else:
    speed_average = speed_best = speed_worst = None

speed_segments = {
    "first_third": speed_all_points[0],
    "second_third": speed_all_points[1],
    "final_third": speed_all_points[2],
}

fighter_id = "fighter_sample_001"
session_id = f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
timestamp_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

session_payload = [{
    "fighter_id": fighter_id,
    "session_id": session_id,
    "timestamp": timestamp_iso,
    "metrics": {
        "offense": {
            "punch_accuracy": {
                "unit": "%",
                "types": accuracy_types,
                "derived": {
                    "average": overall_accuracy_percent,
                    "average_description": "Average of all recorded punch accuracies across the session.",
                    "best": accuracy_best,
                    "worst": accuracy_worst,
                    "all_points": window_accuracy_points
                }
            },
            "punch_reaction_time": {
                "unit": "ms",
                "types": reaction_types_ms,
                "derived": {
                    "average": reaction_average_ms,
                    "average_description": "Average reaction time, in milliseconds, measured across all punches.",
                    "best": reaction_best_ms,
                    "worst": reaction_worst_ms,
                    "all_points": reaction_all_points_ms,
                    "all_points_description": "Reaction time averages for each 10-second segment of the session."
                }
            },
            "punch_speed": {
                "unit": "px/s",
                "types": speed_types,
                "derived": {
                    "average": speed_average,
                    "average_description": "Mean punch speed across all punch types during the session.",
                    "best": speed_best,
                    "worst": speed_worst,
                    "all_points": speed_all_points,
                    "combo_tempo": {
                        "unit": "px/s",
                        "segments": speed_segments,
                        "description": "Average punch speed per 10-second segment of the session."
                    }
                }
            }
        }
    }
}]

with open("session_metrics.json", "w", encoding="utf-8") as metric_file:
    json.dump(session_payload, metric_file, indent=2)

print("Session metrics written to session_metrics.json")

cap.release()
cv2.destroyAllWindows()
quit()
