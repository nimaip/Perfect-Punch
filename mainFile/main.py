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

speed_jab = []
speed_hook = []
speed_uppercut = []
correct_punches_thrown = 0
punches_thrown = 0

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
        print(actual_fps)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue  # skip this iteration and try again

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False

        results = pose.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        h, w = image.shape[:2]

        try:
            landmarks = results.pose_landmarks.landmark
        except AttributeError:
            landmarks = None

        

        if TARGET_CENTER is None and landmarks is not None:
            TARGET_CENTER = respawn_target(landmarks, w, h, TARGET_RADIUS)
            CURRENT_TYPE = choose_punch_type()
            last_spawn_ts = time.time()
            circle_spawn_ts = last_spawn_ts
            protect_release_ts = circle_spawn_ts + SPAWN_PROTECT_S

        collide = False
        protecting = protect_release_ts is not None and time.time() < protect_release_ts
        if landmarks is not None and TARGET_CENTER is not None and not protecting:
            collide = wrists_hit_circle(landmarks, w, h, TARGET_CENTER, TARGET_RADIUS)
        
        if not protecting and collide:
            reference_time = circle_spawn_ts if circle_spawn_ts is not None else protect_release_ts
            reaction_time = time.time() - reference_time
            reaction_time_punch[CURRENT_TYPE].append(reaction_time)
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
                if predicted_punch == CURRENT_TYPE:
                    correct_punches_thrown += 1
                punches_thrown += 1
                    
                print(f"Model Output: {output}")
                print(f"Predicted Punch: {predicted_punch}")
            TARGET_CENTER = respawn_target(landmarks, w, h, TARGET_RADIUS)
            CURRENT_TYPE = choose_punch_type()
            print(CURRENT_TYPE, "<-----")
            last_spawn_ts = time.time()
            circle_spawn_ts = last_spawn_ts
            protect_release_ts = circle_spawn_ts + SPAWN_PROTECT_S
        
        if TARGET_CENTER is not None and CURRENT_TYPE is not None:
            color = PUNCH_COLORS.get(CURRENT_TYPE, (0, 0, 255))
            cv2.circle(image, TARGET_CENTER, TARGET_RADIUS, color, -1)

        cv2.imshow("Mediapipe Feed (Press q to quit)", cv2.flip(image,1))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        if time.time() - start_time >= MAX_RUNTIME:
            break
for i in reaction_time_punch:
    reaction_time_punch[i] = sum(reaction_time_punch[i]) / len(reaction_time_punch[i])
    print("Reaction time:", i, reaction_time_punch[i])
print(punches_thrown)
print((correct_punches_thrown/punches_thrown * 100), "% punches correctly thrown")
cap.release()
cv2.destroyAllWindows()
quit()
