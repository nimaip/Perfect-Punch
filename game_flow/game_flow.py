import cv2
import mediapipe as mp
import numpy as np
import random
from target_utils import respawn_target, wrists_hit_circle, choose_punch_type, PUNCH_COLORS
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
TARGET_CENTER = None
TARGET_RADIUS = 25 
SPAWN_PROTECT_S = 0.5
last_spawn_ts = 0.0
MAX_RUNTIME = 30

start_time = time.time()
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

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

        elapsed = time.time() - start_time
        remaining = max(0, int(MAX_RUNTIME - elapsed))
        cv2.putText(
            image,
            f"Time: {remaining}s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        if TARGET_CENTER is None and landmarks is not None:
            TARGET_CENTER = respawn_target(landmarks, w, h, TARGET_RADIUS)
            CURRENT_TYPE = choose_punch_type()
            last_spawn_ts = time.time()

        collide = False
        protecting = (time.time() - last_spawn_ts) < SPAWN_PROTECT_S
        if landmarks is not None and TARGET_CENTER is not None and not protecting:
            collide = wrists_hit_circle(landmarks, w, h, TARGET_CENTER, TARGET_RADIUS)
        
        if collide:
            TARGET_CENTER = respawn_target(landmarks, w, h, TARGET_RADIUS)
            CURRENT_TYPE = choose_punch_type()
            last_spawn_ts = time.time()
        
        if TARGET_CENTER is not None and CURRENT_TYPE is not None:
            color = PUNCH_COLORS.get(CURRENT_TYPE, (0, 0, 255))
            cv2.circle(image, TARGET_CENTER, TARGET_RADIUS, color, -1)

        cv2.imshow("Mediapipe Feed (Press q to quit)", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        if time.time() - start_time >= MAX_RUNTIME:
            break

cap.release()
cv2.destroyAllWindows()
