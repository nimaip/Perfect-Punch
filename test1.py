import cv2
from collections import deque
import mediapipe as mp
import numpy as np
import time
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
frame_buffer = deque(maxlen=4)
coord_buffer = deque(maxlen=4)
frame_index = 0
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    frame_index += 1
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags['WRITEABLE'] = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    image.flags['WRITEABLE'] = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    coords_record = {"frame": frame_index, "landmarks": None}
    if results.pose_landmarks:
        selected = [
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_WRIST,
        ]
        h, w, _ = image.shape
        coords = {}
        for landmark_id in selected:
            lm = results.pose_landmarks.landmark[landmark_id]
            x, y = int(lm.x * w), int(lm.y * h)
            coords[landmark_id.name] = {"x": x, "y": y}
            cv2.circle(image, (x, y), 10, (0, 255, 0), -1)
        coords_record["landmarks"] = coords
    coord_buffer.append(coords_record)
    frame_buffer.append(image.copy())

    print(f"\nFrame {frame_index}: last {len(coord_buffer)} frames (oldest → newest)")
    for record in coord_buffer:
        coords = record["landmarks"]
        if coords:
            elbow = coords.get("RIGHT_ELBOW")
            wrist = coords.get("RIGHT_WRIST")
            print(f"  Frame {record['frame']}:")
            print(f"    RIGHT_ELBOW -> x: {elbow['x']}, y: {elbow['y']}" if elbow else "    RIGHT_ELBOW -> not detected")
            print(f"    RIGHT_WRIST -> x: {wrist['x']}, y: {wrist['y']}" if wrist else "    RIGHT_WRIST -> not detected")
        else:
            print(f"  Frame {record['frame']}: Pose landmarks not detected.")

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q') or key == 27:
      break
    time.sleep(0.5)
cap.release()
cv2.destroyAllWindows()
