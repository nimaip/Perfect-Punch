import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.signal import find_peaks
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the model class (same as training)
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

# Load the trained model
model = Model()
model.load_state_dict(torch.load('models/model_state.pt'))
model.eval()

PUNCH_CLASSES = {0: 'hook', 1: 'jab', 2: 'uppercut'}

def preprocess_coords(coords_buffer):
    """Preprocess 15 frames of coords the same way as training"""
    # coords_buffer is a list of 15 frames, each frame has 8 values: [x1,y1,x2,y2,x3,y3,x4,y4]
    arr = np.array(coords_buffer)  # shape: (15, 8)
    
    # Make relative to first frame for each landmark
    for i in range(4):
        x_idx, y_idx = i*2, i*2+1
        x0, y0 = arr[0, x_idx], arr[0, y_idx]
        arr[:, x_idx] = arr[:, x_idx] - x0
        arr[:, y_idx] = arr[:, y_idx] - y0
    
    # Reshape to (15, 4, 2) for distance calculation
    coords_reshaped = arr.reshape(15, 4, 2)
    distances = np.linalg.norm(coords_reshaped, axis=2)
    max_distance = distances.max()
    
    if max_distance > 0:
        coords_normalized = coords_reshaped / max_distance
    else:
        coords_normalized = coords_reshaped
    
    # Flatten to 120 features
    return coords_normalized.flatten()

def classify_punch(coords_buffer):
    """Run the classifier on the coordinates buffer"""
    preprocessed = preprocess_coords(coords_buffer)
    input_tensor = torch.FloatTensor(preprocessed).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax().item()
    
    return PUNCH_CLASSES[predicted_class]

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
left_data_list = []
time_list = []

cap = cv2.VideoCapture(0)

start_time = time.time()
last_time_entry = 0.0
current_extension_left = 0.0  # Track latest extensionLeft value

# For live peak detection
detected_peaks = []  # Store detected peak times to avoid duplicates
last_value = None
peak_cooldown = 0  # Prevent detecting multiple peaks too quickly

# For velocity-based detection
velocity_history = []  # Store recent velocities
extension_history = []  # Store recent extensions for smoothing
is_punching = False  # Track if we're in a punch motion
punch_start_extension = None  # Extension at start of punch

# Buffer to store last 15 frames of coordinates for classification
coords_buffer = []  # Each entry: [x1,y1,x2,y2,x3,y3,x4,y4] = RIGHT_WRIST, LEFT_WRIST, RIGHT_ELBOW, LEFT_ELBOW

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the frame
        results = pose.process(image)

        # Convert back to BGR for display
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get shoulder and wrist positions
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]

            # Store coordinates for classification: [RIGHT_WRIST, LEFT_WRIST, RIGHT_ELBOW, LEFT_ELBOW]
            frame_coords = [
                right_wrist.x, right_wrist.y,
                left_wrist.x, left_wrist.y,
                right_elbow.x, right_elbow.y,
                left_elbow.x, left_elbow.y
            ]
            coords_buffer.append(frame_coords)
            if len(coords_buffer) > 15:
                coords_buffer.pop(0)  # Keep only last 15 frames

            left_wrist_array = np.array([left_wrist.x, left_wrist.y, left_wrist.z])
            left_shoulder_array = np.array([left_shoulder.x, left_shoulder.y, left_shoulder.z])
            extension_left = abs(np.linalg.norm(left_wrist_array - left_shoulder_array))         
            current_extension_left = extension_left  # Update current value

            # Check if 0.1 seconds has passed since last time entry
            elapsed = time.time() - start_time
            while elapsed >= last_time_entry + 0.1:
                last_time_entry += 0.1
                time_list.append(round(last_time_entry, 1))
                left_data_list.append(current_extension_left)

            # Live peak detection using velocity (rate of change)
            if peak_cooldown > 0:
                peak_cooldown -= 1
            
            # Store extension history for smoothing
            extension_history.append(extension_left)
            if len(extension_history) > 5:
                extension_history.pop(0)
            
            # Calculate velocity (change in extension)
            if last_value is not None:
                velocity = extension_left - last_value
                velocity_history.append(velocity)
                if len(velocity_history) > 10:
                    velocity_history.pop(0)
                
                # Only process if we have enough history
                if len(velocity_history) >= 5 and len(left_data_list) >= 30:
                    avg_velocity = np.mean(velocity_history[-5:])
                    max_recent_velocity = max(velocity_history[-5:]) if velocity_history[-5:] else 0
                    
                    # Velocity threshold - needs significant outward movement to start punch detection
                    velocity_threshold = 0.008  # Adjust this based on testing
                    
                    # Start tracking a punch when we see fast outward movement
                    if not is_punching and max_recent_velocity > velocity_threshold and peak_cooldown == 0:
                        is_punching = True
                        punch_start_extension = extension_left
                    
                    # If we're in a punch, look for the peak (velocity goes from positive to negative)
                    if is_punching:
                        # Check if we've reached the peak (velocity becomes negative after being positive)
                        if avg_velocity < -0.002 and extension_left > punch_start_extension + 0.05:
                            # This is a real punch peak!
                            current_time = round(time.time() - start_time, 1)
                            
                            # Classify the punch using last 15 frames
                            if len(coords_buffer) >= 15:
                                punch_type = classify_punch(coords_buffer[-15:])
                                print(f"\n*** PUNCH detected at {current_time}s! Extension: {extension_left:.4f} ***")
                                print(f"*** Punch Type: {punch_type.upper()} ***\n")
                                detected_peaks.append(('PUNCH', current_time, extension_left, punch_type))
                            else:
                                print(f"\n*** PUNCH detected at {current_time}s! Extension: {extension_left:.4f} ***")
                                print(f"*** Not enough frames for classification ***\n")
                                detected_peaks.append(('PUNCH', current_time, extension_left, 'unknown'))
                            
                            is_punching = False
                            punch_start_extension = None
                            peak_cooldown = 30  # Cooldown to avoid duplicate detections
                        
                        # Reset if extension dropped below start (aborted punch or arm returned)
                        elif extension_left < punch_start_extension - 0.02:
                            is_punching = False
                            punch_start_extension = None
            
            last_value = extension_left

            # Print positions every frame
            # print(f"Left Shoulder:  x={left_shoulder.x:.3f}, y={left_shoulder.y:.3f}, z={left_shoulder.z:.3f}")
            # print(f"Right Shoulder: x={right_shoulder.x:.3f}, y={right_shoulder.y:.3f}, z={right_shoulder.z:.3f}")
            # print(f"Left Wrist:     x={left_wrist.x:.3f}, y={left_wrist.y:.3f}, z={left_wrist.z:.3f}")
            # print(f"Right Wrist:    x={right_wrist.x:.3f}, y={right_wrist.y:.3f}, z={right_wrist.z:.3f}")
            # print("-" * 50)

            # Draw pose landmarks on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Mirror the image horizontally
        image = cv2.flip(image, 1)
        cv2.imshow('Pose Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print("\n=== LIVE DETECTED PEAKS ===")
for peak_type, t, v, punch_class in detected_peaks:
    if punch_class:
        print(f"  Time: {t}s, Extension: {v:.4f} ({peak_type}) - Punch: {punch_class}")
    else:
        print(f"  Time: {t}s, Extension: {v:.4f} ({peak_type})")

print("\nTime list:", time_list)
print("Left data list:", left_data_list)
df = pd.DataFrame({"extension": left_data_list}, index=time_list)

# Calculate dynamic threshold based on data spread
data_std = np.std(left_data_list)
data_median = np.median(left_data_list)
prominence_threshold = data_std * 1.5  # Only detect peaks that stand out significantly

# Find prominent local maxima
maxima, max_props = find_peaks(df["extension"], prominence=prominence_threshold, distance=5)
# Find prominent local minima (invert the signal)
minima, min_props = find_peaks(-df["extension"], prominence=prominence_threshold, distance=5)

# Filter maxima: only keep those significantly ABOVE median (true punch extensions)
# Filter minima: only keep those significantly BELOW median (true retractions)
valid_maxima = [i for i in maxima if left_data_list[i] > data_median + data_std]
valid_minima = [i for i in minima if left_data_list[i] < data_median - data_std]

# Combine and sort by time
all_peaks = sorted(valid_maxima + valid_minima)
peak_times = [time_list[i] for i in all_peaks]
peak_values = [left_data_list[i] for i in all_peaks]

print(f"\nMedian: {data_median:.4f}, Std: {data_std:.4f}")
print(f"Max threshold: > {data_median + data_std:.4f}")
print(f"Min threshold: < {data_median - data_std:.4f}")
print("\nPunch peaks detected at times:")
for t, v in zip(peak_times, peak_values):
    peak_type = "MAX" if v > data_median else "MIN"
    print(f"  Time: {t}s, Extension: {v:.4f} ({peak_type})")

df.plot(title="DataFrame Plot")
# Plot maxima in red, minima in blue
max_times = [time_list[i] for i in valid_maxima]
max_values = [left_data_list[i] for i in valid_maxima]
min_times = [time_list[i] for i in valid_minima]
min_values = [left_data_list[i] for i in valid_minima]

plt.scatter(max_times, max_values, color='r', s=100, zorder=5, label='Maxima')
plt.scatter(min_times, min_values, color='b', s=100, zorder=5, label='Minima')
plt.axhline(y=data_median, color='gray', linestyle='-', alpha=0.5, label='Median')
plt.axhline(y=data_median + data_std, color='r', linestyle='--', alpha=0.3)
plt.axhline(y=data_median - data_std, color='b', linestyle='--', alpha=0.3)
plt.legend()
plt.show()