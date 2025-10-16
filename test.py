import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)


def bbox_center_area_from_landmarks(landmarks, img_w, img_h):
    """Return bounding box (x,y,w,h), center (cx,cy), area and average z for a list of landmarks."""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    zs = [lm.z for lm in landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Clamp and convert to pixel coords
    x = max(0, int(min_x * img_w))
    y = max(0, int(min_y * img_h))
    w = max(1, int((max_x - min_x) * img_w))
    h = max(1, int((max_y - min_y) * img_h))
    cx = int((min_x + max_x) / 2.0 * img_w)
    cy = int((min_y + max_y) / 2.0 * img_h)
    area = w * h
    avg_z = sum(zs) / len(zs)
    return (x, y, w, h), (cx, cy), area, avg_z


# Set up MediaPipe Hands
with mp_hands.Hands(
    static_image_mode=False,       # For real-time detection
    max_num_hands=2,              # Detect up to 2 hands
    min_detection_confidence=0.5, # Minimum confidence for detection
    min_tracking_confidence=0.5   # Minimum confidence for tracking
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        # Convert the frame to RGB (MediaPipe requires RGB input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = hands.process(rgb_frame)

        h_img, w_img, _ = frame.shape

        # Collect detected hands information
        hands_info = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label  # 'Left' or 'Right'
                score = handedness.classification[0].score
                bbox, center, area, avg_z = bbox_center_area_from_landmarks(
                    hand_landmarks.landmark, w_img, h_img)
                hands_info.append({
                    'label': label,
                    'score': score,
                    'landmarks': hand_landmarks,
                    'bbox': bbox,
                    'center': center,
                    'area': area,
                    'avg_z': avg_z,
                })

        # If we detected two hands, decide which one is "in front" using area as a proxy for depth.
        # If areas are very close we consider them "Even". We also compute left/right based on bbox center.
        if len(hands_info) == 2:
            a0 = hands_info[0]['area']
            a1 = hands_info[1]['area']
            # Avoid division by zero
            if max(a0, a1) == 0:
                diff_ratio = 0.0
            else:
                diff_ratio = abs(a0 - a1) / max(a0, a1)

            AREA_EVEN_THRESHOLD = 0.15  # 15% area difference -> consider even

            if diff_ratio < AREA_EVEN_THRESHOLD:
                # Consider them even
                for info in hands_info:
                    x, y, w, hh = info['bbox']
                    cx, cy = info['center']
                    side = 'Left' if cx < (w_img / 2) else 'Right'
                    mp_drawing.draw_landmarks(frame, info['landmarks'], mp_hands.HAND_CONNECTIONS)
                    cv2.rectangle(frame, (x, y), (x + w, y + hh), (0, 200, 200), 2)
                    cv2.putText(frame, f"{info['label']} Even ({side})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)
            else:
                # Larger area -> hand farther to camera (in front)
                front_idx = 0 if a0 > a1 else 1
                for i, info in enumerate(hands_info):
                    x, y, w, hh = info['bbox']
                    cx, cy = info['center']
                    side = 'Left' if cx < (w_img / 2) else 'Right'
                    status = 'Front' if i == front_idx else 'Back'
                    mp_drawing.draw_landmarks(frame, info['landmarks'], mp_hands.HAND_CONNECTIONS)
                    color = (0, 255, 0) if status == 'Front' else (0, 120, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + hh), color, 2)
                    cv2.putText(frame, f"{info['label']} {status} ({side})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        elif len(hands_info) == 1:
            # Single hand: just show left/right and handedness
            info = hands_info[0]
            x, y, w, hh = info['bbox']
            cx, cy = info['center']
            side = 'Left' if cx < (w_img / 2) else 'Right'
            mp_drawing.draw_landmarks(frame, info['landmarks'], mp_hands.HAND_CONNECTIONS)
            cv2.rectangle(frame, (x, y), (x + w, y + hh), (200, 200, 0), 2)
            cv2.putText(frame, f"{info['label']} ({side})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 0), 2)

        # Show the frame
        cv2.imshow("Hand Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
