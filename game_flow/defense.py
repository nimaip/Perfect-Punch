import cv2
import mediapipe as mp
import numpy as np
import random
import time
from target_utils import _to_px
# --- CONSTANTS ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Game Settings
MAX_RUNTIME = 30  # seconds
SQUARE_SIZE = 40  # pixels
SQUARE_SPEED = 7  # pixels per frame
SQUARE_COLOR = (0, 255, 255) # Constant Grey color for the target
BODY_POINTS = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
               mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
               mp_pose.PoseLandmark.NOSE]
spawn_buffer = 0
despawn_time = 0
# Game State (Initial State Declaration)
numBlocked = 0 # Track successful blocks (correct arm defense)
numDodged = 0  # Track squares that went off screen (misses)
numLanded = 0  # Track squares that hit the body or the wrong arm
CURRENT_SQUARE = None
start_time = time.time()

# --- UTILITY CLASSES AND FUNCTIONS ---

class DefenseSquare:
    """Represents a moving square target."""
    def __init__(self, x, y, size, dx, side, color):
        self.x = x
        self.y = y
        self.size = size
        self.dx = dx
        self.side = side  # 'L' for left-to-right, 'R' for right-to-left
        self.color = color

def spawn_square(w, h):
    """Creates a new square spawning from the left or right side."""
    side = random.choice(['L', 'R'])
    size = SQUARE_SIZE
    
    # Randomize vertical position (y) within the frame boundaries
    y = random.randint(int(h * 0.2), int(h * 0.8) - size) 

    if side == 'L':
        # Spawns from left, moves right (positive dx)
        x = -size 
        dx = SQUARE_SPEED
    else:
        # Spawns from right, moves left (negative dx)
        x = w
        dx = -SQUARE_SPEED
    
    # Use the constant grey color
    return DefenseSquare(x, y, size, dx, side, SQUARE_COLOR)

def is_point_inside_square(px, py, square):
    """Checks if a point (px, py) is inside the square's bounding box."""
    if square is None:
        return False
    
    # Square bounding box coordinates
    sq_x1, sq_y1 = square.x, square.y
    sq_x2, sq_y2 = square.x + square.size, square.y + square.size
    
    # Check if the point is within the square's X and Y boundaries
    is_inside = (px >= sq_x1 and px <= sq_x2 and
                 py >= sq_y1 and py <= sq_y2)
    
    return is_inside

def check_collision(landmarks, w, h, square):
    if square is None or landmarks is None:
        return None

    # Helper: takes either enum or raw index
    def get_coords(lm_enum_or_index):
        if isinstance(lm_enum_or_index, int):
            lm = landmarks[lm_enum_or_index]
        else:
            lm = landmarks[lm_enum_or_index.value]
        return int(lm.x * w), int(lm.y * h)

    # --- Key joints ---
    # Arms
    rw_x, rw_y = get_coords(mp_pose.PoseLandmark.RIGHT_WRIST)
    lw_x, lw_y = get_coords(mp_pose.PoseLandmark.LEFT_WRIST)
    re_x, re_y = get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW)
    le_x, le_y = get_coords(mp_pose.PoseLandmark.LEFT_ELBOW)
    rs_x, rs_y = get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    ls_x, ls_y = get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)

    # Torso / head
    lh_x, lh_y = get_coords(mp_pose.PoseLandmark.LEFT_HIP)
    rh_x, rh_y = get_coords(mp_pose.PoseLandmark.RIGHT_HIP)
    nose_x, nose_y = get_coords(mp_pose.PoseLandmark.NOSE)

    # Square rect
    sq_x1 = square.x
    sq_y1 = square.y
    sq_x2 = square.x + square.size
    sq_y2 = square.y + square.size

    def rects_overlap(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
        return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

    # --- 1. BLOCK: overlap with either arm rectangle ---
    ARM_PADDING = 20  # makes the arm "thicker"

    # Right arm rectangle (shoulder–elbow–wrist)
    r_arm_x1 = min(rs_x, re_x, rw_x) - ARM_PADDING
    r_arm_x2 = max(rs_x, re_x, rw_x) + ARM_PADDING
    r_arm_y1 = min(rs_y, re_y, rw_y) - ARM_PADDING
    r_arm_y2 = max(rs_y, re_y, rw_y) + ARM_PADDING

    # Left arm rectangle
    l_arm_x1 = min(ls_x, le_x, lw_x) - ARM_PADDING
    l_arm_x2 = max(ls_x, le_x, lw_x) + ARM_PADDING
    l_arm_y1 = min(ls_y, le_y, lw_y) - ARM_PADDING
    l_arm_y2 = max(ls_y, le_y, lw_y) + ARM_PADDING

    if (rects_overlap(sq_x1, sq_y1, sq_x2, sq_y2,
                      r_arm_x1, r_arm_y1, r_arm_x2, r_arm_y2) or
        rects_overlap(sq_x1, sq_y1, sq_x2, sq_y2,
                      l_arm_x1, l_arm_y1, l_arm_x2, l_arm_y2)):
        return 'Block'

    # --- 2. HIT: square overlaps torso/head region ---
    torso_x1 = min(ls_x, rs_x, lh_x, rh_x)
    torso_x2 = max(ls_x, rs_x, lh_x, rh_x)
    torso_y1 = min(ls_y, rs_y, lh_y, rh_y, nose_y)
    torso_y2 = max(ls_y, rs_y, lh_y, rh_y, nose_y)

    overlap_x = not (sq_x2 < torso_x1 or sq_x1 > torso_x2)
    overlap_y = not (sq_y2 < torso_y1 or sq_y1 > torso_y2)

    if overlap_x and overlap_y:
        return 'Hit'

    # --- 3. DODGE: passed center line without hitting anything ---
    center_x = (ls_x + rs_x) / 2

    if square.side == 'R':
        # Coming from the right: once its right edge is left of your center
        if sq_x2 < center_x:
            return 'Dodge'
    else:  # 'L'
        # Coming from the left: once its left edge is right of your center
        if sq_x1 > center_x:
            return 'Dodge'

    return None

# --- MAIN GAME LOOP ---

cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        # Removed the global declaration to fix the SyntaxError. 
        # Variables defined at the top level are modifiable here.
        
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror view
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # 1. MediaPipe Processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw Landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        try:
            landmarks = results.pose_landmarks.landmark
        except AttributeError:
            landmarks = None

        # Calculate time remaining
        elapsed = time.time() - start_time
        remaining = max(0, int(MAX_RUNTIME - elapsed))
        # 2. Game Logic
        if remaining > 0:
            # Spawn a square if none exists
            if CURRENT_SQUARE is None and landmarks is not None and time.time() - despawn_time >= spawn_buffer:
                CURRENT_SQUARE = spawn_square(w, h)
            
            if CURRENT_SQUARE is not None:
                # Update position
                CURRENT_SQUARE.x += CURRENT_SQUARE.dx
                
                # Check for collision
                collision_result = check_collision(landmarks, w, h, CURRENT_SQUARE)
            

                if collision_result == 'Block':
                    numBlocked += 1  # Successfully blocked
                    spawn_buffer = random.randint(3, 7)
                    despawn_time = time.time()
                    print("REGISTERED: Block")
                    CURRENT_SQUARE = None  # Respawn on successful block
                
                elif collision_result == 'Hit':
                    numLanded += 1   # Square landed on wrong arm/body
                    spawn_buffer = random.randint(3, 7)
                    despawn_time = time.time()
                    print("REGISTERED: Hit")
                    CURRENT_SQUARE = None # Respawn on hit
                # Check if the square missed (went off screen)
                elif (collision_result == 'Dodge'):
                    numDodged += 1   # Square missed (dodged/missed block)
                    spawn_buffer = random.randint(3, 7) #change back to 3, 7 when done
                    despawn_time = time.time()
                    print("REGISTERED: Dodge")
                    CURRENT_SQUARE = None # Respawn on miss
                print()


            # Draw the square
            if CURRENT_SQUARE is not None:
                cv2.rectangle(
                    image, 
                    (CURRENT_SQUARE.x, CURRENT_SQUARE.y), 
                    (CURRENT_SQUARE.x + CURRENT_SQUARE.size, CURRENT_SQUARE.y + CURRENT_SQUARE.size), 
                    CURRENT_SQUARE.color, 
                    -1
                )
        
        # 3. HUD Display
        
        # Display Time
        # cv2.putText(image, f"TIME: {remaining}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Display Stats (Placed on the right side)
        # cv2.putText(image, f"BLOCKED: {numBlocked}", (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA) # Green
        # cv2.putText(image, f"LANDED: {numLanded}", (w - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA) # Red
        # cv2.putText(image, f"DODGED: {numDodged}", (w - 250, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA) # Yellow
        #^^^^ USEFUL FOR DEBUGGING
        # Time's Up Display
        if remaining <= 0:
            final_message = f"Drill Complete! Stats: Blocked: {numBlocked}, Landed: {numLanded}, Dodged: {numDodged}"
            
            # Draw Time's Up message in the center
            text_size = cv2.getTextSize(final_message, cv2.FONT_HERSHEY_DUPLEX, 1, 2)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            
            
            # Wait indefinitely until 'q' is pressed to close the window
            cv2.imshow("Defense Training (Press q to quit)", image)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

        else:
            cv2.imshow("Defense Training (Press q to quit)", image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

print(numBlocked, numDodged, numLanded)
cap.release()
cv2.destroyAllWindows()