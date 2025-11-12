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
SQUARE_SIZE = 60
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

    # Helper function to get pixel coordinates from normalized landmark
    def get_coords(landmark_id):
        lm = landmarks[landmark_id]
        return int(lm.x * w), int(lm.y * h)

    # Get Wrist Coordinates
    rw_x, rw_y = get_coords(mp_pose.PoseLandmark.RIGHT_WRIST)
    lw_x, lw_y = get_coords(mp_pose.PoseLandmark.LEFT_WRIST)
    ls_x, ls_y = get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)
    rs_x, rs_y = get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    nose_x, nose_y = get_coords(mp_pose.PoseLandmark.NOSE)
    # --- 1. Arm Collision Checks (Highest Priority) ---
    is_rw_colliding = is_point_inside_square(rw_x, rw_y, square)
    is_lw_colliding = is_point_inside_square(lw_x, lw_y, square)
    center_x = (ls_x + rs_x) / 2
    hit_y_min = min(rs_y, nose_y) 
    hit_y_max = max(rs_y, nose_y) 
    # Y overlap: Square vertically intersects the shoulder-to-nose range
    y_intersects_zone = (square.y < hit_y_max) and (square.y + square.size > hit_y_min)
    # if square.side == 'R':
    #     # Square coming from RIGHT, expecting RIGHT arm block
    #     if is_rw_colliding:
    #         return 'Hit'
    #     if is_lw_colliding:
    #         return 'Block'
    #     if CURRENT_SQUARE.x + CURRENT_SQUARE.size < (ls_x + rs_x) / 2: 
    #         return 'Dodge'
    # elif square.side == 'L':
    #     # Square coming from LEFT, expecting LEFT arm block
    #     if is_lw_colliding:
    #         return 'Hit'
    #     if is_rw_colliding:
    #         return 'Block'
    #     if CURRENT_SQUARE.x - CURRENT_SQUARE.size > (ls_x + rs_x) / 2: 
    #         return 'Dodge'
    # # --- 2. Body Collision Check (Lower Priority) ---
    # # Check if any major body landmark (nose, shoulders, hips) is hit
    # for point in BODY_POINTS:
    #     px, py = get_coords(point.value)
    #     if is_point_inside_square(px, py, square):
    #         return 'Hit'
            
    # return None
    if square.side == 'R':
        # Square coming from RIGHT, expecting RIGHT arm block
        if is_rw_colliding or (CURRENT_SQUARE.x + CURRENT_SQUARE.size < (ls_x + rs_x) / 2 and y_intersects_zone):
            return 'Hit'
        if is_lw_colliding: #or square.x < center_x and square.x + square.size < center_x:
            return 'Block'
        if CURRENT_SQUARE.x + CURRENT_SQUARE.size < (ls_x + rs_x) / 2: 
            return 'Dodge'
    
    elif square.side == 'L':
        # Square coming from LEFT, expecting LEFT arm block
        if is_lw_colliding or ((CURRENT_SQUARE.x - CURRENT_SQUARE.size > (ls_x + rs_x) / 2) and y_intersects_zone):
            return 'Hit'
        if is_rw_colliding: #or ((CURRENT_SQUARE.size > (ls_x + rs_x) / 2) and rs_y <= CURRENT_SQUARE.y <= nose_y):
            return 'Block'
        if CURRENT_SQUARE.x - CURRENT_SQUARE.size > (ls_x + rs_x) / 2: 
            return 'Dodge'
    # --- 2. Body Collision Check (Lower Priority) ---
    # Check if any major body landmark (nose, shoulders, hips) is hit
    for point in BODY_POINTS:
        px, py = get_coords(point.value)
        if is_point_inside_square(px, py, square):
            return 'Hit'
            
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
                    CURRENT_SQUARE = None  # Respawn on successful block
                
                elif collision_result == 'Hit':
                    numLanded += 1   # Square landed on wrong arm/body
                    spawn_buffer = random.randint(3, 7)
                    despawn_time = time.time()
                    CURRENT_SQUARE = None # Respawn on hit
                # Check if the square missed (went off screen)
                elif (collision_result == 'Dodge'):
                    numDodged += 1   # Square missed (dodged/missed block)
                    spawn_buffer = random.randint(3, 7) #change back to 3, 7 when done
                    despawn_time = time.time()
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
cap.release()
cv2.destroyAllWindows()