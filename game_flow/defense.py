import cv2
import mediapipe as mp
import random

mp_pose = mp.solutions.pose

SQUARE_SIZE = 40
SQUARE_SPEED = 7
SQUARE_COLOR = (0, 255, 255)
SPAWN_DELAY_RANGE = (3, 7)
ARM_PADDING = 20


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
    """Create a new square that starts from the left or right edge of the frame."""

    side = random.choice(["L", "R"])
    size = SQUARE_SIZE
    y_min = int(h * 0.2)
    y_max = max(y_min + 1, int(h * 0.8) - size)
    y = random.randint(y_min, y_max)

    if side == "L":
        x = -size
        dx = SQUARE_SPEED
    else:
        x = w
        dx = -SQUARE_SPEED

    return DefenseSquare(x, y, size, dx, side, SQUARE_COLOR)


def check_collision(landmarks, w, h, square):
    """Classify the interaction between the moving square and the defender."""

    if square is None or landmarks is None:
        return None

    def get_coords(lm_enum_or_index):
        if isinstance(lm_enum_or_index, int):
            lm = landmarks[lm_enum_or_index]
        else:
            lm = landmarks[lm_enum_or_index.value]
        return int(lm.x * w), int(lm.y * h)

    # Arm joints
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

    sq_x1 = square.x
    sq_y1 = square.y
    sq_x2 = square.x + square.size
    sq_y2 = square.y + square.size

    def rects_overlap(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
        return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

    r_arm_x1 = min(rs_x, re_x, rw_x) - ARM_PADDING
    r_arm_x2 = max(rs_x, re_x, rw_x) + ARM_PADDING
    r_arm_y1 = min(rs_y, re_y, rw_y) - ARM_PADDING
    r_arm_y2 = max(rs_y, re_y, rw_y) + ARM_PADDING

    l_arm_x1 = min(ls_x, le_x, lw_x) - ARM_PADDING
    l_arm_x2 = max(ls_x, le_x, lw_x) + ARM_PADDING
    l_arm_y1 = min(ls_y, le_y, lw_y) - ARM_PADDING
    l_arm_y2 = max(ls_y, le_y, lw_y) + ARM_PADDING

    if (
        rects_overlap(sq_x1, sq_y1, sq_x2, sq_y2, r_arm_x1, r_arm_y1, r_arm_x2, r_arm_y2)
        or rects_overlap(sq_x1, sq_y1, sq_x2, sq_y2, l_arm_x1, l_arm_y1, l_arm_x2, l_arm_y2)
    ):
        return "Block"

    torso_x1 = min(ls_x, rs_x, lh_x, rh_x)
    torso_x2 = max(ls_x, rs_x, lh_x, rh_x)
    torso_y1 = min(ls_y, rs_y, lh_y, rh_y, nose_y)
    torso_y2 = max(ls_y, rs_y, lh_y, rh_y, nose_y)

    overlap_x = not (sq_x2 < torso_x1 or sq_x1 > torso_x2)
    overlap_y = not (sq_y2 < torso_y1 or sq_y1 > torso_y2)

    if overlap_x and overlap_y:
        return "Hit"

    center_x = (ls_x + rs_x) / 2
    if square.side == "R":
        if sq_x2 < center_x:
            return "Dodge"
    else:
        if sq_x1 > center_x:
            return "Dodge"

    return None


class DefenseGame:
    """Stateful manager for the flying-blocks defense drill."""

    def __init__(self, spawn_delay_range=SPAWN_DELAY_RANGE):
        self.spawn_delay_range = spawn_delay_range
        self.reset()

    def reset(self):
        self.current_square = None
        self.spawn_buffer = 0.0
        self.despawn_time = 0.0
        self.stats = {
            "blocked": 0,
            "dodged": 0,
            "hit": 0,
        }

    def _schedule_respawn(self, now):
        delay = random.randint(*self.spawn_delay_range)
        self.spawn_buffer = float(delay)
        self.despawn_time = now
        self.current_square = None

    def update(self, frame, landmarks, now):
        """Advance game state, draw the active square, and return any outcome."""

        if frame is None:
            return None

        h, w = frame.shape[:2]

        if (
            self.current_square is None
            and landmarks is not None
            and (now - self.despawn_time) >= self.spawn_buffer
        ):
            self.current_square = spawn_square(w, h)

        outcome = None
        if self.current_square is not None:
            self.current_square.x += self.current_square.dx
            outcome = check_collision(landmarks, w, h, self.current_square)

            if outcome == "Block":
                self.stats["blocked"] += 1
                self._schedule_respawn(now)
            elif outcome == "Hit":
                self.stats["hit"] += 1
                self._schedule_respawn(now)
            elif outcome == "Dodge":
                self.stats["dodged"] += 1
                self._schedule_respawn(now)

        if self.current_square is not None:
            cv2.rectangle(
                frame,
                (self.current_square.x, self.current_square.y),
                (
                    self.current_square.x + self.current_square.size,
                    self.current_square.y + self.current_square.size,
                ),
                self.current_square.color,
                -1,
            )

        return outcome

    def get_stats(self):
        return dict(self.stats)
