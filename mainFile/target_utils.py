import random
import mediapipe as mp

mp_pose = mp.solutions.pose

PUNCH_COLORS = {
    "jab": (0, 0, 255),        # red
    "hook": (0, 255, 0),       # green
    "uppercut": (255, 0, 0)    # blue
}

#Convert normalized landmark to pixel coordinates
def _to_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

#Respawn target within the area defined by shoulders, hips, and nose
def respawn_target(landmarks, w, h, target_radius):
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

    (lsx, lsy) = _to_px(ls, w, h)
    (rsx, rsy) = _to_px(rs, w, h)
    (lhy, rhy) = (_to_px(lh, w, h)[1], _to_px(rh, w, h)[1])
    (nx, ny)   = _to_px(nose, w, h)

    shoulder_width = abs(rsx - lsx)

    xmin = max(target_radius, min(lsx, rsx) - shoulder_width)
    xmax = min(w - target_radius, max(lsx, rsx) + shoulder_width)

    waist_y = (lhy + rhy) // 2
    ymin = max(target_radius, min(ny, waist_y))
    ymax = min(h - target_radius, max(ny, waist_y))

    if xmax <= xmin or ymax <= ymin:
        return None

    return (random.randint(xmin, xmax), random.randint(ymin, ymax))

#Detect if either wrist is within the target circle
def wrists_hit_circle(landmarks, w, h, center, radius):
    if center is None:
        return False
    cx, cy = center
    for wid in (mp_pose.PoseLandmark.LEFT_WRIST.value,
                mp_pose.PoseLandmark.RIGHT_WRIST.value):
        lm = landmarks[wid]
        if lm.visibility < 0.5:
            continue
        x, y = _to_px(lm, w, h)
        dx, dy = x - cx, y - cy
        if dx*dx + dy*dy <= radius*radius:
            return True
    return False

#Randomly choose a punch type based on defined probabilities (Will update to use coach agent to determine probabilities)
def choose_punch_type():
    r = random.random()
    if r < 0.625:
        return "jab"
    elif r < 0.625 + 0.25:
        return "hook"
    else:
        return "uppercut"