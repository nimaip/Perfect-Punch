
Small pose-tracking punching game using OpenCV + MediaPipe.


- Webcam feed with pose tracking
- Target circle spawns between nose and waist, within shoulder width bounds
- Circle color indicates requested punch type:
  - Red = Jab (62.5%)
  - Green = Hook (25%)
  - Blue = Uppercut (12.5%)
- Hit detection with wrist–circle collision
- Spawn protection window to avoid accidental hits
- Auto-exit after 30 seconds + on-screen timer

