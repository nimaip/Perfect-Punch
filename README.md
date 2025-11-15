% Perfect Punch

# Perfect Punch

Realtime boxing drill that blends reaction targets, punch recognition, and defensive "flying blocks" using nothing more than a webcam, MediaPipe pose tracking, and a lightweight PyTorch classifier. Every 30‑second session produces detailed offense and defense analytics saved to `session_metrics.json` for review or downstream analysis.

## Project Overview

- **Dynamic Offense Targets** – Randomized colored circles appear around the torso. A one-second protection window enforces realistic reaction times before a punch can score.
- **Punch Type Classification** – A 3-layer fully connected PyTorch network (`models/model_state.pt`) infers jab, hook, or uppercut from 15-frame wrist/elbow landmark histories captured by `PoseTracker`.
- **Defense Game** – `DefenseGame` adds left/right "flying blocks" that must be blocked or dodged. Each outcome increments blocked, dodged, or hit counts and feeds defensive analytics.
- **Comprehensive Metrics** – Reaction times, per-type accuracy, wrist displacement speeds, coverage/exposure percentages, endurance, and avoided punches are calculated and exported.

## Repository Structure

```
Perfect-Punch/
├── mainFile/
│   ├── main.py               # Primary application loop (targets, defense, analytics, JSON export)
│   ├── extractDataPoints.py  # PoseTracker buffering frames & normalized landmarks
│   ├── defense.py            # Flying block defense mini-game reused by main.py
│   └── target_utils.py       # Target spawning, punch color mapping, wrist hit detection
├── models/
│   ├── model_state.pt        # Trained PyTorch weights for punch classification (hook/jab/uppercut)
│   └── linear-classifier.ipynb # Reference notebook for training/evaluation of the classifier
├── session_metrics.json      # Latest session export (overwritten each run)
└── README.md                # Project documentation
```

## Requirements

- Python 3.10 or newer
- Webcam capable of 30 FPS (higher rates improve wrist speed calculations)
- Python packages:
  - `opencv-python`
  - `mediapipe`
  - `torch`
  - `numpy`
  - `pillow` (indirect dependency installed automatically)

> **Tip:** `main.py` tolerates missing pose detections; if MediaPipe loses the athlete (no one or too many people in frame) the drill keeps running and metrics resume once a single pose is reacquired.

## Setup Instructions

1. **Clone or download the repository** and open it in your editor/terminal.
2. **Create a virtual environment (optional but recommended):**
	```powershell
	python -m venv .venv
	.\.venv\Scripts\activate
	```
3. **Install dependencies:**
	```powershell
	pip install opencv-python mediapipe torch numpy
	```
	(Use `pip install -r game_flow/requirements.txt` if you prefer the provided template.)
4. **Verify model weights:** ensure `models/model_state.pt` is present; it ships with the repo.

## Running the Drill

1. Activate your virtual environment (if not already active):
	```powershell
	.\.venv\Scripts\activate
	```
2. Start the application from the project root:
	```powershell
	python mainFile/main.py
	```
3. Stand in view of the webcam. A mirrored window opens showing:
	- Colored circular targets (offense)
	- Moving squares (defense)
4. Begin punching once the one-second shield expires. Block or dodge the squares with your arms/shoulders.
5. The session ends automatically after 30 seconds or when you press `q` to exit early.
6. Review `session_metrics.json` for detailed results (overwritten each run). Example categories:
	- `offense.punch_accuracy`, `offense.punch_reaction_time`, `offense.punch_speed`
	- `defense.critical_hit_opportunities`, `defense.exposure_weights`, `defense.endurance`, `defense.punches_avoided`
	- `miscellaneous.flying_blocks_summary`

## Key Metrics

- **Reaction Time:** average, best, worst, and per-third values computed from circle spawn → contact (excluding the one-second shield).
- **Punch Accuracy:** classifier agreement with the prompted punch type, broken down by type and by 10-second segments.
- **Punch Speed:** wrist displacement (pixels/second) between the first and fifteenth buffered frames of each punch.
- **Critical Hit Opportunities:** percentage of head/body regions uncovered by either arm across all frames.
- **Exposure Weights:** uncovered ratios (0–1) for left/right shoulder, chest, abdomen, and hips.
- **Endurance:** punch accuracy per third of the 30-second session, highlighting fatigue effects.
- **Punches Avoided:** proportion of flying blocks that were blocked or dodged.

## Customization

Adjust constants inside `mainFile/main.py` and `mainFile/defense.py` to tailor the drill:

- `MAX_RUNTIME` – total session length (seconds)
- `SPAWN_PROTECT_S` – protection window before a target becomes hittable
- `SQUARE_SPEED`, `SPAWN_DELAY_RANGE` – flying block behaviour
- Coverage padding values in `update_coverage_metrics` – tweak how strictly arm coverage is judged

## Troubleshooting

- **No pose detected / multiple people:** MediaPipe may fail to lock onto the athlete. Ensure a single person in frame and good lighting; the drill will pause punch classification until the pose stabilises.
- **Slow frame rate:** reduces speed accuracy. Close other applications or lower resolution if necessary.
- **Missing JSON:** check console output; the app prints `Session metrics written to session_metrics.json` on success.

## License & Credits

Add your license terms here. This project depends on OpenCV, MediaPipe, and PyTorch—please acknowledge their licenses in any distribution.

---

Enjoy training with Perfect Punch! Let the metrics guide your combination speed, defensive coverage, and endurance from session to session.

