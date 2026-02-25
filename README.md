# Hand Gesture Mouse Controller (Windows + Webcam)

Control your mouse using hand gestures detected by your laptop webcam.

## Features
- Cursor movement using index fingertip
- Left click with thumb + index pinch
- Right click with thumb + middle pinch
- Vertical two-finger movement for scrolling
- Cursor smoothing to reduce jitter
- On-screen FPS display
- Hand landmarks + connections visualization (MediaPipe)
- Auto-detects screen resolution via PyAutoGUI
- Adjustable sensitivity parameters in code

## Project Structure
- `main.py` - app entry point and webcam loop
- `src/hand_tracker.py` - MediaPipe hand detection wrapper
- `src/gesture_controller.py` - gesture logic and PyAutoGUI actions
- `src/utils.py` - helpers (FPS, smoothing, geometry)
- `requirements.txt` - required libraries

## Installation (Windows)
1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run
```powershell
python main.py
```

Press `q` in the video window to exit.

## Gesture Tuning
Open `main.py` and adjust `MouseGestureController(...)` values:
- `smoothing_factor`: higher = smoother, slower response
- `movement_margin`: larger = less accidental edge jumps
- `pinch_click_threshold`: lower = tighter left-click pinch needed
- `pinch_right_click_threshold`: lower = tighter right-click pinch needed
- `scroll_sensitivity`: higher = faster scrolling

## Notes for Better Real-Time Performance
- Use good lighting and keep your hand visible.
- Keep only one hand in frame.
- Reduce webcam resolution in `main.py` if FPS is low.
- Close heavy background apps if tracking lags.
