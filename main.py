import cv2
import pyautogui

from src.gesture_controller import MouseGestureController
from src.hand_tracker import HandTracker
from src.utils import FPSCounter


def draw_legend(frame: cv2.typing.MatLike) -> None:
    """Draw a small on-screen legend for gesture controls."""
    legend_lines = [
        "Legend",
        "Index finger: Move cursor",
        "Thumb + Index pinch: Left click",
        "Thumb + Middle pinch: Right click",
        "Index + Middle move up/down: Scroll",
        "Press Q: Quit",
    ]

    x1, y1 = 700, 20
    x2, y2 = 1260, 210

    # Filled rectangle background for readability.
    cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 30), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 2)

    for i, text in enumerate(legend_lines):
        color = (0, 255, 255) if i == 0 else (240, 240, 240)
        scale = 0.7 if i == 0 else 0.55
        thickness = 2 if i == 0 else 1
        y = y1 + 30 + (i * 28)
        cv2.putText(
            frame,
            text,
            (x1 + 12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def main() -> None:
    """
    Entry point for the hand-gesture mouse controller.

    Steps:
    1. Capture frames from webcam.
    2. Detect hand landmarks with MediaPipe.
    3. Convert gestures into mouse actions.
    4. Draw overlay info and display FPS.
    """
    # Use default webcam (0). On some laptops, try 1 if 0 does not work.
    cap = cv2.VideoCapture(0)

    # Reduce internal buffering for lower latency (when supported).
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # A common webcam resolution that balances speed and quality.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions and device availability.")

    # Disable PyAutoGUI fail-safe so cursor can reach all edges without exceptions.
    # If you prefer safety, set this back to True and avoid top-left corner.
    pyautogui.FAILSAFE = False

    tracker = HandTracker(
        max_num_hands=1,
        detection_confidence=0.7,
        tracking_confidence=0.7,
    )

    controller = MouseGestureController(
        camera_width=1280,
        camera_height=720,
        smoothing_factor=0.25,
        movement_margin=120,
        pinch_click_threshold=0.035,
        pinch_right_click_threshold=0.04,
        scroll_sensitivity=35,
    )

    fps_counter = FPSCounter()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Mirror image so movement feels natural (like looking in a mirror).
            frame = cv2.flip(frame, 1)

            frame, hand_result = tracker.process(frame)

            # Convert hand landmarks into mouse actions.
            if hand_result is not None:
                controller.update(hand_result)

            # Draw guide rectangle for active control region.
            controller.draw_interaction_zone(frame)

            # Update and draw FPS.
            fps = fps_counter.update()
            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Display current mode/state for debugging and learning.
            cv2.putText(
                frame,
                f"Mode: {controller.current_mode}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            draw_legend(frame)
            cv2.imshow("Hand Gesture Mouse Controller", frame)

            # Press 'q' to quit.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


if __name__ == "__main__":
    main()
