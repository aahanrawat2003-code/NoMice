from collections import deque
from dataclasses import dataclass

import cv2
import pyautogui

from src.hand_tracker import HandData
from src.utils import clamp, distance_2d, lerp


@dataclass
class SmoothPoint:
    x: float
    y: float


class MouseGestureController:
    """
    Converts hand landmarks into mouse movement, click, and scroll actions.

    Gestures:
    - Move cursor with index fingertip.
    - Left click with thumb-index pinch.
    - Right click with thumb-middle pinch.
    - Scroll when index and middle fingers are up and moving vertically.
    """

    def __init__(
        self,
        camera_width: int,
        camera_height: int,
        smoothing_factor: float = 0.25,
        movement_margin: int = 100,
        pinch_click_threshold: float = 0.035,
        pinch_right_click_threshold: float = 0.04,
        scroll_sensitivity: int = 30,
    ) -> None:
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.smoothing_factor = smoothing_factor
        self.margin = movement_margin
        self.left_pinch_threshold = pinch_click_threshold
        self.right_pinch_threshold = pinch_right_click_threshold
        self.scroll_sensitivity = scroll_sensitivity

        self.screen_width, self.screen_height = pyautogui.size()

        self.prev_cursor = SmoothPoint(self.screen_width / 2, self.screen_height / 2)
        self.scroll_history = deque(maxlen=5)

        self.left_click_latched = False
        self.right_click_latched = False

        self.current_mode = "Idle"

    def update(self, hand: HandData) -> None:
        """Update mouse state based on latest hand landmarks."""
        lm = hand.landmarks

        # Landmark indices from MediaPipe Hands.
        thumb_tip = lm[4]
        index_tip = lm[8]
        index_pip = lm[6]
        middle_tip = lm[12]
        middle_pip = lm[10]

        # Detect if index/middle are extended (simple heuristic).
        index_extended = index_tip.y < index_pip.y
        middle_extended = middle_tip.y < middle_pip.y

        # Always attempt cursor movement using index tip for a smooth, direct feel.
        self._move_cursor(index_tip.x, index_tip.y)

        left_pinch_dist = distance_2d(thumb_tip.x, thumb_tip.y, index_tip.x, index_tip.y)
        right_pinch_dist = distance_2d(thumb_tip.x, thumb_tip.y, middle_tip.x, middle_tip.y)

        # Scroll mode when both index and middle are extended and pinch gestures are not active.
        if index_extended and middle_extended and left_pinch_dist > self.left_pinch_threshold:
            self._handle_scroll(index_tip.y, middle_tip.y)
            self.current_mode = "Scroll"
        else:
            self.scroll_history.clear()

        # Left click with thumb-index pinch (edge-triggered to avoid repeated clicks).
        if left_pinch_dist < self.left_pinch_threshold and not self.left_click_latched:
            pyautogui.click(button="left")
            self.left_click_latched = True
            self.current_mode = "Left Click"
        elif left_pinch_dist >= self.left_pinch_threshold:
            self.left_click_latched = False

        # Right click with thumb-middle pinch.
        if right_pinch_dist < self.right_pinch_threshold and not self.right_click_latched:
            pyautogui.click(button="right")
            self.right_click_latched = True
            self.current_mode = "Right Click"
        elif right_pinch_dist >= self.right_pinch_threshold:
            self.right_click_latched = False

        # If no click or scroll is active, default mode is move.
        if (
            left_pinch_dist >= self.left_pinch_threshold
            and right_pinch_dist >= self.right_pinch_threshold
            and not (index_extended and middle_extended)
        ):
            self.current_mode = "Move"

    def _move_cursor(self, finger_x: float, finger_y: float) -> None:
        """
        Map normalized camera coordinates to screen coordinates with:
        - active margin region to reduce accidental edge jumps
        - linear interpolation (lerp) for smooth movement
        """
        # Convert normalized [0..1] to pixel coordinates in camera frame.
        cam_x = finger_x * self.camera_width
        cam_y = finger_y * self.camera_height

        # Keep movement inside an interaction region.
        min_x = self.margin
        max_x = self.camera_width - self.margin
        min_y = self.margin
        max_y = self.camera_height - self.margin

        cam_x = clamp(cam_x, min_x, max_x)
        cam_y = clamp(cam_y, min_y, max_y)

        # Map camera region to full screen.
        target_x = (cam_x - min_x) / (max_x - min_x) * self.screen_width
        target_y = (cam_y - min_y) / (max_y - min_y) * self.screen_height

        # Interpolate from previous position to reduce jitter.
        smooth_x = lerp(self.prev_cursor.x, target_x, self.smoothing_factor)
        smooth_y = lerp(self.prev_cursor.y, target_y, self.smoothing_factor)

        self.prev_cursor = SmoothPoint(smooth_x, smooth_y)
        pyautogui.moveTo(smooth_x, smooth_y, _pause=False)

    def _handle_scroll(self, index_y: float, middle_y: float) -> None:
        """
        Scroll based on vertical movement of average index/middle fingertip y.
        Moving fingers up/down generates scroll up/down.
        """
        avg_y = (index_y + middle_y) / 2.0
        self.scroll_history.append(avg_y)

        if len(self.scroll_history) < 2:
            return

        delta = self.scroll_history[-2] - self.scroll_history[-1]
        scroll_amount = int(delta * self.scroll_sensitivity * 100)

        # Ignore tiny noisy movement.
        if abs(scroll_amount) > 1:
            pyautogui.scroll(scroll_amount)

    def draw_interaction_zone(self, frame: cv2.typing.MatLike) -> None:
        """Draw the active camera region used for cursor mapping."""
        cv2.rectangle(
            frame,
            (self.margin, self.margin),
            (self.camera_width - self.margin, self.camera_height - self.margin),
            (100, 255, 100),
            2,
        )
