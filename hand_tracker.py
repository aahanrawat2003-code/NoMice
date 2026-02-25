from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


@dataclass
class LandmarkPoint:
    """Simple container for normalized landmark coordinates."""

    x: float
    y: float
    z: float


@dataclass
class HandData:
    """Holds all 21 landmarks for one hand in normalized coordinates."""

    landmarks: list[LandmarkPoint]


class HandTracker:
    """
    Wraps MediaPipe Hands to keep main loop clean and beginner-friendly.
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.7,
    ) -> None:
        model_path = self._ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=tracking_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.hands = vision.HandLandmarker.create_from_options(options)
        self.connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

    def _ensure_model(self) -> Path:
        """Make sure the hand landmarker model exists locally."""
        assets_dir = Path(__file__).resolve().parent.parent / "assets"
        assets_dir.mkdir(exist_ok=True)
        model_path = assets_dir / "hand_landmarker.task"

        if model_path.exists():
            return model_path

        model_url = (
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        )
        urlretrieve(model_url, model_path)
        return model_path

    def process(self, frame: cv2.typing.MatLike) -> Tuple[cv2.typing.MatLike, Optional[HandData]]:
        """
        Process one frame and return:
        - frame with hand landmarks drawn
        - HandData for first detected hand (or None)
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        results = self.hands.detect_for_video(mp_image, timestamp_ms=timestamp_ms)

        if not results.hand_landmarks:
            return frame, None

        hand_landmarks = results.hand_landmarks[0]
        self._draw_landmarks(frame, hand_landmarks)

        landmarks = [
            LandmarkPoint(lm.x, lm.y, lm.z)
            for lm in hand_landmarks
        ]

        return frame, HandData(landmarks=landmarks)

    def _draw_landmarks(self, frame: cv2.typing.MatLike, hand_landmarks: list) -> None:
        """Draw landmark points and connection lines with OpenCV."""
        height, width = frame.shape[:2]

        for connection in self.connections:
            start = hand_landmarks[connection.start]
            end = hand_landmarks[connection.end]
            x1, y1 = int(start.x * width), int(start.y * height)
            x2, y2 = int(end.x * width), int(end.y * height)
            cv2.line(frame, (x1, y1), (x2, y2), (100, 220, 100), 2)

        for landmark in hand_landmarks:
            x, y = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    def close(self) -> None:
        """Release MediaPipe resources cleanly."""
        self.hands.close()
