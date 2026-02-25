import math
import time


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp a number to [min_value, max_value]."""
    return max(min_value, min(value, max_value))


def lerp(start: float, end: float, t: float) -> float:
    """Linear interpolation between start and end using factor t (0..1)."""
    return start + (end - start) * t


def distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two 2D points."""
    return math.hypot(x2 - x1, y2 - y1)


class FPSCounter:
    """
    Lightweight FPS counter.
    Uses reciprocal of frame time for near-real-time FPS feedback.
    """

    def __init__(self) -> None:
        self.prev_time = time.perf_counter()
        self.fps = 0.0

    def update(self) -> float:
        current = time.perf_counter()
        delta = current - self.prev_time
        self.prev_time = current

        if delta > 0:
            # Smooth displayed FPS so numbers do not flicker heavily.
            instant_fps = 1.0 / delta
            self.fps = (0.85 * self.fps) + (0.15 * instant_fps)

        return self.fps
