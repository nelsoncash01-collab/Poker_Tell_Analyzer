"""Frame <-> time conversion for a single video source.

A ``FrameClock`` is the only place frame-rate arithmetic should live. Keeping it
centralized avoids the classic off-by-fps mistakes that silently shift every
downstream frame boundary (a leading cause of sync drift).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameClock:
    """Converts between frame indices and wall-clock seconds for one video.

    Frames are 0-indexed. ``fps`` is frames per second (may be fractional, e.g.
    29.97 for NTSC broadcast footage — getting this exactly right matters over
    a long episode, where rounding to 30 accumulates seconds of drift).
    """

    fps: float

    def __post_init__(self) -> None:
        if not self.fps > 0:
            raise ValueError(f"fps must be positive, got {self.fps!r}")

    def frame_to_time(self, frame: int) -> float:
        """Seconds elapsed at the start of ``frame``."""
        if frame < 0:
            raise ValueError(f"frame must be non-negative, got {frame}")
        return frame / self.fps

    def time_to_frame(self, seconds: float, *, mode: str = "nearest") -> int:
        """Frame index containing ``seconds``.

        ``mode`` controls rounding: ``"floor"`` (frame currently showing),
        ``"ceil"`` (next frame boundary), or ``"nearest"`` (default).
        """
        if seconds < 0:
            raise ValueError(f"seconds must be non-negative, got {seconds}")
        exact = seconds * self.fps
        if mode == "floor":
            import math

            return math.floor(exact)
        if mode == "ceil":
            import math

            return math.ceil(exact)
        if mode == "nearest":
            # Round half up for determinism (banker's rounding would make
            # boundary frames depend on parity, which is surprising here).
            import math

            return math.floor(exact + 0.5)
        raise ValueError(f"unknown rounding mode {mode!r}")

    def duration_seconds(self, start_frame: int, end_frame: int) -> float:
        """Wall-clock seconds spanned by the inclusive frame range."""
        if end_frame < start_frame:
            raise ValueError(
                f"end_frame ({end_frame}) precedes start_frame ({start_frame})"
            )
        # +1 because the range is inclusive of both endpoints.
        return (end_frame - start_frame + 1) / self.fps
