"""Frame <-> time conversion for a single video source.

Two implementations of the same ``Timebase`` contract:

- ``FrameClock`` — pure ``frame = seconds * fps`` arithmetic. Correct only for
  genuinely constant-frame-rate (CFR) material. Cheap; good for synthetic data
  and tests.
- ``VideoTimeline`` — backed by the *real* per-frame presentation timestamps
  (PTS) read from the file. This is what to use for broadcast footage, which is
  frequently variable-frame-rate (VFR) or carries an inaccurate container fps;
  trusting a single nominal fps there is a leading cause of the silent sync
  drift this project is built to avoid.

Centralizing frame-rate arithmetic in one place avoids off-by-fps mistakes that
shift every downstream frame boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Timebase(Protocol):
    """Anything that can map between frame indices and seconds for one video.

    ``SyncTable`` depends only on this surface, so it works with a ``FrameClock``
    (CFR) or a ``VideoTimeline`` (real PTS) interchangeably.
    """

    def frame_to_time(self, frame: int) -> float: ...

    def time_to_frame(self, seconds: float, *, mode: str = "nearest") -> int: ...


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
            return math.floor(exact)
        if mode == "ceil":
            return math.ceil(exact)
        if mode == "nearest":
            # Round half up for determinism (banker's rounding would make
            # boundary frames depend on parity, which is surprising here).
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


class VideoTimeline:
    """Frame<->time backed by the real per-frame presentation timestamps (PTS).

    Construct from the seconds-valued PTS of every frame in display order (see
    ``ingest.read_pts``). Unlike ``FrameClock`` this makes no constant-fps
    assumption, so it stays correct on VFR / telecined / inaccurate-fps
    broadcast files. ``time_to_frame`` is a nearest-PTS lookup rather than
    arithmetic.
    """

    def __init__(self, pts_seconds: np.ndarray):
        pts = np.asarray(pts_seconds, dtype=float)
        if pts.ndim != 1 or pts.size == 0:
            raise ValueError("pts_seconds must be a non-empty 1-D array")
        if np.any(np.diff(pts) < 0):
            raise ValueError(
                "pts_seconds must be non-decreasing (sort into display order "
                "before constructing a VideoTimeline)"
            )
        self.pts_seconds = pts

    @property
    def n_frames(self) -> int:
        return int(self.pts_seconds.size)

    def frame_to_time(self, frame: int) -> float:
        if frame < 0 or frame >= self.n_frames:
            raise ValueError(
                f"frame {frame} out of range [0, {self.n_frames - 1}]"
            )
        return float(self.pts_seconds[frame])

    def time_to_frame(self, seconds: float, *, mode: str = "nearest") -> int:
        """Frame index for ``seconds`` via PTS lookup, clamped to range.

        ``floor`` = last frame at or before ``seconds``; ``ceil`` = first frame
        at or after; ``nearest`` = closest PTS (default).
        """
        if seconds < 0:
            raise ValueError(f"seconds must be non-negative, got {seconds}")
        pts = self.pts_seconds
        n = self.n_frames
        if mode == "floor":
            idx = int(np.searchsorted(pts, seconds, side="right")) - 1
            return max(0, idx)
        if mode == "ceil":
            idx = int(np.searchsorted(pts, seconds, side="left"))
            return min(n - 1, idx)
        if mode == "nearest":
            hi = int(np.searchsorted(pts, seconds, side="left"))
            if hi <= 0:
                return 0
            if hi >= n:
                return n - 1
            lo = hi - 1
            # Tie goes to the earlier frame (deterministic).
            return lo if (seconds - pts[lo]) <= (pts[hi] - seconds) else hi
        raise ValueError(f"unknown rounding mode {mode!r}")

    # -- frame-rate diagnostics --------------------------------------------

    def frame_intervals(self) -> np.ndarray:
        """Per-frame durations (seconds between consecutive PTS)."""
        if self.n_frames < 2:
            return np.empty(0, dtype=float)
        return np.diff(self.pts_seconds)

    def median_fps(self) -> float:
        iv = self.frame_intervals()
        if iv.size == 0:
            raise ValueError("need >=2 frames to estimate fps")
        return float(1.0 / np.median(iv))

    def mean_fps(self) -> float:
        if self.n_frames < 2:
            raise ValueError("need >=2 frames to estimate fps")
        span = self.pts_seconds[-1] - self.pts_seconds[0]
        return float((self.n_frames - 1) / span) if span > 0 else float("inf")

    def is_vfr(self, rel_tol: float = 0.02) -> bool:
        """True if frame intervals vary beyond ``rel_tol`` of the median.

        A clean CFR file has near-identical intervals; broadcast VFR/telecine
        shows patterned or jittery intervals. This flag is recorded at ingest so
        downstream code knows whether the single-fps shortcut would have lied.
        """
        iv = self.frame_intervals()
        if iv.size == 0:
            return False
        med = float(np.median(iv))
        if med <= 0:
            return True
        return bool(np.max(np.abs(iv - med)) / med > rel_tol)

    def span_seconds(self, start_frame: int, end_frame: int) -> float:
        """Presentation seconds between the start of two frames (inclusive end
        adds one median interval so a single frame has non-zero duration)."""
        if end_frame < start_frame:
            raise ValueError(
                f"end_frame ({end_frame}) precedes start_frame ({start_frame})"
            )
        base = self.frame_to_time(end_frame) - self.frame_to_time(start_frame)
        iv = self.frame_intervals()
        tail = float(np.median(iv)) if iv.size else 0.0
        return base + tail
