import numpy as np
import pytest

from poker_tell.video import FrameClock, Timebase, VideoTimeline


def cfr_pts(fps, n):
    return np.arange(n) / fps


def test_satisfies_timebase_protocol():
    tl = VideoTimeline(cfr_pts(25.0, 10))
    assert isinstance(tl, Timebase)
    assert isinstance(FrameClock(25.0), Timebase)


def test_frame_to_time_and_back_cfr():
    tl = VideoTimeline(cfr_pts(25.0, 100))
    assert tl.frame_to_time(0) == 0.0
    assert tl.frame_to_time(25) == pytest.approx(1.0)
    assert tl.time_to_frame(1.0) == 25
    assert tl.n_frames == 100


def test_time_to_frame_modes():
    tl = VideoTimeline(cfr_pts(10.0, 10))  # pts = 0.0, 0.1, ..., 0.9
    # 0.24s sits between frame 2 (0.2) and 3 (0.3)
    assert tl.time_to_frame(0.24, mode="floor") == 2
    assert tl.time_to_frame(0.24, mode="ceil") == 3
    assert tl.time_to_frame(0.24, mode="nearest") == 2
    assert tl.time_to_frame(0.26, mode="nearest") == 3
    # out-of-range clamps
    assert tl.time_to_frame(99.0, mode="nearest") == 9
    assert tl.time_to_frame(0.0, mode="floor") == 0


def test_nearest_tie_breaks_to_earlier_frame():
    tl = VideoTimeline(cfr_pts(10.0, 5))
    # exactly halfway between frame 1 (0.1) and 2 (0.2)
    assert tl.time_to_frame(0.15, mode="nearest") == 1


def test_cfr_is_not_flagged_vfr():
    tl = VideoTimeline(cfr_pts(30.0, 300))
    assert tl.is_vfr() is False
    assert tl.median_fps() == pytest.approx(30.0)
    assert tl.mean_fps() == pytest.approx(30.0)


def test_vfr_is_flagged():
    # Frame intervals jitter well beyond the 2% tolerance.
    pts = np.array([0.0, 0.04, 0.04 + 0.06, 0.04 + 0.06 + 0.033, 0.25])
    tl = VideoTimeline(pts)
    assert tl.is_vfr() is True


def test_rejects_non_monotonic_pts():
    with pytest.raises(ValueError):
        VideoTimeline(np.array([0.0, 0.1, 0.05]))


def test_rejects_empty():
    with pytest.raises(ValueError):
        VideoTimeline(np.array([]))


def test_out_of_range_frame_raises():
    tl = VideoTimeline(cfr_pts(25.0, 10))
    with pytest.raises(ValueError):
        tl.frame_to_time(10)
    with pytest.raises(ValueError):
        tl.frame_to_time(-1)
