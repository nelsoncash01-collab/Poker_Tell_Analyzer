import math

import pytest

from poker_tell.video import FrameClock


def test_round_trip_at_integer_frames():
    clock = FrameClock(fps=30.0)
    assert clock.frame_to_time(0) == 0.0
    assert clock.frame_to_time(30) == 1.0
    assert clock.time_to_frame(1.0) == 30


def test_rounding_modes():
    clock = FrameClock(fps=30.0)
    # 1.01s -> exact frame 30.3
    assert clock.time_to_frame(1.01, mode="floor") == 30
    assert clock.time_to_frame(1.01, mode="ceil") == 31
    assert clock.time_to_frame(1.01, mode="nearest") == 30
    # halfway rounds up
    assert clock.time_to_frame(0.5 / 30, mode="nearest") == 1


def test_fractional_fps_accumulates_correctly():
    # NTSC: rounding to 30 would drift; keep it exact.
    clock = FrameClock(fps=29.97)
    t = clock.frame_to_time(29970)
    assert math.isclose(t, 1000.0, rel_tol=1e-9)


def test_duration_is_inclusive():
    clock = FrameClock(fps=30.0)
    # frames 0..29 inclusive = 30 frames = 1 second
    assert math.isclose(clock.duration_seconds(0, 29), 1.0)


def test_invalid_fps_and_negative_inputs():
    with pytest.raises(ValueError):
        FrameClock(fps=0)
    clock = FrameClock(fps=30.0)
    with pytest.raises(ValueError):
        clock.frame_to_time(-1)
    with pytest.raises(ValueError):
        clock.time_to_frame(-0.5)
    with pytest.raises(ValueError):
        clock.duration_seconds(10, 5)
