import numpy as np
import pytest

from poker_tell.broadcast.detect import Box
from poker_tell.broadcast.formats import (
    FormatProfile,
    profiles_differ,
    segment_profiles,
    track_formats,
)


def _profile(hue, x=30, frame=(480, 854)):
    return FormatProfile(pot_color_hsv=(hue, 200, 150),
                         pot_box=Box(x, 18, 160, 44), scale=160.0,
                         frame_shape=frame)


# --- pure logic ------------------------------------------------------------

def test_profiles_differ_on_color():
    a, b = _profile(0), _profile(120)  # red vs blue hue
    assert profiles_differ(a, b)
    assert not profiles_differ(a, _profile(2))  # tiny hue drift = same


def test_profiles_differ_on_location():
    a, b = _profile(0, x=30), _profile(0, x=500)  # same colour, moved banner
    assert profiles_differ(a, b)


def test_segment_profiles_splits_on_change():
    a, b = _profile(0), _profile(120)
    obs = [(0, a), (10, a), (20, b), (30, b)]
    segs = segment_profiles(obs)
    assert len(segs) == 2
    assert (segs[0].start_frame, segs[0].end_frame) == (0, 10)
    assert (segs[1].start_frame, segs[1].end_frame) == (20, 30)


def test_segment_profiles_none_extends_current():
    a = _profile(0)
    segs = segment_profiles([(0, a), (10, None), (20, a)])
    assert len(segs) == 1
    assert segs[0].end_frame == 20


def test_segment_profiles_empty():
    assert segment_profiles([]) == []


# --- track_formats over a real synthetic two-format clip -------------------

av = pytest.importorskip("av")


@pytest.fixture
def two_format_clip(tmp_path):
    """40-frame clip: first 20 frames red, last 20 blue (RGB)."""
    path = tmp_path / "fmt.mp4"
    c = av.open(str(path), "w")
    s = c.add_stream("libx264", rate=25)
    s.width, s.height, s.pix_fmt = 160, 120, "yuv420p"
    for i in range(40):
        rgb = (np.array([200, 0, 0]) if i < 20 else np.array([0, 0, 200]))
        arr = np.full((120, 160, 3), rgb, dtype=np.uint8)
        for pkt in s.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
            c.mux(pkt)
    for pkt in s.encode():
        c.mux(pkt)
    c.close()
    return path


def test_track_formats_finds_the_switch(two_format_clip):
    from types import SimpleNamespace

    def fake_observe(frame_bgr):
        # frame_bgr is BGR; decide red vs blue and return distinct profiles
        b = float(frame_bgr[..., 0].mean())
        r = float(frame_bgr[..., 2].mean())
        if r >= b:
            return _profile(0, x=30, frame=frame_bgr.shape[:2])
        return _profile(120, x=120, frame=frame_bgr.shape[:2])

    source = SimpleNamespace(path=str(two_format_clip))
    segments = track_formats(source, sample_seconds=0.16, observe=fake_observe)
    assert len(segments) == 2
    hues = {seg.profile.pot_color_hsv[0] for seg in segments}
    assert hues == {0, 120}
