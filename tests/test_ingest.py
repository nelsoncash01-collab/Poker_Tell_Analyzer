"""Ingestion tests.

The integration tests need PyAV; they are skipped if it is not installed so the
rest of the suite still runs. When PyAV is present they exercise the real path:
encode a tiny clip, probe its true PTS, register it, round-trip the manifest,
and decode a frame back.
"""

import numpy as np
import pytest

from poker_tell.ingest import (
    VideoManifest,
    VideoSource,
    extract_frame,
    ingest_video,
    probe_video,
)

av = pytest.importorskip("av")


FPS = 25
N = 20
W, H = 64, 48


@pytest.fixture
def clip(tmp_path):
    """A short CFR H.264 clip whose frame i is a solid shade of i."""
    path = tmp_path / "clip.mp4"
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
    for i in range(N):
        arr = np.full((H, W, 3), (i * 11) % 256, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    return path


def test_probe_reads_real_timeline(clip):
    timeline, meta = probe_video(clip)
    assert timeline.n_frames == N
    assert meta["median_fps"] == pytest.approx(FPS, rel=1e-3)
    assert meta["is_vfr"] is False
    assert (meta["width"], meta["height"]) == (W, H)
    assert "h264" in meta["codec"]


def test_ingest_writes_manifest_and_timeline(tmp_path, clip):
    src = ingest_video(
        root=tmp_path,
        player_id="negreanu",
        video_id="hsp_s1_e1",
        path=clip,
        source="High Stakes Poker S1E1",
        air_date="2006-01-16",
    )
    assert src.player_id == "negreanu"
    assert src.n_frames == N
    assert len(src.sha256) == 64
    assert not src.is_vfr

    # Manifest + timeline persisted under the player's namespace.
    manifest = VideoManifest.load(tmp_path, "negreanu")
    assert "hsp_s1_e1" in manifest.sources
    reloaded = manifest.sources["hsp_s1_e1"]
    assert reloaded.source == "High Stakes Poker S1E1"
    assert reloaded.air_date == "2006-01-16"

    tl = manifest.timeline("hsp_s1_e1")
    assert tl.n_frames == N


def test_manifest_rejects_foreign_player(tmp_path):
    m = VideoManifest(player_id="negreanu", root=tmp_path)
    foreign = VideoSource(
        video_id="x", player_id="ivey", path=str(tmp_path / "f.mp4"),
        sha256="0" * 64, size_bytes=1, n_frames=1, duration_s=1.0,
        mean_fps=25.0, median_fps=25.0, is_vfr=False, width=W, height=H,
        codec="h264",
    )
    from poker_tell.video import VideoTimeline

    with pytest.raises(PermissionError):
        m.add(foreign, VideoTimeline(np.array([0.0, 0.04])))


def test_duplicate_video_id_rejected(tmp_path, clip):
    ingest_video(tmp_path, "negreanu", "dup", clip)
    with pytest.raises(ValueError):
        ingest_video(tmp_path, "negreanu", "dup", clip)


def test_extract_frame_round_trips_content(clip):
    # Ingest (no save needed) to get a VideoSource, then decode frames back.
    src = ingest_video(
        root=clip.parent, player_id="negreanu", video_id="c", path=clip,
        save=False,
    )
    # Frame i was encoded as a solid shade (i*11)%256. H.264 is lossy, so
    # allow tolerance, but distinct frames must remain distinguishable.
    f0 = extract_frame(src, 0)
    f10 = extract_frame(src, 10)
    assert f0.shape == (H, W, 3)
    assert abs(int(f0.mean()) - 0) < 12
    assert abs(int(f10.mean()) - (10 * 11) % 256) < 12
