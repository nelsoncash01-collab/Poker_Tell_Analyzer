"""CLI tests — invoke main([...]) directly.

The video commands need PyAV (skipped if absent). Hand-history and ls commands
run without it.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from poker_tell.cli import main
from poker_tell.hand_ingest import HandHistoryStore
from poker_tell.ingest import VideoManifest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_ingest_hands_json_cli(tmp_path):
    rc = main(["--root", str(tmp_path), "ingest-hands",
               "--player", "negreanu", "--path", str(EXAMPLES / "hands.json")])
    assert rc == 0
    store = HandHistoryStore.load(tmp_path, "negreanu")
    assert len(store) == 2


def test_ingest_hands_csv_cli(tmp_path):
    rc = main(["--root", str(tmp_path), "ingest-hands",
               "--player", "negreanu", "--path", str(EXAMPLES / "hands.csv")])
    assert rc == 0
    assert len(HandHistoryStore.load(tmp_path, "negreanu")) == 2


def test_ls_cli(tmp_path, capsys):
    main(["--root", str(tmp_path), "ingest-hands", "--player", "negreanu",
          "--path", str(EXAMPLES / "hands.json")])
    rc = main(["--root", str(tmp_path), "ls", "--player", "negreanu"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "player: negreanu" in out
    assert "hands: 2" in out


def test_ingest_video_missing_args_returns_error(tmp_path):
    # No --catalog and no --video-id/--path -> usage error code 2.
    rc = main(["--root", str(tmp_path), "ingest-video", "--player", "negreanu"])
    assert rc == 2


def test_foreign_player_reported_cleanly(tmp_path, capsys):
    # hands.csv contains negreanu hands; ingesting under ivey hits the leakage
    # guard and should print a clean error (no traceback) with exit code 1.
    rc = main(["--root", str(tmp_path), "ingest-hands", "--player", "ivey",
               "--path", str(EXAMPLES / "hands.csv")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "foreign_player" in err


# --- PyAV-backed video commands -------------------------------------------

av = pytest.importorskip("av")


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=25)
    stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
    for i in range(15):
        arr = np.full((48, 64, 3), (i * 13) % 256, dtype=np.uint8)
        for pkt in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    return path


def test_ingest_video_single_cli(tmp_path, clip):
    rc = main(["--root", str(tmp_path), "ingest-video", "--player", "negreanu",
               "--video-id", "v1", "--path", str(clip),
               "--source", "Test S1E1"])
    assert rc == 0
    vm = VideoManifest.load(tmp_path, "negreanu")
    assert "v1" in vm.sources
    assert vm.sources["v1"].source == "Test S1E1"


def test_ingest_video_catalog_cli(tmp_path, clip):
    catalog = tmp_path / "cat.csv"
    catalog.write_text(
        "video_id,path,source,air_date,notes\n"
        f"v1,{clip},Test S1E1,2006-01-16,\n"
        f"v2,{clip},Test S1E2,2006-01-23,note\n"
    )
    rc = main(["--root", str(tmp_path), "ingest-video", "--player", "negreanu",
               "--catalog", str(catalog)])
    assert rc == 0
    vm = VideoManifest.load(tmp_path, "negreanu")
    assert set(vm.sources) == {"v1", "v2"}


def test_ingest_video_json_catalog_cli(tmp_path, clip):
    catalog = tmp_path / "cat.json"
    catalog.write_text(json.dumps([
        {"video_id": "v1", "path": str(clip), "source": "Test"},
    ]))
    rc = main(["--root", str(tmp_path), "ingest-video", "--player", "negreanu",
               "--catalog", str(catalog)])
    assert rc == 0
    assert "v1" in VideoManifest.load(tmp_path, "negreanu").sources
