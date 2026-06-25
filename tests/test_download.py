from pathlib import Path

import pytest

from poker_tell.cli import main
from poker_tell.download import (
    build_format_selector,
    build_ydl_opts,
    download_video,
    download_videos,
)


# --- pure format/option building -------------------------------------------

def test_format_selector_best_by_default():
    sel = build_format_selector()
    assert sel.startswith("bestvideo[ext=mp4]+bestaudio")
    assert "height" not in sel


def test_format_selector_caps_height():
    sel = build_format_selector(1080)
    assert "height<=1080" in sel
    # falls back through progressive mp4 then anything
    assert sel.endswith("/best")


def test_ydl_opts_target_dest_and_mp4(tmp_path):
    opts = build_ydl_opts(tmp_path, max_height=720)
    assert opts["merge_output_format"] == "mp4"
    assert str(tmp_path) in opts["outtmpl"]
    assert "%(id)s" in opts["outtmpl"]
    assert "height<=720" in opts["format"]
    # No cookie keys unless asked for.
    assert "cookiefile" not in opts
    assert "cookiesfrombrowser" not in opts
    # Remote components default-enabled so YouTube works out of the box.
    assert opts["remote_components"] == ["ejs:github"]


def test_ydl_opts_cookies(tmp_path):
    opts = build_ydl_opts(tmp_path, cookiefile="cookies.txt",
                          cookies_from_browser="chrome")
    assert opts["cookiefile"] == "cookies.txt"
    assert opts["cookiesfrombrowser"] == ("chrome", None, None, None)


def test_ydl_opts_remote_components_can_be_disabled(tmp_path):
    opts = build_ydl_opts(tmp_path, remote_components=[])
    assert "remote_components" not in opts


# --- orchestration with an injected fake YoutubeDL (no network) ------------

class FakeYDL:
    """Stand-in for yt_dlp.YoutubeDL: records opts, fakes a download."""

    last_opts = None

    def __init__(self, opts):
        FakeYDL.last_opts = opts
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download):
        assert download is True
        # Simulate the file yt-dlp would have written into the dest dir.
        dest = Path(self.opts["outtmpl"]).parent
        out = dest / "Some Poker Episode [abc123].mp4"
        out.write_bytes(b"fake")
        return {
            "title": "Some Poker Episode",
            "id": "abc123",
            "requested_downloads": [{"filepath": str(out)}],
        }


def test_download_video_returns_path(tmp_path):
    paths = download_video("https://example/watch?v=abc123", tmp_path,
                           _ydl_cls=FakeYDL)
    assert len(paths) == 1
    assert paths[0].exists()
    assert paths[0].suffix == ".mp4"
    assert FakeYDL.last_opts["format"]  # opts were built and passed through


def test_download_videos_maps_over_urls(tmp_path):
    paths = download_videos(["u1", "u2"], tmp_path, max_height=1080,
                            _ydl_cls=FakeYDL)
    assert len(paths) == 2


class FakePlaylistYDL(FakeYDL):
    def extract_info(self, url, download):
        dest = Path(self.opts["outtmpl"]).parent
        entries = []
        for vid in ("a", "b", "c"):
            out = dest / f"vid_{vid}.mp4"
            out.write_bytes(b"x")
            entries.append({"requested_downloads": [{"filepath": str(out)}]})
        return {"entries": entries}


def test_download_handles_playlist(tmp_path):
    paths = download_video("https://example/playlist", tmp_path,
                           _ydl_cls=FakePlaylistYDL)
    assert len(paths) == 3


# --- CLI -------------------------------------------------------------------

def test_download_cli_bad_url_reports_cleanly(tmp_path, capsys):
    # Real yt-dlp with a bogus URL fails; CLI should print a clean error, not a
    # traceback, and exit 1.
    rc = main(["download", "--url", "not-a-real-url://x", "--dest",
               str(tmp_path / "dl"), "--quiet"])
    assert rc == 1
    assert "error:" in capsys.readouterr().err
