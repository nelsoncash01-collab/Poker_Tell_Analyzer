"""Footage acquisition: download source video with yt-dlp.

This is a convenience step *before* ingestion — it just puts video files on your
disk. It is deliberately not per-player and not tied to the leakage guard:
downloading is raw-pixel acquisition, and the same file may later be ingested
under one or more players. After downloading, register a file with
``poker-tell ingest-video`` (or ``ingest_video``).

Quality defaults to the best available MP4. Pass ``max_height`` (e.g. 1080) to
cap resolution. Note that merging the best separate video+audio streams into one
MP4 requires the **ffmpeg binary** on your machine (``brew install ffmpeg`` /
``apt install ffmpeg`` / https://ffmpeg.org). Without it, yt-dlp falls back to a
single progressive stream, which YouTube usually caps around 720p.

Downloading from YouTube currently also needs two things in the environment:
- a **JavaScript runtime** (install Deno: ``curl -fsSL https://deno.land/install.sh | sh``)
  so yt-dlp can solve YouTube's stream-URL ("n") challenge; without it only
  thumbnail images are offered and the download fails with "Requested format is
  not available".
- **cookies** when the request comes from a datacenter IP (e.g. a Codespace):
  export a cookies.txt and pass ``cookiefile`` / the CLI ``--cookies``.

Only download footage you have the right to use; respect each site's terms.

``yt_dlp`` is an optional dependency, imported lazily with a clear error if it
is missing.
"""

from __future__ import annotations

from pathlib import Path

# Default output naming: human-readable title plus the stable video id (keeps
# names unique and gives a predictable handle to reuse as a --video-id).
DEFAULT_OUTTMPL = "%(title)s [%(id)s].%(ext)s"


def _require_yt_dlp():
    try:
        import yt_dlp  # noqa: F401
    except ImportError as exc:  # pragma: no cover - only without yt-dlp
        raise ImportError(
            "downloading needs yt-dlp. Install it with `pip install yt-dlp` "
            "(and install the ffmpeg binary for best-quality merged MP4s)."
        ) from exc
    return __import__("yt_dlp")


def build_format_selector(max_height: int | None = None) -> str:
    """Build a yt-dlp ``format`` string.

    Prefers a merged best-video + best-audio MP4, then a progressive MP4, then
    anything — optionally capped to ``max_height`` pixels tall.
    """
    if max_height is None:
        return "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
    h = int(max_height)
    return (
        f"bestvideo[height<={h}][ext=mp4]+bestaudio[ext=m4a]/"
        f"best[height<={h}][ext=mp4]/best[height<={h}]/best"
    )


def build_ydl_opts(
    dest_dir: Path | str,
    *,
    max_height: int | None = None,
    outtmpl: str | None = None,
    quiet: bool = False,
    cookiefile: str | Path | None = None,
    cookies_from_browser: str | None = None,
) -> dict:
    """Assemble the options dict passed to ``yt_dlp.YoutubeDL``.

    ``cookiefile`` points at an exported cookies.txt; ``cookies_from_browser``
    names a local browser to read cookies from (e.g. ``"chrome"``). Cookies let
    YouTube see you as a logged-in human, which is usually required when
    downloading from a datacenter IP (e.g. a Codespace).
    """
    dest = Path(dest_dir)
    opts: dict = {
        "format": build_format_selector(max_height),
        "merge_output_format": "mp4",
        "outtmpl": str(dest / (outtmpl or DEFAULT_OUTTMPL)),
        "quiet": quiet,
        "noprogress": quiet,
        "ignoreerrors": False,
    }
    if cookiefile:
        opts["cookiefile"] = str(cookiefile)
    if cookies_from_browser:
        # yt-dlp expects a (browser, profile, keyring, container) tuple.
        opts["cookiesfrombrowser"] = (cookies_from_browser, None, None, None)
    return opts


def _paths_from_info(info: dict) -> list[Path]:
    """Pull the actual on-disk file path(s) out of a yt-dlp info dict.

    Handles a single video, and a playlist (``entries``). Prefers the
    post-processing ``requested_downloads[*].filepath`` recorded by modern
    yt-dlp.
    """
    if info is None:
        return []
    if "entries" in info and info["entries"] is not None:
        paths: list[Path] = []
        for entry in info["entries"]:
            paths.extend(_paths_from_info(entry))
        return paths
    for dl in info.get("requested_downloads") or []:
        fp = dl.get("filepath") or dl.get("_filename")
        if fp:
            return [Path(fp)]
    fp = info.get("filepath") or info.get("_filename")
    return [Path(fp)] if fp else []


def download_video(
    url: str,
    dest_dir: Path | str,
    *,
    max_height: int | None = None,
    quiet: bool = False,
    cookiefile: str | Path | None = None,
    cookies_from_browser: str | None = None,
    _ydl_cls=None,
) -> list[Path]:
    """Download ``url`` into ``dest_dir``; return the downloaded file path(s).

    A single link yields one path; a playlist link yields one per entry. The
    directory is created if needed. See ``build_ydl_opts`` for the cookie
    options. ``_ydl_cls`` is an injection point for tests; leave it unset to use
    the real ``yt_dlp.YoutubeDL``.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    ydl_cls = _ydl_cls
    if ydl_cls is None:
        ydl_cls = _require_yt_dlp().YoutubeDL
    opts = build_ydl_opts(
        dest, max_height=max_height, quiet=quiet, cookiefile=cookiefile,
        cookies_from_browser=cookies_from_browser,
    )
    with ydl_cls(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    return _paths_from_info(info)


def download_videos(
    urls: list[str],
    dest_dir: Path | str,
    *,
    max_height: int | None = None,
    quiet: bool = False,
    cookiefile: str | Path | None = None,
    cookies_from_browser: str | None = None,
    _ydl_cls=None,
) -> list[Path]:
    """Download several links into ``dest_dir``; return all resulting paths."""
    paths: list[Path] = []
    for url in urls:
        paths.extend(
            download_video(
                url, dest_dir, max_height=max_height, quiet=quiet,
                cookiefile=cookiefile, cookies_from_browser=cookies_from_browser,
                _ydl_cls=_ydl_cls,
            )
        )
    return paths
