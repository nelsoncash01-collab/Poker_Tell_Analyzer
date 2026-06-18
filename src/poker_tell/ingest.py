"""Video ingestion: register raw footage and read its true PTS timeline.

This is stage 1 of the pipeline. Its job is *not* to dump frames to disk — it
is to (a) register a raw file as an immutable, hashed ``VideoSource`` in a
per-player manifest, and (b) read the real per-frame presentation timestamps
(PTS) so the sync layer gets a trustworthy ``VideoTimeline`` instead of trusting
a single nominal fps (see ``video.VideoTimeline`` for why that matters on
broadcast footage).

Raw footage is never moved or modified — we record its path and SHA-256.
Full-resolution frames are extracted lazily, only for the ranges that get
synced to hands.

A note on namespacing and shared footage: a broadcast file physically contains
every player at the table. The no-cross-player-transfer rule governs *learned*
content — features, normalization stats, model weights — not raw pixels. So the
same physical file may legitimately be registered under several players'
manifests (the shared SHA-256 lets you recognize it as one underlying file).
What must never be shared is anything downstream of feature extraction.

PyAV (``av``) is an optional dependency; the metadata/timeline functions import
it lazily and raise a clear error if it is missing. The pure-data parts
(``VideoSource``, ``VideoManifest``) work without it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from poker_tell.paths import guard_single_player, player_data_dir, validate_player_id
from poker_tell.video import VideoTimeline


def _require_av():
    try:
        import av  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without PyAV
        raise ImportError(
            "video probing/extraction needs PyAV. Install it with "
            "`pip install av` (wheels bundle ffmpeg)."
        ) from exc
    return __import__("av")


# --- the immutable record produced by ingestion ---------------------------


@dataclass(frozen=True)
class VideoSource:
    """An ingested raw video, owned by one player's namespace.

    ``path`` points at the untouched original. ``sha256`` pins its identity so
    re-runs and cross-namespace duplicates are detectable. Frame-rate fields
    come from the real PTS timeline, and ``is_vfr`` records whether a single-fps
    assumption would have been wrong for this file.
    """

    video_id: str
    player_id: str
    path: str
    sha256: str
    size_bytes: int
    n_frames: int
    duration_s: float
    mean_fps: float
    median_fps: float
    is_vfr: bool
    width: int
    height: int
    codec: str
    container_avg_rate: float | None = None
    # provenance — fill these in so every synced hand traces back to a source.
    source: str = "unspecified"
    air_date: str | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VideoSource":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


# --- hashing + PTS reading -------------------------------------------------


def sha256_file(path: Path | str, *, chunk: int = 1 << 20) -> str:
    """Stream a file through SHA-256 without loading it into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_pts(path: Path | str, *, decode: bool = False) -> tuple[np.ndarray, dict]:
    """Read per-frame presentation timestamps (seconds) for a video.

    By default this *demuxes* packets (fast, no pixel decode) and sorts their
    PTS into display order — correct even with B-frame reordering. Set
    ``decode=True`` to decode every frame instead (slower, but robust to files
    with missing packet PTS). Returns ``(pts_seconds_sorted, stream_meta)``.
    """
    av = _require_av()
    pts_list: list[float] = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"{path}: no video stream")
        vstream = container.streams.video[0]
        time_base = float(vstream.time_base) if vstream.time_base else None
        meta = {
            "width": int(vstream.width or 0),
            "height": int(vstream.height or 0),
            "codec": vstream.codec_context.name,
            "container_avg_rate": (
                float(vstream.average_rate) if vstream.average_rate else None
            ),
        }
        if decode:
            for frame in container.decode(video=0):
                if frame.pts is not None and time_base is not None:
                    pts_list.append(frame.pts * time_base)
        else:
            for packet in container.demux(vstream):
                if packet.pts is not None and time_base is not None:
                    pts_list.append(packet.pts * time_base)

    if not pts_list:
        raise ValueError(
            f"{path}: no frame timestamps found; retry with decode=True"
        )
    pts = np.sort(np.asarray(pts_list, dtype=float))
    # Re-base so the first frame is t=0 (PTS often starts at a nonzero offset).
    pts = pts - pts[0]
    return pts, meta


def probe_video(path: Path | str, *, decode: bool = False) -> tuple[VideoTimeline, dict]:
    """Read the timeline and stream metadata for ``path`` (no manifest write)."""
    pts, meta = read_pts(path, decode=decode)
    timeline = VideoTimeline(pts)
    meta = {
        **meta,
        "n_frames": timeline.n_frames,
        "duration_s": float(timeline.span_seconds(0, timeline.n_frames - 1)),
        "mean_fps": timeline.mean_fps(),
        "median_fps": timeline.median_fps(),
        "is_vfr": timeline.is_vfr(),
    }
    return timeline, meta


# --- the per-player manifest ----------------------------------------------


@dataclass
class VideoManifest:
    """All ``VideoSource`` records for one player, plus their PTS timelines.

    Lives under ``data/<player_id>/video/``. Bound to a single player; adding a
    foreign source raises. Timelines are stored as compressed ``.npy`` files
    alongside the JSON so a ``VideoTimeline`` can be rebuilt without re-probing.
    """

    player_id: str
    root: Path
    sources: dict[str, VideoSource] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_player_id(self.player_id)
        self.root = Path(self.root)

    # -- locations ----------------------------------------------------------

    @property
    def dir(self) -> Path:
        return player_data_dir(self.root, self.player_id) / "video"

    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    def _timeline_path(self, video_id: str) -> Path:
        return self.dir / "timelines" / f"{video_id}.npy"

    # -- mutation -----------------------------------------------------------

    def add(self, source: VideoSource, timeline: VideoTimeline) -> None:
        if source.player_id != self.player_id:
            raise PermissionError(
                f"manifest is bound to player {self.player_id!r}; refusing "
                f"source for {source.player_id!r}"
            )
        if source.video_id in self.sources:
            raise ValueError(f"duplicate video_id {source.video_id!r}")
        guard_single_player(self.player_id, [source.path], root=self.root)
        self._timeline_path(source.video_id).parent.mkdir(parents=True, exist_ok=True)
        np.save(self._timeline_path(source.video_id), timeline.pts_seconds)
        self.sources[source.video_id] = source

    def timeline(self, video_id: str) -> VideoTimeline:
        path = self._timeline_path(video_id)
        if not path.exists():
            raise FileNotFoundError(f"no stored timeline for {video_id!r}")
        return VideoTimeline(np.load(path))

    # -- persistence --------------------------------------------------------

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "player_id": self.player_id,
            "sources": [s.to_dict() for s in self.sources.values()],
        }
        self.manifest_path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, root: Path | str, player_id: str) -> "VideoManifest":
        m = cls(player_id=player_id, root=Path(root))
        if not m.manifest_path.exists():
            return m
        payload = json.loads(m.manifest_path.read_text())
        if payload.get("player_id") != player_id:
            raise PermissionError(
                f"manifest on disk is for {payload.get('player_id')!r}, not "
                f"{player_id!r}"
            )
        for d in payload.get("sources", []):
            src = VideoSource.from_dict(d)
            if src.player_id != player_id:
                raise PermissionError(
                    f"manifest contains foreign source {src.player_id!r}"
                )
            m.sources[src.video_id] = src
        return m


# --- top-level ingest entrypoint ------------------------------------------


def ingest_video(
    root: Path | str,
    player_id: str,
    video_id: str,
    path: Path | str,
    *,
    source: str = "unspecified",
    air_date: str | None = None,
    notes: str = "",
    decode: bool = False,
    save: bool = True,
) -> VideoSource:
    """Probe ``path``, build a ``VideoSource``, and register it for ``player_id``.

    The raw file is left untouched. Returns the new ``VideoSource``; if
    ``save`` is True the player's manifest (and the PTS timeline) is written to
    disk under ``data/<player_id>/video/``.
    """
    validate_player_id(player_id)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    timeline, meta = probe_video(path, decode=decode)
    src = VideoSource(
        video_id=video_id,
        player_id=player_id,
        path=str(path),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        n_frames=meta["n_frames"],
        duration_s=meta["duration_s"],
        mean_fps=meta["mean_fps"],
        median_fps=meta["median_fps"],
        is_vfr=meta["is_vfr"],
        width=meta["width"],
        height=meta["height"],
        codec=meta["codec"],
        container_avg_rate=meta["container_avg_rate"],
        source=source,
        air_date=air_date,
        notes=notes,
    )
    manifest = VideoManifest.load(root, player_id)
    manifest.add(src, timeline)
    if save:
        manifest.save()
    return src


# --- lazy frame extraction (for manual spot-checks / later CV) -------------


def extract_frame(source: VideoSource, frame_index: int) -> np.ndarray:
    """Decode a single frame as an RGB ``ndarray`` (H, W, 3).

    Seeks to the keyframe at/just-before the target PTS then decodes forward —
    frame-accurate, at the cost of decoding a short run of frames. Intended for
    spot-checking sync (eyeball the frame at a street boundary), not for bulk
    extraction.
    """
    return next(iter(extract_frames(source, [frame_index])))


def extract_frames(source: VideoSource, frame_indices: list[int]):
    """Yield RGB frames for the given frame indices, in ascending index order.

    Uses the source's stored timeline implicitly via re-probe of PTS so the
    index->PTS mapping matches ingestion. Yields ``ndarray`` per requested index.
    """
    if not frame_indices:
        return
    av = _require_av()
    pts, _ = read_pts(source.path)
    timeline = VideoTimeline(pts)
    wanted = sorted(set(frame_indices))
    for fi in wanted:
        target_t = timeline.frame_to_time(fi)
        with av.open(source.path) as container:
            stream = container.streams.video[0]
            tb = float(stream.time_base)
            container.seek(int(target_t / tb), stream=stream, any_frame=False)
            out = None
            for frame in container.decode(video=0):
                if frame.pts is None:
                    continue
                t = frame.pts * tb - pts[0]
                if t + 1e-9 >= target_t:
                    out = frame.to_ndarray(format="rgb24")
                    break
            if out is None:
                raise RuntimeError(
                    f"could not decode frame {fi} (t={target_t:.3f}s) from "
                    f"{source.path}"
                )
            yield out
