"""``poker-tell`` — command-line entrypoint for ingesting footage and hands.

Everything is per-player and reference-in-place: raw video files are never
copied or moved, only recorded by absolute path + SHA-256. Run ``poker-tell
--help`` (or ``python -m poker_tell.cli --help``) for usage.

Subcommands:
- ``ingest-video`` — register one file, or batch-register from a catalog.
- ``ingest-hands`` — load hands from JSON/CSV file(s).
- ``ls`` — show what has been ingested for a player.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from poker_tell.download import download_videos
from poker_tell.hand_ingest import ingest_hand_history, label_source_summary
from poker_tell.ingest import VideoManifest, ingest_video

def _read_video_catalog(path: Path) -> list[dict]:
    """Parse a video catalog (CSV or JSON) into a list of entry dicts.

    Each entry needs at least ``video_id`` and ``path``; ``source``,
    ``air_date``, and ``notes`` are optional.
    """
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        entries = payload["videos"] if isinstance(payload, dict) else payload
    elif path.suffix.lower() == ".csv":
        with open(path, newline="") as f:
            entries = list(csv.DictReader(f))
    else:
        raise ValueError(f"catalog must be .csv or .json, got {path.suffix!r}")
    for e in entries:
        if not e.get("video_id") or not e.get("path"):
            raise ValueError(f"catalog entry missing video_id/path: {e!r}")
    return entries


def _cmd_ingest_video(args: argparse.Namespace) -> int:
    if args.catalog:
        entries = _read_video_catalog(Path(args.catalog))
        for e in entries:
            src = ingest_video(
                root=args.root,
                player_id=args.player,
                video_id=e["video_id"],
                path=e["path"],
                source=e.get("source") or "unspecified",
                air_date=e.get("air_date") or None,
                notes=e.get("notes") or "",
                decode=args.decode,
            )
            print(f"ingested video {src.video_id!r}: {src.n_frames} frames, "
                  f"{src.median_fps:.3f} fps"
                  f"{' [VFR]' if src.is_vfr else ''}")
        print(f"done: {len(entries)} video(s) for {args.player!r}")
        return 0

    if not args.video_id or not args.path:
        print("error: provide --video-id and --path, or --catalog",
              file=sys.stderr)
        return 2
    src = ingest_video(
        root=args.root,
        player_id=args.player,
        video_id=args.video_id,
        path=args.path,
        source=args.source,
        air_date=args.air_date,
        notes=args.notes,
        decode=args.decode,
    )
    print(f"ingested video {src.video_id!r}: {src.n_frames} frames, "
          f"{src.median_fps:.3f} fps{' [VFR]' if src.is_vfr else ''}, "
          f"sha256={src.sha256[:12]}…")
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    try:
        paths = download_videos(
            args.url, args.dest, max_height=args.max_height, quiet=args.quiet,
        )
    except ImportError:
        raise  # handled by main() with its install hint
    except Exception as exc:  # yt-dlp DownloadError, network issues, etc.
        raise ValueError(f"download failed: {exc}") from exc
    if not paths:
        print("warning: no files were downloaded", file=sys.stderr)
        return 1
    for p in paths:
        print(f"downloaded {p}")
    print(f"done: {len(paths)} file(s) into {args.dest}")
    print("next: register a file with `poker-tell ingest-video --player <id> "
          "--video-id <id> --path <file>`")
    return 0


def _cmd_ingest_hands(args: argparse.Namespace) -> int:
    total_new = 0
    for p in args.path:
        store, issues = ingest_hand_history(
            root=args.root,
            player_id=args.player,
            path=p,
            format=args.format,
        )
        warnings = [i for i in issues if i.severity == "warning"]
        print(f"ingested hands from {p}: store now holds {len(store)} hand(s)")
        for w in warnings:
            hid = f" ({w.hand_id})" if w.hand_id else ""
            print(f"  warning [{w.code}]{hid}: {w.message}", file=sys.stderr)
        total_new += 1
    # Report the labeling-source picture for the whole store.
    from poker_tell.hand_ingest import HandHistoryStore

    store = HandHistoryStore.load(args.root, args.player)
    summ = label_source_summary(store.hands())
    print(f"label source: {summ['n_revealed_hole_cards']}/{summ['n_hands']} "
          f"hands have revealed hole cards "
          f"({summ['revealed_fraction']:.0%}); "
          f"{summ['n_unrevealed']} rely on showdown only "
          f"(selection-bias caveat applies)")
    return 0


def _cmd_ls(args: argparse.Namespace) -> int:
    from poker_tell.hand_ingest import HandHistoryStore

    vm = VideoManifest.load(args.root, args.player)
    print(f"player: {args.player}")
    print(f"videos: {len(vm.sources)}")
    for s in vm.sources.values():
        print(f"  - {s.video_id}: {s.n_frames} frames, {s.median_fps:.3f} fps"
              f"{' [VFR]' if s.is_vfr else ''} — {s.source}")
    hs = HandHistoryStore.load(args.root, args.player)
    summ = label_source_summary(hs.hands())
    print(f"hands: {summ['n_hands']} "
          f"({summ['n_revealed_hole_cards']} with revealed hole cards, "
          f"{summ['revealed_fraction']:.0%})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="poker-tell",
        description="Per-individual poker tell analysis — ingestion CLI.",
    )
    parser.add_argument(
        "--root", default=".",
        help="project root holding data/<player_id>/ (default: cwd)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pd = sub.add_parser("download", help="download source video with yt-dlp")
    pd.add_argument("--url", required=True, action="append",
                    help="video/playlist URL (repeat for several)")
    pd.add_argument("--dest", required=True, help="directory to download into")
    pd.add_argument("--max-height", type=int, default=None,
                    help="cap resolution, e.g. 1080 (default: best available)")
    pd.add_argument("--quiet", action="store_true")
    pd.set_defaults(func=_cmd_download)

    pv = sub.add_parser("ingest-video", help="register video footage")
    pv.add_argument("--player", required=True)
    pv.add_argument("--video-id", help="id for a single video")
    pv.add_argument("--path", help="path to a single video file")
    pv.add_argument("--catalog", help="CSV/JSON catalog for batch ingest")
    pv.add_argument("--source", default="unspecified")
    pv.add_argument("--air-date", default=None)
    pv.add_argument("--notes", default="")
    pv.add_argument("--decode", action="store_true",
                    help="decode every frame for PTS (slower, more robust)")
    pv.set_defaults(func=_cmd_ingest_video)

    ph = sub.add_parser("ingest-hands", help="load hand histories (JSON/CSV)")
    ph.add_argument("--player", required=True)
    ph.add_argument("--path", required=True, nargs="+",
                    help="one or more JSON/CSV hand-history files")
    ph.add_argument("--format", default="auto", choices=["auto", "json", "csv"])
    ph.set_defaults(func=_cmd_ingest_hands)

    pl = sub.add_parser("ls", help="show what's ingested for a player")
    pl.add_argument("--player", required=True)
    pl.set_defaults(func=_cmd_ls)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, PermissionError, FileNotFoundError, ImportError) as exc:
        # Expected, user-facing failures (bad input, leakage guard, missing
        # file, missing optional dependency) — report cleanly rather than
        # dumping a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
