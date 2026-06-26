"""``poker-tell`` — command-line entrypoint for ingesting footage and hands.

Everything is per-player and reference-in-place: raw video files are never
copied or moved, only recorded by absolute path + SHA-256. Run ``poker-tell
--help`` (or ``python -m poker_tell.cli --help``) for usage.

Subcommands:
- ``download`` — fetch source video with yt-dlp.
- ``ingest-video`` — register one file, or batch-register from a catalog.
- ``ingest-hands`` — load hands from JSON/CSV file(s).
- ``ls`` — show what has been ingested for a player.
- ``read-frame`` — OCR one frame's overlays (pot / board / name plates).
- ``dump-regions`` — save the overlay crops of one frame for layout calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from poker_tell.download import download_videos
from poker_tell.hand_ingest import ingest_hand_history, label_source_summary
from poker_tell.ingest import VideoManifest, ingest_video


def _parse_timestamp(text: str) -> float:
    """Parse ``HH:MM:SS`` / ``MM:SS`` / plain seconds into float seconds."""
    text = text.strip()
    if ":" in text:
        parts = [float(p) for p in text.split(":")]
        seconds = 0.0
        for p in parts:
            seconds = seconds * 60 + p
        return seconds
    return float(text)


def _resolve_frame(args) -> "tuple":
    """Resolve (frame_rgb, video_seconds) for read-frame / dump-regions.

    Accepts either ``--path FILE`` or ``--video-id ID --player P`` (looked up in
    the player's manifest). Reuses the ingest frame-extraction path.
    """
    from poker_tell.ingest import extract_frames, probe_video

    seconds = _parse_timestamp(args.at)
    if args.path:
        timeline, _ = probe_video(args.path)
        source = SimpleNamespace(path=str(args.path))
    else:
        if not (args.video_id and args.player):
            raise ValueError("provide --path, or --video-id with --player")
        manifest = VideoManifest.load(args.root, args.player)
        if args.video_id not in manifest.sources:
            raise ValueError(f"no video {args.video_id!r} for player {args.player!r}")
        source = manifest.sources[args.video_id]
        timeline = manifest.timeline(args.video_id)
    frame_index = timeline.time_to_frame(seconds)
    frame = next(iter(extract_frames(source, [frame_index])))
    return frame, seconds

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
    remote = [] if args.no_remote_components else (
        args.remote_components or ["ejs:github"])
    try:
        paths = download_videos(
            args.url, args.dest, max_height=args.max_height, quiet=args.quiet,
            cookiefile=args.cookies, cookies_from_browser=args.cookies_from_browser,
            remote_components=remote,
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


def _load_layout(name: str):
    from poker_tell.broadcast.layout import POKERGO_CLASSIC_HSP

    layouts = {"pokergo_classic_hsp": POKERGO_CLASSIC_HSP}
    if name not in layouts:
        raise ValueError(f"unknown layout {name!r}; known: {sorted(layouts)}")
    return layouts[name]


def _cmd_read_frame(args: argparse.Namespace) -> int:
    from poker_tell.broadcast import read_frame
    from poker_tell.broadcast.cards import load_templates

    frame, seconds = _resolve_frame(args)
    # Default to appearance-based detection; --layout forces the manual fallback.
    layout = _load_layout(args.layout) if args.layout else None
    rank_t = suit_t = None
    if args.card_templates:
        templates = load_templates(args.card_templates)
        rank_t = {k[5:]: v for k, v in templates.items() if k.startswith("rank_")}
        suit_t = {k[5:]: v for k, v in templates.items() if k.startswith("suit_")}
    roster = args.roster.split(",") if args.roster else None
    reading = read_frame(frame, layout, rank_templates=rank_t,
                         suit_templates=suit_t, roster=roster)

    print(f"@ {seconds:.1f}s")
    print(f"pot: {reading.pot}")
    print(f"board: {' '.join(reading.board) if reading.board else '(none)'}")
    if not args.card_templates:
        print("  (cards not read — pass --card-templates <dir> once calibrated)")
    for s in reading.seats:
        st = s.status
        if st.kind == "action":
            status = f"{st.action_type}" + (f" {st.amount}" if st.amount else "")
        elif st.kind == "equity":
            status = f"{st.equity_pct}%"
        else:
            status = st.kind
        cards = " ".join(s.hole_cards) if s.hole_cards else "??"
        print(f"  - {s.name}: {cards} [{status}]")
    return 0


def _cmd_detect_overlays(args: argparse.Namespace) -> int:
    """Draw the detected overlay boxes on a frame and save it, for calibration."""
    from poker_tell.broadcast.detect import detect_overlays, draw_overlays

    cv2 = __import__("cv2")
    frame, seconds = _resolve_frame(args)
    bgr = frame[:, :, ::-1]  # extract gives RGB; cv2 wants BGR
    det = detect_overlays(bgr)
    annotated = draw_overlays(bgr, det)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), annotated)
    print(f"frame @ {seconds:.1f}s — detected: "
          f"pot={'yes' if det.pot else 'no'}, "
          f"{len(det.seats)} plate(s), {len(det.board)} board card(s)")
    print(f"wrote annotated frame to {out}. Open it: each box should sit on the "
          "POT banner, name plates, hole-card tiles, and board. Tune color/size "
          "thresholds (not coordinates) if a box is off.")
    return 0


def _cmd_format_segments(args: argparse.Namespace) -> int:
    """Detect and print where the compilation switches graphics formats."""
    from poker_tell.broadcast.formats import track_formats

    manifest = VideoManifest.load(args.root, args.player)
    if args.video_id not in manifest.sources:
        raise ValueError(f"no video {args.video_id!r} for player {args.player!r}")
    source = manifest.sources[args.video_id]
    timeline = manifest.timeline(args.video_id)
    segments = track_formats(source, timeline=timeline,
                             sample_seconds=args.sample_seconds)
    print(f"{len(segments)} format segment(s):")
    for seg in segments:
        t0 = timeline.frame_to_time(seg.start_frame)
        t1 = timeline.frame_to_time(min(seg.end_frame, timeline.n_frames - 1))
        p = seg.profile
        print(f"  frames {seg.start_frame}-{seg.end_frame} "
              f"({t0:.0f}s-{t1:.0f}s): pot HSV={p.pot_color_hsv} "
              f"box=({p.pot_box.x},{p.pot_box.y},{p.pot_box.w},{p.pot_box.h})")
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
    pd.add_argument("--cookies", default=None,
                    help="path to an exported cookies.txt (needed when YouTube "
                         "demands sign-in, e.g. from a Codespace/datacenter IP)")
    pd.add_argument("--cookies-from-browser", default=None,
                    help="read cookies from a local browser, e.g. chrome "
                         "(only works where that browser is installed)")
    pd.add_argument("--remote-components", action="append", default=None,
                    help="yt-dlp components it may fetch when needed "
                         "(default: ejs:github, for YouTube's JS challenge "
                         "solver). Repeat for several.")
    pd.add_argument("--no-remote-components", action="store_true",
                    help="forbid all remote-component fetching (use if the "
                         "yt-dlp-ejs package is installed locally)")
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

    def _add_frame_source(p):
        p.add_argument("--at", required=True,
                       help="timestamp HH:MM:SS / MM:SS / seconds")
        p.add_argument("--path", help="path to a video file")
        p.add_argument("--video-id", help="registered video id (with --player)")
        p.add_argument("--player", help="player namespace for --video-id")

    pr = sub.add_parser("read-frame",
                        help="read one frame's overlays (pot/board/plates)")
    _add_frame_source(pr)
    pr.add_argument("--card-templates", default=None,
                    help="dir of rank_*.npy / suit_*.npy card templates")
    pr.add_argument("--roster", default=None,
                    help="comma-separated known surnames to snap names to")
    pr.add_argument("--layout", default=None,
                    help="force the manual fractional layout (default: auto-detect)")
    pr.set_defaults(func=_cmd_read_frame)

    po = sub.add_parser("detect-overlays",
                        help="draw detected overlay boxes on a frame (calibration)")
    _add_frame_source(po)
    po.add_argument("--out", required=True, help="output PNG path")
    po.set_defaults(func=_cmd_detect_overlays)

    pf = sub.add_parser("format-segments",
                        help="detect where the compilation switches formats")
    pf.add_argument("--player", required=True)
    pf.add_argument("--video-id", required=True)
    pf.add_argument("--sample-seconds", type=float, default=2.0,
                    help="seconds between sampled frames (default: 2)")
    pf.set_defaults(func=_cmd_format_segments)

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
