# Poker Tell Analyzer

Per-individual behavioral modeling of poker players from recorded video +
structured hand history. The system ingests footage (broadcast tape, session
recordings) and hand-history data for **one player at a time** and builds a
model of that player's tendencies (e.g. bluff probability at a decision point)
from bet sizing, timing, position, stack depth, in-session history, and visual
cues.

## The one rule that shapes everything

**Models are strictly per-individual — there is no cross-player transfer, in
any form.** No shared weights, no pretrained embeddings, no population priors,
no cross-player feature normalization. The *pipeline* (code) is reused across
players; nothing *learned* ever is. See `CLAUDE.md` and
`.claude/skills/poker-tell-analysis/SKILL.md` for the full rationale and the
domain failure modes (sync drift, labeling bias, small-N statistics).

This is enforced in code, not just convention: storage is namespaced per player
(`data/<player_id>/`, `models/<player_id>/`), `SyncTable` is bound to a single
player and refuses foreign entries, and `paths.guard_single_player` turns an
accidental cross-namespace read into a loud error.

## Pipeline stages

1. **Ingestion** — register raw footage as an immutable, hashed `VideoSource`
   (true per-frame PTS) and load the player's hand histories from JSON/CSV into
   a per-player store. **Built and tested**, with a `poker-tell` CLI.
2. **Synchronization** — align video frames to hand-history events at
   hand/street/action granularity. **The load-bearing wall; built and tested.**
3. **Feature extraction** — CV + game-state features, per player. *(not yet built)*
4. **Labeling** — bluff/value ground truth, preferring revealed-hole-card hands. *(not yet built)*
5. **Per-individual baseline & model** — trained from scratch on this player only. *(not yet built)*
6. **Reporting** — every claim carries sample size, confidence, and clip references. *(not yet built)*

## Giving the system videos and hands

Install the CLI (`pip install -e .` exposes `poker-tell`; or run
`python -m poker_tell.cli`). All commands take `--root` (the project dir holding
`data/<player_id>/`, default current dir) and a `--player` id.

**Videos** live anywhere on your local disk — an external drive, a NAS mount, a
`footage/` folder. They are **never copied into the repo**; the system records
the absolute path plus a SHA-256 and reads the true PTS timeline.

If you need to fetch footage from YouTube etc., there's a `download` helper
(uses yt-dlp). It defaults to the best available MP4; pass `--max-height 1080`
to cap resolution. For best-quality merged MP4s, also install the **ffmpeg
binary** on your machine (`brew install ffmpeg` / `apt install ffmpeg`).
Download only footage you have the right to use.

Downloading from YouTube also needs a **JavaScript runtime** (install Deno:
`curl -fsSL https://deno.land/install.sh | sh`) so yt-dlp can unlock the video
stream — without it the download fails with "Requested format is not
available". And from a datacenter IP (e.g. a GitHub Codespace) YouTube usually
demands sign-in: export a `cookies.txt` from a browser where you're logged in
and pass `--cookies cookies.txt`.

```bash
poker-tell download --url "https://www.youtube.com/watch?v=..." --dest ./footage
poker-tell download --url URL1 --url URL2 --dest ./footage --max-height 1080
```

This just puts files on disk; you then register them with `ingest-video` below.

```bash
# one file
poker-tell ingest-video --player negreanu --video-id hsp_s1_e1 \
    --path /footage/negreanu/hsp_s1_e1.mp4 \
    --source "High Stakes Poker S1E1" --air-date 2006-01-16

# many episodes at once, from a catalog (see examples/videos_catalog.csv)
poker-tell ingest-video --player negreanu --catalog episodes.csv
```

**Hand histories** are supplied as JSON (canonical, round-trips the model) or a
flat CSV (one row per action — convenient for manual reconstruction in a
spreadsheet, since broadcast footage rarely has a standard HH export):

```bash
poker-tell ingest-hands --player negreanu --path hands.json
poker-tell ingest-hands --player negreanu --path hands.csv      # auto-detected
poker-tell ls --player negreanu                                 # what's ingested
```

See `examples/hands.json`, `examples/hands.csv`, and `examples/videos_catalog.csv`
for the exact schemas. JSON is an array of hands (`hand_id`, `player_id`,
`hand_start_time`, optional `hole_cards` + `hole_cards_revealed`, and an
`actions` list of `{street, actor, action_type, amount, wall_clock}`); the CSV
repeats the hand-level fields on each action row and is grouped by `hand_id`.

> Hand-history ingestion produces the `HandHistory` records that the **anchoring
> step** (the next stage) and `SyncTable.coverage()` consume — it does not itself
> build the sync table's frame mappings.

### Reconstructing hands from broadcast graphics (Stage 1: read one frame)

When there's no hand-history export, hands are reconstructed from the on-screen
graphics (pot, board, and each player's name plate with hole cards + their
action / equity %). Stage 1 reads a single frame, and is the piece you calibrate
first. It needs the OCR extra plus the tesseract binary:

```bash
pip install -e ".[ocr]"
sudo apt-get install -y tesseract-ocr        # or: brew install tesseract

# dump the overlay crops of one frame to check the regions line up
poker-tell dump-regions --video-id negreanu_hsp_compilation --player negreanu \
    --at 00:10:00 --out ./calib
# read pot / board / name-plates (actions) from one frame
poker-tell read-frame --video-id negreanu_hsp_compilation --player negreanu \
    --at 00:10:00 --roster NEGREANU,IVEY,HELLMUTH,GREENSTEIN
```

Card recognition is numpy-only and needs templates calibrated from real frames
(`--card-templates <dir>`); without them you still get pot, names, and actions.
The overlay layout (`pokergo_classic_hsp`) is a first-pass calibration — tune it
with `dump-regions` against your footage. Later stages (auto hand segmentation,
assembling `HandHistory` + a `SyncTable`, then bluff/value labeling) build on a
trustworthy single-frame reader.

## What's implemented now

`src/poker_tell/`:

- **`ingest`** — registers raw footage without moving or modifying it: hashes
  the file (SHA-256), reads the **real per-frame PTS** via PyAV, and stores a
  per-player `VideoManifest` under `data/<player_id>/video/` (with the PTS
  timeline cached alongside). Records an `is_vfr` flag so downstream code knows
  whether a single-fps assumption would have lied about this file. Frames are
  extracted lazily (`extract_frame`) for spot-checks, never bulk-dumped. The
  same physical broadcast file may be registered under several players'
  manifests — the no-transfer rule governs *learned* content, not raw pixels.
- **`video.VideoTimeline`** — PTS-backed timebase: `time_to_frame` is a
  nearest-PTS lookup, correct on VFR/telecined footage where `FrameClock`'s
  arithmetic would drift. Both satisfy the `Timebase` protocol, so `SyncTable`
  accepts either.
- **`video.FrameClock`** — pure frame↔time arithmetic for CFR/synthetic data
  (handles fractional fps like 29.97 so drift doesn't accumulate).
- **`hand_history`** — minimal `Street` / `Action` / `HandHistory` model, with
  hole-card-reveal flags (the gold-standard bluff/value labeling source) and
  JSON (de)serialization.
- **`hand_ingest`** — loads a player's hands from the canonical JSON format or a
  flat one-row-per-action CSV, validates them (foreign player, duplicate ids,
  out-of-order streets, hole-card/reveal-flag mismatches), and stores them in a
  per-player, single-player-enforced `HandHistoryStore` under
  `data/<player_id>/hand_history/`. Reports a label-source summary (how many
  hands have revealed hole cards vs. showdown-only) so the selection-bias caveat
  surfaces at ingest time.
- **`sync.SyncTable`** — maps `hand_id -> (start_frame, end_frame, per-street
  boundaries)` for one player, with:
  - **structural validation** — streets inside the hand span and in betting
    order, no street/hand frame overlaps, double-assigned frame ranges flagged;
  - **clock-drift detection** — regresses the video timeline against the
    hand-history wall-clock timeline across hands; a near-1.0 slope with small
    residuals indicates sound sync, and individual hands exceeding a tolerance
    are flagged as likely mis-syncs (catching the *silent* drift that otherwise
    corrupts the dataset unnoticed).
- **`paths`** — per-player namespaced directories + the cross-player leakage
  guard.

> Validation catches structural impossibilities and statistical drift. It does
> **not** certify a sync that is internally consistent but uniformly shifted by
> a real offset — always spot-check a sample of hands against the footage
> manually before trusting a sync table at scale.

## Setup

```bash
pip install -r requirements.txt        # or: pip install -e ".[dev]"
PYTHONPATH=src python -m pytest        # run the test suite
```

## Status

Scaffolding + PTS-based video ingestion + JSON/CSV hand-history ingestion (with
a `poker-tell` CLI) + a validated sync foundation. The next stage is the
**anchoring step** that populates the sync table from footage + hands. Later
stages (CV features, labeling, modeling, reporting) are intentionally not built
yet — per the project rules, nothing should be built on top of an unvalidated
sync.
