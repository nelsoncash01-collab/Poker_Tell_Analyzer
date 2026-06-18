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
   in a per-player manifest and read its true per-frame PTS timeline.
   **Built and tested (PTS-based).**
2. **Synchronization** — align video frames to hand-history events at
   hand/street/action granularity. **The load-bearing wall; built and tested.**
3. **Feature extraction** — CV + game-state features, per player. *(not yet built)*
4. **Labeling** — bluff/value ground truth, preferring revealed-hole-card hands. *(not yet built)*
5. **Per-individual baseline & model** — trained from scratch on this player only. *(not yet built)*
6. **Reporting** — every claim carries sample size, confidence, and clip references. *(not yet built)*

## What's implemented now: the sync layer

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
  hole-card-reveal flags (the gold-standard bluff/value labeling source).
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

Scaffolding + PTS-based video ingestion + a validated sync foundation.
Downstream stages (CV features, labeling, modeling, reporting) are intentionally
not built yet — per the project rules, nothing should be built on top of an
unvalidated sync.
