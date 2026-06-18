# Poker Tell Analysis — Project Instructions

## What this project is

Software that ingests recorded poker video (broadcast footage, session recordings, training tape) plus structured hand-history data, and builds a **per-individual** behavioral model predicting things like bluff probability at a given decision point — based on bet sizing, facial/body cues, position, decision timing, stack depth, and in-session hand history.

## The core design constraint — read before writing any modeling code

**Models are strictly per-individual. There is no cross-player transfer, in any form.**

- Each player has their own model, trained from scratch on only their own video + hand history.
- No shared weights, no pretrained "behavioral embedding," no population prior used as a model input (not even as a Bayesian prior or shrinkage target), no cross-player feature normalization.
- Generic poker tell folklore (e.g., Caro-style heuristics) may be referenced in comments or documentation as background context for a human reader, but must never feed into a model or be used to initialize/regularize a per-player model.
- The codebase/pipeline is reused across individuals (same scripts, same feature extraction logic). What is *learned* is never reused across individuals.
- If a change anywhere would cause information to leak from one player's data into another player's model or normalization stats, that change is wrong — flag it rather than implementing it, even if it would improve apparent performance.

This is intentional, not a placeholder for "future work to add transfer learning." Don't propose adding shared/pretrained components as an improvement.

## Scope and intent

- Built for **post-hoc review of recorded footage** — broadcast tape, training material, or session recordings — not real-time in-hand assistance. If a future task asks for live, in-hand inference during active play, treat that as a separate project with different constraints (latency, and likely game-integrity/rules issues at a real table) and flag it rather than building it into this codebase.
- Output is a probability/tendency report with sample size and confidence attached, plus links back to the specific hands/clips supporting each claim — not a black-box "this player is bluffing right now" assertion presented without evidence.
- First target case: a player with a large volume of broadcast footage where hole cards are revealed on the broadcast regardless of whether the hand goes to showdown (this solves the bluff/value labeling problem — see the skill doc for why this matters). Daniel Negreanu's televised hands (High Stakes Poker, Poker After Dark, WSOP final tables) are the initial worked example, but the pipeline should be written generically enough to point at any player with comparable footage.

## How to work in this codebase

1. Read `.claude/skills/poker-tell-analysis/SKILL.md` before writing any sync, feature-extraction, or modeling code — it has the domain-specific failure modes (sync drift, labeling bias, small-N statistics, and the no-transfer constraint in detail) that aren't obvious from generic CV/ML defaults.
2. Treat the video↔hand-history sync layer as the foundation. Don't build feature extraction or modeling on top of an unvalidated sync.
3. Keep each player's data, features, and model artifacts in separate, clearly namespaced locations (e.g., `data/<player_id>/...`, `models/<player_id>/...`). Never write code that reads across player namespaces during training.
4. Any reported tendency/tell in code comments, docs, or output must carry a sample size and a reference back to the underlying hands — treat this as a hard requirement, not a nice-to-have.
5. Validate models with session- or date-based holdout, not random hand-level holdout — hands within a session are correlated, and a player's strategy can drift over a multi-year footage archive.
6. When labeling ground truth, prefer revealed-hole-card hands (bluff/value known regardless of outcome) over showdown-only hands; if only showdown data is available, note the selection bias explicitly wherever that data is used.

## Current state

- **Repository scaffolding + PTS-based video ingestion + sync layer foundation** is in place (`src/poker_tell/`):
  - `paths.py` — per-player namespaced storage paths with a cross-player leakage guard.
  - `video.py` — a `Timebase` protocol with two implementations: `FrameClock` (pure CFR arithmetic) and `VideoTimeline` (real per-frame PTS, with VFR detection and nearest-PTS lookup). `SyncTable` accepts either.
  - `ingest.py` — registers raw footage immutably (SHA-256 hashed, never moved/modified), reads true per-frame PTS via PyAV into a `VideoTimeline`, and persists a per-player `VideoManifest` under `data/<player_id>/video/` (timeline cached alongside). Records an `is_vfr` flag; extracts frames lazily for spot-checks. Chosen approach is PTS-based (not CFR-normalize) to avoid silent drift on VFR/inaccurate-fps broadcast files.
  - `hand_history.py` — minimal `Street` / `Action` / `HandHistory` data model used to anchor sync.
  - `sync.py` — `SyncTable` mapping `hand_id -> (start_frame, end_frame, per-street frame boundaries)`, with structural validation and clock-drift detection (linear fit of video time vs. hand-history wall-clock time, residual flagging).
  - `tests/` — unit tests + real PyAV-backed ingestion integration tests (encode a clip, probe PTS, round-trip the manifest, decode frames back).
- PyAV is a real dependency now (wheels bundle ffmpeg; no system ffmpeg needed). A broadcast file containing several players may be registered under multiple players' manifests — the no-transfer rule governs learned content/features, not raw pixels.
- Not yet built: hand-history ingestion from real exports, the anchoring step that populates `SyncTable` from footage + HH (scene-cut/audio/overlay-OCR/manual anchors), CV feature extraction, labeling, per-player baseline distributions, and the per-individual model. Build these only on top of a validated sync table.
- No real footage or hand history has been ingested yet; no player baseline exists.
