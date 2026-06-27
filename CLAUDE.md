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

- **Repository scaffolding + PTS-based video ingestion + JSON/CSV hand-history ingestion + a `poker-tell` CLI + sync layer foundation** is in place (`src/poker_tell/`):
  - `paths.py` — per-player namespaced storage paths with a cross-player leakage guard.
  - `video.py` — a `Timebase` protocol with two implementations: `FrameClock` (pure CFR arithmetic) and `VideoTimeline` (real per-frame PTS, with VFR detection and nearest-PTS lookup). `SyncTable` accepts either.
  - `ingest.py` — registers raw footage immutably (SHA-256 hashed, never moved/modified — referenced in place), reads true per-frame PTS via PyAV into a `VideoTimeline`, and persists a per-player `VideoManifest` under `data/<player_id>/video/` (timeline cached alongside). Records an `is_vfr` flag; extracts frames lazily for spot-checks. Chosen approach is PTS-based (not CFR-normalize) to avoid silent drift on VFR/inaccurate-fps broadcast files.
  - `hand_history.py` — `Street` / `Action` / `HandHistory` model (with JSON (de)serialization) used to anchor sync.
  - `hand_ingest.py` — loads a player's hands from canonical JSON or a flat one-row-per-action CSV, validates them, and stores them in a per-player single-player-enforced `HandHistoryStore` under `data/<player_id>/hand_history/`. Reports a label-source summary (revealed-hole-card vs. showdown-only) to surface the selection-bias caveat at ingest. Produces the `HandHistory` records the anchoring step and `SyncTable.coverage()` consume; it does not build the frame mappings itself.
  - `cli.py` — `poker-tell` CLI (entry point in `pyproject.toml`; also `python -m poker_tell.cli`): `download` (yt-dlp), `ingest-video` (single or `--catalog` batch), `ingest-hands` (JSON/CSV), `ls`, `read-frame` (`--reader llm` [default, Claude vision] or `--reader detect` [free OpenCV]), `detect-overlays` / `format-segments` (detection calibration). Expected user errors (bad input, leakage guard, missing file/API key, missing optional dep) print cleanly and return exit code 1.
  - `broadcast/` — Stage 1 of hand reconstruction from on-screen graphics (reads game-state ground truth only — pot/board/cards/actions — so it sits outside the no-transfer constraint). Two readers, both producing the same `FrameReading`:
    - **`llm_reader.py` (default, recommended)** — reads a frame via Claude vision (`read_frame_llm`, default model `claude-haiku-4-5`), returning structured `{pot, board, seats[name/hole_cards/status]}` via `output_config` json-schema. Pure JSON→`FrameReading` mapper (`reading_from_json`) + base64 PNG image encoding; anthropic/Pillow lazy-imported; API key from `ANTHROPIC_API_KEY` env (never committed). Used per-frame as a reader AND, per the cost-optimal plan, as the one-time **calibration** caller + a **fallback** on low-confidence frames.
    - **appearance detection (free fallback)** — `detect.py` (OpenCV: maroon POT/name chrome by colour + white card tiles by shape; POT banner anchors + scales the rest), `formats.py` (learn the POT banner colour/location + detect format changes via `track_formats`/`segment_profiles`), `cards.py` (numpy rank+suit recognizer), `ocr.py` (pure `parse_pot`/`parse_status` + `read_frame`). `layout.py` (fixed fractional ROIs) is a manual fallback. NOTE: appearance detection mostly mis-read the real footage (thresholds tuned on synthetic frames); the LLM reader is the chosen path, with detection demoted to the cheap frame-*sampler* + offline fallback.
    - Stages 2–4 (auto hand segmentation; LLM calibration → free detector → LLM-fallback router; assembling `HandHistory` + `SyncTable`; deterministic bluff/value labeling that never calls the LLM) are not yet built.
  - `sync.py` — `SyncTable` mapping `hand_id -> (start_frame, end_frame, per-street frame boundaries)`, with structural validation and clock-drift detection (linear fit of video time vs. hand-history wall-clock time, residual flagging).
  - `examples/` — sample `hands.json`, `hands.csv`, `videos_catalog.csv` documenting the input schemas (also used by tests).
  - `tests/` — unit tests + real PyAV-backed integration tests (video ingest + CLI); HH loader/store/validation tests including the seam feeding loaded hands into `SyncTable.coverage()`.
- PyAV is a real dependency (wheels bundle ffmpeg; no system ffmpeg needed). Footage is referenced in place. A broadcast file containing several players may be registered under multiple players' manifests — the no-transfer rule governs learned content/features, not raw pixels.
- Real footage ingested: a ~4h PokerGO "every Negreanu HSP hand" compilation under player `negreanu` (`negreanu_hsp_compilation`, 441,903 frames @ 29.97fps). No hand history reconstructed yet; no player baseline exists.
- Broadcast graphics are the PokerGO-classic-HSP style: fixed POT banner, left-column name plates (cards + NAME + a status line showing the player's action `RAISE TO $X`/`CALL $X`/etc., or equity %, or WINNER), bottom-center board cards; content pillarboxed (4:3 in 16:9). Betting actions ARE on-screen (read them); stack sizes are NOT (chips only).
- Not yet built: broadcast OCR Stages 2–4 (auto hand segmentation; assembling `HandHistory` + populating `SyncTable` from board-appearance street boundaries — the long-deferred anchoring step; bluff/value labeling defined as hand strength vs opponent range on the board, never realized/implied odds, and never a model input), plus CV behavioral feature extraction, per-player baseline distributions, and the per-individual model. Build these only on top of a validated sync table.
