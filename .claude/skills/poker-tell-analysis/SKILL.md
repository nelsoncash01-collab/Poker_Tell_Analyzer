---
name: poker-tell-analysis
description: Use this skill when building, extending, or debugging software that analyzes video footage of poker players to detect tells and predict individual behavior (e.g., bluff probability). Covers computer vision feature extraction (facial action units, posture, hand/chip movement), synchronization of video with structured hand-history data (bet sizing, timing, position, stack depth, session history), and the per-individual modeling architecture (strictly no cross-player transfer — every player's model is trained from scratch on only their own data). Trigger for any work on poker tell detection, player-specific behavioral prediction, or bluff-probability modeling from video.
---

# Poker Tell Analysis — Engineering Skill

## Read this first: the governing constraint

This project predicts behavior **per individual, in isolation**. Each player gets their own model, trained only on data from that player. There is no shared model, no pretrained "behavioral embedding," no population prior, and no parameter sharing across players — not even as a Bayesian prior. The pipeline (code, feature extraction, sync logic) is reused across individuals; the *learned* content is not. If you ever find yourself implicitly leaking information across players — e.g., initializing player B's model from player A's weights, normalizing player B's features using statistics computed across multiple players, or using population-level tell folklore (Caro-style heuristics) as a model input — stop. That folklore can be a sanity check for a human reading the output, never a model input.

The practical cost of this constraint: cold start per player is slow, because you cannot borrow statistical power from anyone else's data. Don't try to work around this with shrinkage toward a population mean or similar tricks — that *is* transfer, just disguised.

This is a measurement-and-statistics problem first, computer vision second. Bad CV gives you noisy features; bad stats turns noisy features into confident-sounding false claims about one specific person.

## Architecture overview

1. **Ingestion**: video files (broadcast footage, session recordings) + hand history (HH) logs for one player at a time.
2. **Synchronization layer**: align video timestamps to HH events at hand/street/action granularity. This is the load-bearing wall of the whole system — everything downstream inherits its errors.
3. **Feature extraction**: per-clip CV features + structured game-state features, computed independently per player (no cross-player normalization).
4. **Ground-truth labeling**: bluff/value labels where obtainable (see below — broadcast hole-card reveals are the best source).
5. **Per-individual baseline**: this player's own distribution of each feature, conditioned on whatever ground truth exists for them.
6. **Per-individual model**: trained from scratch, on this player's data only, to predict bluff probability (or other behavior) at decision points.
7. **Output**: per-player report with confidence/sample size and clip references for every claimed pattern — never assert a tendency without an evidence trail back to specific hands.

## Data sources and sync

- **Hand history (HH)**: hand ID, timestamp, seat/position, stack sizes, action sequence with sizing, board cards, and (critically) hole cards if revealed.
- **Hole-card ground truth is the single most valuable data source here.** Many broadcast formats (e.g., shows using an under-the-table hole-card camera) reveal a player's cards to the viewer/commentary track regardless of whether the hand reaches showdown or is folded. That means you can label a folded, uncalled bet as "bluff" or "value" with certainty — solving the labeling problem that sinks most amateur tell-detection projects, which only have showdown hands (a biased subsample: bluffs that got called or value bets that got called/shown, which skews the sample toward thinner spots).
- **Video timestamp anchoring**: overlay clocks, dealer-button placement, or audio cues (chip sounds, commentary references to bet sizes) to align frames to HH timestamps. If you control the capture, log frame-accurate timestamps at hand start directly instead of inferring them.
- Build a sync table: `hand_id -> (video_start_frame, video_end_frame, per-street frame boundaries)`. Spot-check a sample manually before trusting it at scale — silent sync drift is the most common way this kind of dataset gets corrupted without anyone noticing.

## Feature taxonomy

### Game-state features (get these exactly right — they're the ground truth layer)
- **Bet sizing**: absolute, % of pot, % of (effective) stack, and — since there's no cross-player normalization — relative to *this player's own* sizing history, both this session and historically.
- **Position**: button/CO/HJ/MP/EP/blinds; in-position vs out-of-position for the current street.
- **Stack depth**: in big blinds, and stack-to-pot ratio (SPR) at the decision point.
- **Timing**: seconds from action-on to action-taken, baselined against this player's own typical latency for this decision type (a snap-bet preflop with a premium hand means something different from a snap-bet facing a scary river bet — don't conflate them in one "speed" feature).
- **In-session hand history**: recent results (won/lost, caught bluffing, big pot won/lost), recent aggression frequency, hands played so far this session (fatigue/tilt proxy).
- **Street**: preflop/flop/turn/river. The same physical behavior can mean opposite things on different streets — never pool across streets without conditioning on street.
- **Role in the action**: lead bettor vs. caller vs. check-raiser. Tells differ by role, not just by the action taken.
- **Board texture**: dynamic/wet vs. static/dry boards change what a given bet size or timing tell implies.

### Visual/behavioral features (the CV layer)
- **Facial action units (FACS)**: use an established library — py-feat or OpenFace for actual FACS-coded action units, or MediaPipe Face Mesh as a faster but less rigorously validated alternative. Don't hand-roll FACS detection.
- **Eye gaze / blink rate**: MediaPipe iris tracking or a dedicated gaze model. Individually noisy; only useful once you have enough of this player's own hands to see a stable baseline.
- **Posture / body language**: pose estimation (MediaPipe Pose, OpenPose, or Detectron2 keypoints) for leaning, shoulder tension, stillness vs. fidgeting.
- **Hand/chip handling**: object detection/tracking on hands and chips (a fine-tuned YOLOv8, or MediaPipe Hands) — bet-placement smoothness, chip-stacking pattern changes, tremor.
- **Vocal/verbal** (if audio exists, e.g. table talk): pitch, pace, word choice via Whisper transcription + librosa for prosody. Voice-stress signal is real but weak and noisy at the individual-clip level — treat cautiously and lean harder on repetition across many of this player's hands before trusting it.
- **Respiration / visible pulse**: occasionally extractable from high-resolution footage but high false-positive rate. Lowest priority unless footage quality is unusually good.

### Derived/contextual features
- **This player's own table image trajectory this session** (have they been caught bluffing recently, are they playing tighter/looser than their own session-opening baseline) — affects how a given behavior should be weighted, but again, only against *their own* history.

## Per-individual baseline and modeling (the core of the "no transfer" constraint)

1. **Cold start**: collect this player's footage + HH. There is no shortcut via another player's data — don't initialize anything from a population model, including normalization constants (means/variances for scaling features must be computed from this player's data alone).
2. **Labeling**: use revealed-hole-card hands (folded or shown) as the primary source of bluff/value ground truth for this player. Showdown-only labels are usable but acknowledge the selection bias (thin/close spots are overrepresented).
3. **Baseline distributions**: for each feature, this player's distribution conditioned on label (bluff vs. value), conditioned also on the game-state context (street, position, role, stack depth) since the same feature means different things in different contexts even for one person.
4. **Modeling approach given likely small N for one player**: start with simple, interpretable methods — logistic regression on a handful of features, or a shallow gradient-boosted model with strong regularization — before anything more complex. A single player's broadcast history is realistically low hundreds of decision points at best; this is closer to a single-case/repeated-measures statistical problem than a big-data ML problem, and should be modeled and validated as such.
5. **Validation**: hold out by *session/date*, not just by hand. Hands within a session correlate (tilt, image, fatigue carry across hands), and date-based holdout also guards against the player's own strategy drifting over years of footage — don't train on 2010 hands and validate on 2024 hands as if they came from a stationary process.
6. **Reporting**: every output must state sample size, confidence interval (or equivalent), and link back to the specific hands supporting it. "This player tends to X when Y" is not a valid output without a number attached.
7. **Iteration across individuals**: the same pipeline code runs for each new player, but each run produces an entirely separate model artifact, trained from that player's data only. Store model artifacts namespaced by player and never load one player's artifact as a starting point for another's.

## Worked example: a single broadcast-heavy player (e.g., a long-running TV poker pro)

Players with many seasons of broadcast hands (especially shows that reveal hole cards on folds) are good first targets precisely because the labeling problem is solved for you. Practical approach:
1. Catalog all available episodes/hands featuring the player; extract HH where available, or reconstruct it manually/via OCR of graphics overlays (pot size, bet size, stack graphics) where no HH export exists.
2. Build the sync table episode-by-episode; broadcast footage often has commercial cuts and graphic overlays that complicate frame-accurate sync — budget real time for this step, it's usually the slowest part.
3. Extract features only for this player's hands; do not pool with any other player's footage even for the CV preprocessing step (e.g., don't train a shared face-embedding fine-tune across players if it bakes in cross-player normalization).
4. Treat the first pass as baseline-building, not prediction — you need the within-player baseline before a bluff classifier is meaningful at all.

## Statistical methods

- Per-feature: compare this player's feature distributions conditioned on bluff vs. value using a test appropriate to small, likely non-normal samples (Mann-Whitney U is a reasonable default for continuous features).
- Multivariate models: logistic regression or shallow gradient-boosted trees combining features, with session/date-based holdout as above.
- Report effect sizes and confidence intervals alongside (not instead of) point estimates and any p-values.
- Explicitly check for confounds: a feature that looks predictive might just be tracking position or stack depth, which independently correlate with bluff frequency. Control for game-state features before crediting a behavioral/visual feature.

## Tooling

- CV: OpenCV (frame extraction/preprocessing), MediaPipe (face mesh, pose, hands), py-feat or OpenFace (FACS action units), Detectron2/YOLOv8 (custom detection: chips, cards, dealer button, graphics-overlay OCR targets).
- Audio: Whisper (transcription), librosa (prosody features) — only if usable audio exists.
- Data: pandas for the synced per-hand feature table; treat this table (one per player, never merged across players) as the central artifact.
- Storage: raw video, extracted clips, and feature tables in separate, namespaced-by-player locations. Never overwrite raw footage; feature extraction will be revisited as methods improve.

## Common pitfalls

- Any form of cross-player leakage — shared normalization stats, shared pretrained weights, population priors used as model inputs. This is the single most important thing to avoid in this codebase.
- Treating showdown-only data as a representative sample (it overrepresents close/thin spots) when revealed-hole-card data is available instead.
- Sync drift between video and hand history silently corrupting the dataset.
- Confirmation bias: searching for a tell you already expect and stopping once you find a plausible-looking one.
- Reporting a pattern without sample size attached.
- Pooling across streets, positions, or roles without conditioning on them, even within one player's data.
- Training on early-career footage and validating on recent footage (or vice versa) as if the player's strategy were stationary over years.

## Workflow checklist for any task in this codebase

1. Confirm the sync layer (video timestamp ↔ hand history) is validated before touching feature extraction or modeling code.
2. Confirm which labeling source is in use for this player (revealed-hole-card vs. showdown-only) and flag the selection-bias caveat if it's showdown-only.
3. Confirm no cross-player artifact (weights, normalization constants, priors) is being loaded anywhere in the pipeline for this player's model.
4. When adding a feature, confirm it's expressed relative to this player's own baseline/context, not in absolute or population-relative terms.
5. When reporting any finding, attach sample size, confidence interval, and the underlying hand/clip references.
6. Validate models with session/date-based holdout, not hand-level holdout.
