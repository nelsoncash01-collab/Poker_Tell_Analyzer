"""Per-individual poker tell analysis.

Core constraint (see CLAUDE.md / the poker-tell-analysis skill): models are
strictly per-individual. Nothing learned is ever shared across players. This
package provides the reusable *pipeline* — the learned content is not reused.

The currently implemented layer is the video<->hand-history synchronization
foundation, which everything downstream depends on.
"""

from poker_tell.hand_history import Action, ActionType, HandHistory, Street
from poker_tell.hand_ingest import (
    HandHistoryIssue,
    HandHistoryStore,
    ingest_hand_history,
    label_source_summary,
    load_hands,
    load_hands_csv,
    load_hands_json,
    validate_hands,
)
from poker_tell.ingest import (
    VideoManifest,
    VideoSource,
    ingest_video,
    probe_video,
)
from poker_tell.sync import (
    HandSyncEntry,
    StreetBoundary,
    SyncIssue,
    SyncTable,
    DriftReport,
)
from poker_tell.video import FrameClock, Timebase, VideoTimeline

__all__ = [
    "Action",
    "ActionType",
    "HandHistory",
    "Street",
    "FrameClock",
    "Timebase",
    "VideoTimeline",
    "VideoManifest",
    "VideoSource",
    "ingest_video",
    "probe_video",
    "HandHistoryIssue",
    "HandHistoryStore",
    "ingest_hand_history",
    "label_source_summary",
    "load_hands",
    "load_hands_csv",
    "load_hands_json",
    "validate_hands",
    "HandSyncEntry",
    "StreetBoundary",
    "SyncIssue",
    "SyncTable",
    "DriftReport",
]
