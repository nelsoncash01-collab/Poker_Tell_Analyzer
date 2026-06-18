"""Video <-> hand-history synchronization: the load-bearing wall.

Everything downstream (CV feature extraction, labeling, modeling) inherits the
errors of this layer, and the characteristic failure mode is *silent* drift —
frame boundaries that creep out of alignment without anyone noticing until the
features are quietly garbage. So this module does two jobs:

1. Hold the sync table: ``hand_id -> (start_frame, end_frame, per-street frame
   boundaries)`` for one player's footage, with structural validation.
2. Detect clock drift by comparing, across many anchored hands, the video
   timeline against the hand-history wall-clock timeline. If they don't relate
   by a clean near-1.0 linear fit with small residuals, the sync is suspect.

Spot-check a sample manually before trusting any sync table at scale; the
validation here catches structural impossibilities and statistical drift, not a
sync that is internally consistent but uniformly shifted by a real offset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from poker_tell.hand_history import HandHistory, Street
from poker_tell.video import FrameClock


@dataclass(frozen=True)
class StreetBoundary:
    """Inclusive frame range for one street within a hand."""

    street: Street
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if self.start_frame < 0:
            raise ValueError(f"start_frame must be >= 0, got {self.start_frame}")
        if self.end_frame < self.start_frame:
            raise ValueError(
                f"{self.street.name}: end_frame ({self.end_frame}) precedes "
                f"start_frame ({self.start_frame})"
            )


@dataclass(frozen=True)
class HandSyncEntry:
    """Maps one hand to its frame range and per-street boundaries.

    ``hh_start_time`` mirrors ``HandHistory.hand_start_time`` and is what drift
    detection regresses the video timeline against. ``verified`` records whether
    a human spot-checked this entry against the footage.
    """

    hand_id: str
    player_id: str
    video_id: str
    start_frame: int
    end_frame: int
    streets: tuple[StreetBoundary, ...] = ()
    hh_start_time: float | None = None
    anchor_method: str = "unspecified"
    verified: bool = False

    def __post_init__(self) -> None:
        if self.end_frame < self.start_frame:
            raise ValueError(
                f"{self.hand_id}: end_frame ({self.end_frame}) precedes "
                f"start_frame ({self.start_frame})"
            )


# --- validation reporting --------------------------------------------------

_SEVERITIES = ("error", "warning")


@dataclass(frozen=True)
class SyncIssue:
    """A single validation finding. ``error`` = structurally impossible /
    corrupting; ``warning`` = suspicious but possibly legitimate (e.g. a frame
    gap that is really a commercial cut)."""

    severity: str
    code: str
    message: str
    hand_id: str | None = None

    def __post_init__(self) -> None:
        if self.severity not in _SEVERITIES:
            raise ValueError(f"severity must be one of {_SEVERITIES}")


@dataclass
class DriftReport:
    """Result of regressing video time on hand-history wall-clock time.

    ``slope`` is the relative clock rate (≈1.0 if the two clocks agree);
    ``intercept`` is the constant offset in video-seconds. ``residuals_s`` maps
    hand_id to signed residual in seconds (video minus fit). ``flagged`` lists
    hands whose absolute residual exceeds the tolerance — the likely drift /
    mis-sync points to inspect first.
    """

    n: int
    slope: float
    intercept: float
    max_abs_residual_s: float
    rmse_s: float
    residuals_s: dict[str, float] = field(default_factory=dict)
    flagged: list[str] = field(default_factory=list)
    tolerance_s: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.flagged


class SyncTable:
    """A single player's sync entries for one or more videos.

    Bound to one ``player_id`` on construction: adding an entry for any other
    player raises. This is a structural enforcement of the no-cross-player
    constraint at the data layer, not just a convention.
    """

    def __init__(self, player_id: str, clock: FrameClock):
        if not player_id:
            raise ValueError("player_id must be non-empty")
        self.player_id = player_id
        self.clock = clock
        self._entries: dict[str, HandSyncEntry] = {}

    # -- construction -------------------------------------------------------

    def add(self, entry: HandSyncEntry) -> None:
        if entry.player_id != self.player_id:
            raise PermissionError(
                f"SyncTable is bound to player {self.player_id!r}; refusing to "
                f"add entry for {entry.player_id!r}. Per-individual data must "
                "stay separated."
            )
        if entry.hand_id in self._entries:
            raise ValueError(f"duplicate hand_id {entry.hand_id!r}")
        self._entries[entry.hand_id] = entry

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, hand_id: str) -> bool:
        return hand_id in self._entries

    def __getitem__(self, hand_id: str) -> HandSyncEntry:
        return self._entries[hand_id]

    def entries(self) -> list[HandSyncEntry]:
        return list(self._entries.values())

    # -- validation ---------------------------------------------------------

    def validate(self) -> list[SyncIssue]:
        """Check structural consistency of every entry and across entries.

        Returns all findings (errors and warnings). An empty list, or a list
        with only warnings, means no structural impossibility was found — it
        does *not* by itself certify the sync is correct (see module docstring).
        """
        issues: list[SyncIssue] = []
        for e in self._entries.values():
            issues.extend(self._validate_entry(e))
        issues.extend(self._validate_overlaps())
        return issues

    def _validate_entry(self, e: HandSyncEntry) -> list[SyncIssue]:
        issues: list[SyncIssue] = []

        if not e.streets:
            issues.append(
                SyncIssue(
                    "warning",
                    "no_street_boundaries",
                    "entry has no per-street boundaries; downstream street "
                    "conditioning will be impossible",
                    e.hand_id,
                )
            )

        prev: StreetBoundary | None = None
        for sb in e.streets:
            # Street frames must lie within the hand's overall frame span.
            if sb.start_frame < e.start_frame or sb.end_frame > e.end_frame:
                issues.append(
                    SyncIssue(
                        "error",
                        "street_outside_hand",
                        f"{sb.street.name} frames [{sb.start_frame},"
                        f"{sb.end_frame}] fall outside hand span "
                        f"[{e.start_frame},{e.end_frame}]",
                        e.hand_id,
                    )
                )
            if prev is not None:
                # Streets must be in betting order.
                if sb.street <= prev.street:
                    issues.append(
                        SyncIssue(
                            "error",
                            "street_order",
                            f"{sb.street.name} appears after {prev.street.name}; "
                            "streets must be strictly increasing",
                            e.hand_id,
                        )
                    )
                # And not run backwards in frames or overlap each other.
                if sb.start_frame <= prev.end_frame:
                    sev = "error" if sb.start_frame < prev.start_frame else "warning"
                    issues.append(
                        SyncIssue(
                            sev,
                            "street_frame_overlap",
                            f"{sb.street.name} starts at {sb.start_frame} but "
                            f"{prev.street.name} ends at {prev.end_frame}",
                            e.hand_id,
                        )
                    )
            prev = sb

        return issues

    def _validate_overlaps(self) -> list[SyncIssue]:
        """Two hands on the same video must not claim overlapping frames."""
        issues: list[SyncIssue] = []
        by_video: dict[str, list[HandSyncEntry]] = {}
        for e in self._entries.values():
            by_video.setdefault(e.video_id, []).append(e)
        for video_id, group in by_video.items():
            ordered = sorted(group, key=lambda x: x.start_frame)
            for a, b in zip(ordered, ordered[1:]):
                if b.start_frame <= a.end_frame:
                    issues.append(
                        SyncIssue(
                            "error",
                            "hand_frame_overlap",
                            f"hands {a.hand_id!r} [{a.start_frame},"
                            f"{a.end_frame}] and {b.hand_id!r} [{b.start_frame},"
                            f"{b.end_frame}] overlap on video {video_id!r}; a "
                            "frame range is double-assigned",
                            b.hand_id,
                        )
                    )
        return issues

    # -- drift detection ----------------------------------------------------

    def drift_report(self, tolerance_s: float = 1.0) -> DriftReport:
        """Regress video start-time on hand-history start-time across hands.

        If the sync is sound, ``video_start_seconds ≈ slope * hh_start_time +
        intercept`` with ``slope`` near 1.0 and small residuals. A hand whose
        residual exceeds ``tolerance_s`` is flagged as a likely mis-sync / drift
        point. Requires at least 2 entries carrying ``hh_start_time``.
        """
        pairs = [
            (e.hand_id, self.clock.frame_to_time(e.start_frame), e.hh_start_time)
            for e in self._entries.values()
            if e.hh_start_time is not None
        ]
        if len(pairs) < 2:
            raise ValueError(
                "drift_report needs >=2 entries with hh_start_time set; "
                f"have {len(pairs)}"
            )
        hand_ids = [p[0] for p in pairs]
        video_t = np.array([p[1] for p in pairs], dtype=float)
        hh_t = np.array([p[2] for p in pairs], dtype=float)

        slope, intercept = np.polyfit(hh_t, video_t, 1)
        predicted = slope * hh_t + intercept
        residuals = video_t - predicted

        residuals_s = {hid: float(r) for hid, r in zip(hand_ids, residuals)}
        flagged = [hid for hid, r in residuals_s.items() if abs(r) > tolerance_s]
        rmse = float(np.sqrt(np.mean(residuals**2)))

        return DriftReport(
            n=len(pairs),
            slope=float(slope),
            intercept=float(intercept),
            max_abs_residual_s=float(np.max(np.abs(residuals))),
            rmse_s=rmse,
            residuals_s=residuals_s,
            flagged=flagged,
            tolerance_s=tolerance_s,
        )

    # -- persistence --------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten to one row per street boundary (or one row per hand with no
        streets). The central, inspectable per-player sync artifact."""
        rows: list[dict] = []
        for e in self._entries.values():
            base = {
                "player_id": e.player_id,
                "hand_id": e.hand_id,
                "video_id": e.video_id,
                "hand_start_frame": e.start_frame,
                "hand_end_frame": e.end_frame,
                "hh_start_time": e.hh_start_time,
                "anchor_method": e.anchor_method,
                "verified": e.verified,
            }
            if e.streets:
                for sb in e.streets:
                    rows.append(
                        {
                            **base,
                            "street": sb.street.name,
                            "street_start_frame": sb.start_frame,
                            "street_end_frame": sb.end_frame,
                        }
                    )
            else:
                rows.append(
                    {**base, "street": None, "street_start_frame": None,
                     "street_end_frame": None}
                )
        return pd.DataFrame(rows)

    def save_csv(self, path) -> None:
        self.to_dataframe().to_csv(path, index=False)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, clock: FrameClock) -> "SyncTable":
        """Rebuild a SyncTable from the flattened frame produced by
        ``to_dataframe`` (or an equivalent CSV). Enforces single-player."""
        players = set(df["player_id"].unique())
        if len(players) != 1:
            raise PermissionError(
                f"expected exactly one player_id in sync table, found {players}"
            )
        player_id = players.pop()
        table = cls(player_id=player_id, clock=clock)
        for hand_id, g in df.groupby("hand_id", sort=False):
            first = g.iloc[0]
            streets: list[StreetBoundary] = []
            for _, r in g.iterrows():
                if pd.notna(r.get("street")):
                    streets.append(
                        StreetBoundary(
                            street=Street[r["street"]],
                            start_frame=int(r["street_start_frame"]),
                            end_frame=int(r["street_end_frame"]),
                        )
                    )
            hh = first.get("hh_start_time")
            table.add(
                HandSyncEntry(
                    hand_id=str(hand_id),
                    player_id=str(first["player_id"]),
                    video_id=str(first["video_id"]),
                    start_frame=int(first["hand_start_frame"]),
                    end_frame=int(first["hand_end_frame"]),
                    streets=tuple(streets),
                    hh_start_time=(None if pd.isna(hh) else float(hh)),
                    anchor_method=str(first.get("anchor_method", "unspecified")),
                    verified=bool(first.get("verified", False)),
                )
            )
        return table

    # -- convenience --------------------------------------------------------

    def coverage(self, hands: list[HandHistory]) -> dict[str, list[str]]:
        """Compare the sync table against a list of hands for this player.

        Returns ``{"synced": [...], "missing_sync": [...], "orphan_sync": [...]}``
        so you can see which hands lack frame mappings and which sync entries
        reference hands you didn't supply. Refuses hands for other players.
        """
        for h in hands:
            if h.player_id != self.player_id:
                raise PermissionError(
                    f"coverage() got hand for player {h.player_id!r}; table is "
                    f"bound to {self.player_id!r}"
                )
        hh_ids = {h.hand_id for h in hands}
        sync_ids = set(self._entries)
        return {
            "synced": sorted(hh_ids & sync_ids),
            "missing_sync": sorted(hh_ids - sync_ids),
            "orphan_sync": sorted(sync_ids - hh_ids),
        }
