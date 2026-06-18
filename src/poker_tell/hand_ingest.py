"""Hand-history ingestion: load a player's hands from disk and store them.

This is the input side that mirrors video ingestion (``ingest.py``). Broadcast
footage rarely ships with a standard hand-history export, so the input format is
one we define for the user to author (by hand / spreadsheet) or generate later
via overlay OCR:

- **JSON** is the canonical format — an array of hand objects that round-trips
  the ``HandHistory`` / ``Action`` model exactly (nested actions, optional hole
  cards). This is what programmatic reconstruction should emit.
- **CSV** is a flat one-row-per-action importer for manual reconstruction in a
  spreadsheet, which is how a human realistically rebuilds hands from a
  broadcast. Hand-level fields are repeated on every row of a hand.

Everything here is per-player and single-player enforced via the existing
``paths`` guard. Ingestion only produces the focal player's hand records; it
does not build the sync table's frame mappings (that is the separate anchoring
step) and does not touch modeling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from poker_tell.hand_history import HandHistory, Street
from poker_tell.paths import player_data_dir, validate_player_id


# --- validation reporting --------------------------------------------------

_SEVERITIES = ("error", "warning")


@dataclass(frozen=True)
class HandHistoryIssue:
    """A validation finding for loaded hands. ``error`` = will not ingest;
    ``warning`` = ingestible but worth a human's attention."""

    severity: str
    code: str
    message: str
    hand_id: str | None = None

    def __post_init__(self) -> None:
        if self.severity not in _SEVERITIES:
            raise ValueError(f"severity must be one of {_SEVERITIES}")


def validate_hands(
    hands: list[HandHistory], player_id: str
) -> list[HandHistoryIssue]:
    """Check loaded hands for structural and labeling problems.

    Errors block ingestion (foreign player, duplicate hand id). Warnings flag
    likely data-entry mistakes and labeling caveats without blocking.
    """
    issues: list[HandHistoryIssue] = []
    seen: set[str] = set()
    for h in hands:
        if h.player_id != player_id:
            issues.append(
                HandHistoryIssue(
                    "error", "foreign_player",
                    f"hand belongs to player {h.player_id!r}, expected "
                    f"{player_id!r}", h.hand_id,
                )
            )
        if h.hand_id in seen:
            issues.append(
                HandHistoryIssue(
                    "error", "duplicate_hand_id",
                    f"hand_id {h.hand_id!r} appears more than once", h.hand_id,
                )
            )
        seen.add(h.hand_id)

        # Action streets should be non-decreasing in recorded order; a river
        # action before a flop action signals a transcription error.
        streets = [a.street for a in h.actions]
        if any(b < a for a, b in zip(streets, streets[1:])):
            issues.append(
                HandHistoryIssue(
                    "warning", "street_order",
                    "actions are not in non-decreasing street order", h.hand_id,
                )
            )

        # Hole-card / reveal-flag consistency. Revealed cards are the
        # gold-standard bluff/value label source, so mismatches matter.
        if h.hole_cards_revealed and not h.hole_cards:
            issues.append(
                HandHistoryIssue(
                    "warning", "revealed_without_cards",
                    "hole_cards_revealed is true but no hole_cards given",
                    h.hand_id,
                )
            )
        if h.hole_cards and not h.hole_cards_revealed:
            issues.append(
                HandHistoryIssue(
                    "warning", "cards_without_reveal",
                    "hole_cards given but hole_cards_revealed is false",
                    h.hand_id,
                )
            )

        # Anchoring/drift detection needs wall-clock times somewhere in the
        # hand; warn if none are present.
        if not h.action_times():
            issues.append(
                HandHistoryIssue(
                    "warning", "no_action_wall_clock",
                    "no action carries a wall_clock; only hand_start_time is "
                    "available for anchoring", h.hand_id,
                )
            )
    return issues


def label_source_summary(hands: list[HandHistory]) -> dict:
    """Summarize the bluff/value labeling source for a set of hands.

    Revealed-hole-card hands are the gold standard (label known regardless of
    whether the hand reached showdown). A low revealed fraction is the skill
    doc's selection-bias warning sign, surfaced here at ingest time.
    """
    total = len(hands)
    revealed = sum(1 for h in hands if h.hole_cards_revealed and h.hole_cards)
    return {
        "n_hands": total,
        "n_revealed_hole_cards": revealed,
        "n_unrevealed": total - revealed,
        "revealed_fraction": (revealed / total) if total else 0.0,
    }


# --- loaders ---------------------------------------------------------------


def load_hands_json(path: Path | str) -> list[HandHistory]:
    """Load hands from the canonical JSON format (array of hand objects)."""
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict) and "hands" in payload:
        payload = payload["hands"]  # tolerate a {"hands": [...]} wrapper
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a JSON array of hands")
    return [HandHistory.from_dict(d) for d in payload]


_CSV_HAND_FIELDS = ("player_id", "hand_start_time", "hole_card_1",
                    "hole_card_2", "hole_cards_revealed")


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "y")


def _opt(value):
    """Normalize a possibly-missing CSV cell to a value or None."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def load_hands_csv(path: Path | str) -> list[HandHistory]:
    """Load hands from a flat one-row-per-action CSV.

    Required columns: ``hand_id, player_id, hand_start_time, street, actor,
    action_type``. Optional: ``amount, wall_clock, hole_card_1, hole_card_2,
    hole_cards_revealed``. Hand-level fields are repeated on each of a hand's
    rows and must be consistent within a ``hand_id``.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=True)
    required = {"hand_id", "player_id", "hand_start_time", "street", "actor",
                "action_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: CSV missing required columns {sorted(missing)}")

    hands: list[HandHistory] = []
    # Preserve first-seen hand order; preserve row order within a hand.
    for hand_id, group in df.groupby("hand_id", sort=False):
        first = group.iloc[0]
        # Enforce hand-level field consistency across the hand's rows.
        for fld in _CSV_HAND_FIELDS:
            if fld in group.columns and group[fld].nunique(dropna=False) > 1:
                raise ValueError(
                    f"{path}: hand {hand_id!r} has inconsistent {fld!r} across "
                    "its rows"
                )
        c1 = _opt(first.get("hole_card_1"))
        c2 = _opt(first.get("hole_card_2"))
        hole = (str(c1), str(c2)) if c1 is not None and c2 is not None else None
        actions = []
        for _, row in group.iterrows():
            actions.append({
                "street": row["street"],
                "actor": row["actor"],
                "action_type": row["action_type"],
                "amount": _opt(row.get("amount")) or 0.0,
                "wall_clock": _opt(row.get("wall_clock")),
            })
        hands.append(HandHistory.from_dict({
            "hand_id": str(hand_id),
            "player_id": first["player_id"],
            "hand_start_time": first["hand_start_time"],
            "actions": actions,
            "hole_cards": hole,
            "hole_cards_revealed": _as_bool(_opt(first.get("hole_cards_revealed"))),
        }))
    return hands


def load_hands(path: Path | str, fmt: str = "auto") -> list[HandHistory]:
    """Load hands, dispatching on ``fmt`` (``auto`` infers from extension)."""
    path = Path(path)
    if fmt == "auto":
        suffix = path.suffix.lower()
        fmt = {".json": "json", ".csv": "csv"}.get(suffix, "")
        if not fmt:
            raise ValueError(
                f"cannot infer format from {path.suffix!r}; pass fmt='json' or "
                "fmt='csv'"
            )
    if fmt == "json":
        return load_hands_json(path)
    if fmt == "csv":
        return load_hands_csv(path)
    raise ValueError(f"unknown format {fmt!r}")


# --- per-player store ------------------------------------------------------


@dataclass
class HandHistoryStore:
    """All ``HandHistory`` records for one player.

    Lives under ``data/<player_id>/hand_history/``. Bound to a single player;
    adding a foreign hand raises. Persists to ``hands.json`` with a player_id
    header so a file for the wrong player is refused on load.
    """

    player_id: str
    root: Path
    _hands: dict[str, HandHistory] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_player_id(self.player_id)
        self.root = Path(self.root)

    @property
    def dir(self) -> Path:
        return player_data_dir(self.root, self.player_id) / "hand_history"

    @property
    def hands_path(self) -> Path:
        return self.dir / "hands.json"

    def add(self, hand: HandHistory) -> None:
        if hand.player_id != self.player_id:
            raise PermissionError(
                f"store is bound to player {self.player_id!r}; refusing hand "
                f"for {hand.player_id!r}"
            )
        if hand.hand_id in self._hands:
            raise ValueError(f"duplicate hand_id {hand.hand_id!r}")
        self._hands[hand.hand_id] = hand

    def hands(self) -> list[HandHistory]:
        return list(self._hands.values())

    def __len__(self) -> int:
        return len(self._hands)

    def __contains__(self, hand_id: str) -> bool:
        return hand_id in self._hands

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "player_id": self.player_id,
            "hands": [h.to_dict() for h in self._hands.values()],
        }
        self.hands_path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, root: Path | str, player_id: str) -> "HandHistoryStore":
        store = cls(player_id=player_id, root=Path(root))
        if not store.hands_path.exists():
            return store
        payload = json.loads(store.hands_path.read_text())
        if payload.get("player_id") != player_id:
            raise PermissionError(
                f"hand-history file on disk is for {payload.get('player_id')!r}, "
                f"not {player_id!r}"
            )
        for d in payload.get("hands", []):
            store.add(HandHistory.from_dict(d))
        return store


# --- top-level ingest entrypoint ------------------------------------------


def ingest_hand_history(
    root: Path | str,
    player_id: str,
    path: Path | str,
    *,
    format: str = "auto",
    save: bool = True,
) -> tuple[HandHistoryStore, list[HandHistoryIssue]]:
    """Load hands from ``path``, validate, and add them to the player's store.

    Raises ``ValueError`` if validation finds any error-severity issue (the
    hands are not ingested in that case). Returns the updated store and the full
    list of issues (including warnings) for the caller to surface.
    """
    validate_player_id(player_id)
    hands = load_hands(path, fmt=format)
    issues = validate_hands(hands, player_id)
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        details = "; ".join(f"[{i.code}] {i.message}" for i in errors)
        raise ValueError(f"refusing to ingest {path}: {details}")

    store = HandHistoryStore.load(root, player_id)
    for h in hands:
        store.add(h)
    if save:
        store.save()
    return store, issues
