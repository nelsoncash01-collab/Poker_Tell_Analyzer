"""Minimal hand-history data model used to anchor and validate sync.

This is intentionally lean: only what the sync layer needs (identity, street
structure, and wall-clock timestamps for drift checks). The full game-state
feature set described in the skill doc (bet sizing, SPR, role, board texture,
etc.) is layered on later, on top of a *validated* sync table.

Wall-clock times are seconds since an arbitrary but fixed epoch for the source
(e.g. seconds into the broadcast, or POSIX time). Only differences matter for
drift detection, so the epoch just has to be consistent within one video.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum


class Street(IntEnum):
    """Betting streets, ordered. IntEnum so they sort/compare naturally."""

    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


class ActionType(str, Enum):
    """The action a player took. Kept minimal for the sync foundation."""

    POST_BLIND = "post_blind"
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"


@dataclass(frozen=True)
class Action:
    """One action by one actor on one street.

    ``wall_clock`` is the time (seconds, source epoch) the action occurred, when
    known — used only for anchoring/drift checks, never required for structure.
    ``amount`` is chips committed by this action (0 for check/fold).
    """

    street: Street
    actor: str
    action_type: ActionType
    amount: float = 0.0
    wall_clock: float | None = None

    def to_dict(self) -> dict:
        return {
            "street": self.street.name,
            "actor": self.actor,
            "action_type": self.action_type.value,
            "amount": self.amount,
            "wall_clock": self.wall_clock,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Action":
        return cls(
            street=Street[d["street"]],
            actor=d["actor"],
            action_type=ActionType(d["action_type"]),
            amount=float(d.get("amount", 0.0)),
            wall_clock=(None if d.get("wall_clock") is None
                        else float(d["wall_clock"])),
        )


@dataclass
class HandHistory:
    """One hand, from the perspective of the player being modeled.

    Only the focal ``player_id`` matters for modeling; other actors appear in
    ``actions`` purely as game context. ``hand_start_time`` is the wall-clock
    second the hand began (source epoch) and is the primary sync anchor.
    """

    hand_id: str
    player_id: str
    hand_start_time: float
    actions: list[Action] = field(default_factory=list)
    # Hole cards if revealed (e.g. broadcast hole-card cam) — the gold-standard
    # bluff/value labeling source. ``None`` means not revealed for this hand.
    hole_cards: tuple[str, str] | None = None
    hole_cards_revealed: bool = False

    def __post_init__(self) -> None:
        if not self.hand_id:
            raise ValueError("hand_id must be non-empty")
        if not self.player_id:
            raise ValueError("player_id must be non-empty")

    def streets_present(self) -> list[Street]:
        """Streets that actually have at least one action, in order."""
        seen = {a.street for a in self.actions}
        return sorted(seen)

    def action_times(self) -> list[float]:
        """Known action wall-clock times, in recorded order (skips None)."""
        return [a.wall_clock for a in self.actions if a.wall_clock is not None]

    def to_dict(self) -> dict:
        return {
            "hand_id": self.hand_id,
            "player_id": self.player_id,
            "hand_start_time": self.hand_start_time,
            "actions": [a.to_dict() for a in self.actions],
            "hole_cards": (None if self.hole_cards is None
                           else list(self.hole_cards)),
            "hole_cards_revealed": self.hole_cards_revealed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HandHistory":
        cards = d.get("hole_cards")
        return cls(
            hand_id=str(d["hand_id"]),
            player_id=str(d["player_id"]),
            hand_start_time=float(d["hand_start_time"]),
            actions=[Action.from_dict(a) for a in d.get("actions", [])],
            hole_cards=(None if cards is None else tuple(cards)),
            hole_cards_revealed=bool(d.get("hole_cards_revealed", False)),
        )
