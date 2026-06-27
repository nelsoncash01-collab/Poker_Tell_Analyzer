"""Read a frame's overlays with a vision LLM (Claude), as structured data.

This is the most robust reader for the broadcast graphics, and it plays two roles
in the cost-optimal pipeline (see the plan / Revision 3):

1. **Calibration caller** — used a small, fixed number of times to *learn* this
   show's graphics (colours, positions, card faces), from which a free
   deterministic detector is built (`llm_calibrate.py`).
2. **Fallback** — called on just the few frames the free detector flags as
   low-confidence, so the big run stays nearly free.

It reads **game-state ground truth from the broadcast graphics only** (pot,
board, names, hole cards, the action shown on each plate) — never behaviour,
tells, or bluff/value judgements. That keeps it on the right side of the
no-cross-player-transfer rule: it is OCR, not a behavioural model, and nothing it
returns is shared into any per-player model.

The Anthropic SDK and Pillow are optional deps, imported lazily. The API key is
read from the ``ANTHROPIC_API_KEY`` environment variable by the SDK — never
hardcode or commit it. The pure JSON->``FrameReading`` mapper and the prompt/
schema are unit-tested without any network or key (inject a fake ``client``).
"""

from __future__ import annotations

import base64
import io
import json

import numpy as np

from poker_tell.broadcast.cards import normalize_card_str
from poker_tell.broadcast.ocr import FrameReading, PlateStatus, SeatReading

DEFAULT_MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = (
    "You read the on-screen graphics of a poker broadcast and report exactly "
    "what the overlays show, as structured data. You are doing OCR of the "
    "graphics — report only what is visibly printed/shown on screen, never "
    "guess hidden cards, never infer game logic, and never judge whether a bet "
    "is a bluff. If something is not shown, report it as null/none."
)

READING_PROMPT = (
    "Read this poker broadcast frame's graphics. Report:\n"
    "- pot: the POT amount as an integer (e.g. $203,800 -> 203800), or null if "
    "no pot graphic is shown.\n"
    "- board: ONLY the community cards in the bottom-center strip, left to right, "
    "each as a 2-character code (rank + suit; rank in 23456789TJQKA with T for "
    "ten, suit in c/d/h/s). If no community-card strip is shown (e.g. a pre-flop "
    "all-in equity screen), return an empty list. NEVER put a player's hole "
    "cards in board.\n"
    "- seats: one entry per player name-plate shown (left column). For each: "
    "name (the surname on the plate), hole_cards (that player's two cards as "
    "2-char codes, or null if not shown), and status — the line under the name: "
    "kind 'action' with action_type (bet/call/raise/check/fold/all_in) and "
    "amount when an action like 'RAISE TO $3,000' is shown; kind 'equity' with "
    "equity_pct when a win % is shown; kind 'winner' when it says WINNER; "
    "otherwise kind 'none'. Use null for fields that don't apply.\n"
    "Read suits from the symbol SHAPE, not just colour. The two black suits are "
    "the common mistake: clubs (three rounded lobes) vs spades (one pointed leaf "
    "above the stem) — look closely and don't confuse them. Red suits: hearts "
    "(rounded) vs diamonds (angular)."
)

# JSON-schema for output_config.format (structured outputs). Every object sets
# additionalProperties:false and lists all properties in required; optional
# values are expressed as nullable via anyOf.
_NULLABLE_INT = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
_STATUS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "kind": {"type": "string", "enum": ["action", "equity", "winner", "none"]},
        "action_type": {
            "anyOf": [
                {"type": "string",
                 "enum": ["bet", "call", "raise", "check", "fold", "all_in"]},
                {"type": "null"},
            ]
        },
        "amount": _NULLABLE_INT,
        "equity_pct": _NULLABLE_INT,
    },
    "required": ["kind", "action_type", "amount", "equity_pct"],
}
_SEAT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "hole_cards": {
            "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]
        },
        "status": _STATUS_SCHEMA,
    },
    "required": ["name", "hole_cards", "status"],
}
READING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "pot": _NULLABLE_INT,
        "board": {"type": "array", "items": {"type": "string"}},
        "seats": {"type": "array", "items": _SEAT_SCHEMA},
    },
    "required": ["pot", "board", "seats"],
}


def _require_anthropic():
    try:
        import anthropic  # noqa: F401
    except ImportError as exc:  # pragma: no cover - only without the llm extra
        raise ImportError(
            "the LLM reader needs the Anthropic SDK: `pip install anthropic` "
            "(and set ANTHROPIC_API_KEY)."
        ) from exc
    return __import__("anthropic")


def _require_pillow():
    try:
        from PIL import Image  # noqa: F401
    except ImportError as exc:  # pragma: no cover - only without Pillow
        raise ImportError("image encoding needs Pillow: `pip install Pillow`.") from exc
    from PIL import Image

    return Image


# --- image encoding --------------------------------------------------------


def frame_to_png_base64(frame_rgb: np.ndarray, *, max_width: int = 1568) -> str:
    """Downscale (to cap image tokens) and PNG-encode an RGB frame as base64."""
    Image = _require_pillow()
    img = Image.fromarray(np.ascontiguousarray(frame_rgb).astype("uint8"))
    if img.width > max_width:
        h = round(img.height * max_width / img.width)
        img = img.resize((max_width, h))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def frame_to_image_block(frame_rgb: np.ndarray, *, max_width: int = 1568) -> dict:
    """An Anthropic image content block for ``frame_rgb`` (base64 PNG)."""
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": frame_to_png_base64(frame_rgb, max_width=max_width),
        },
    }


# --- json -> FrameReading --------------------------------------------------


def _norm_card(s) -> str | None:
    if not s:
        return None
    try:
        return normalize_card_str(str(s))
    except ValueError:
        return None


def _status_from_json(d: dict | None) -> PlateStatus:
    if not d:
        return PlateStatus("none")
    kind = d.get("kind") or "none"
    if kind not in ("action", "equity", "winner", "none"):
        kind = "none"
    return PlateStatus(
        kind=kind,
        action_type=d.get("action_type"),
        amount=(None if d.get("amount") is None else int(d["amount"])),
        equity_pct=(None if d.get("equity_pct") is None else int(d["equity_pct"])),
    )


def reading_from_json(d: dict) -> FrameReading:
    """Map the LLM's JSON object into a ``FrameReading`` (pure, no network).

    Cards are normalized (``10h`` -> ``Th``); unrecognized cards drop to None.
    """
    pot = d.get("pot")
    pot = None if pot is None else int(pot)

    board = [c for c in (_norm_card(x) for x in d.get("board", []) or []) if c]

    seats: list[SeatReading] = []
    for s in d.get("seats", []) or []:
        raw_cards = s.get("hole_cards")
        hole = None
        if raw_cards:
            norm = [_norm_card(x) for x in raw_cards]
            if len(norm) >= 2 and norm[0] and norm[1]:
                hole = (norm[0], norm[1])
        seats.append(SeatReading(
            name=(s.get("name") or None),
            hole_cards=hole,
            status=_status_from_json(s.get("status")),
        ))
    return FrameReading(pot=pot, board=board, seats=seats)


def _first_text(message) -> str:
    for block in message.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise ValueError("LLM response contained no text block")


def read_frame_llm(
    frame_rgb: np.ndarray,
    *,
    client=None,
    model: str = DEFAULT_MODEL,
    max_width: int = 1568,
) -> FrameReading:
    """Read one frame's overlays via Claude; return a ``FrameReading``.

    ``client`` is injectable for tests; left unset it constructs
    ``anthropic.Anthropic()`` (which reads ``ANTHROPIC_API_KEY``).
    """
    if client is None:
        client = _require_anthropic().Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                frame_to_image_block(frame_rgb, max_width=max_width),
                {"type": "text", "text": READING_PROMPT},
            ],
        }],
        output_config={"format": {"type": "json_schema", "schema": READING_SCHEMA}},
    )
    return reading_from_json(json.loads(_first_text(message)))
