"""Read a frame's overlays into a structured ``FrameReading``.

The pure parsers (``parse_pot``, ``parse_status``) work on OCR'd strings and are
fully unit-testable. The image→string step (``ocr_text``) lazily imports
``pytesseract`` (+ ``Pillow``); card recognition is numpy-only via
``poker_tell.broadcast.cards``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from poker_tell.broadcast.cards import recognize_card
from poker_tell.broadcast.layout import OverlayLayout, crop, detect_active_area

# Action keywords as they appear on the name-plate status line.
_ACTION_KEYWORDS = (
    ("all_in", ("ALL IN", "ALL-IN", "ALLIN")),
    ("raise", ("RAISE",)),
    ("call", ("CALL",)),
    ("check", ("CHECK",)),
    ("fold", ("FOLD",)),
    ("bet", ("BET",)),
)


@dataclass
class PlateStatus:
    """The status sub-line on a name plate, classified.

    ``kind`` is one of ``action`` / ``equity`` / ``winner`` / ``none``.
    """

    kind: str
    action_type: str | None = None
    amount: int | None = None
    equity_pct: int | None = None
    raw: str = ""


@dataclass
class SeatReading:
    name: str | None
    hole_cards: tuple[str, str] | None
    status: PlateStatus


@dataclass
class FrameReading:
    pot: int | None = None
    board: list[str] = field(default_factory=list)
    seats: list[SeatReading] = field(default_factory=list)


def _digits_to_int(text: str) -> int | None:
    """Pull a money/number amount out of text (strip ``$`` and thousands commas)."""
    m = re.search(r"\$?\s*([\d][\d,]*)", text)
    if not m:
        return None
    return int(m.group(1).replace(",", ""))


def parse_pot(text: str) -> int | None:
    """``'POT $203,800'`` / ``'$8,300'`` -> ``203800`` / ``8300``."""
    return _digits_to_int(text)


def parse_status(text: str) -> PlateStatus:
    """Classify a plate status line into an action / equity / winner / none."""
    raw = text.strip()
    up = raw.upper()
    if not up:
        return PlateStatus("none", raw=raw)
    if "WINNER" in up:
        return PlateStatus("winner", raw=raw)
    pct = re.search(r"(\d{1,3})\s*%", up)
    if pct:
        return PlateStatus("equity", equity_pct=int(pct.group(1)), raw=raw)
    for action_type, keywords in _ACTION_KEYWORDS:
        if any(k in up for k in keywords):
            amount = _digits_to_int(up) if action_type not in ("check", "fold") else None
            return PlateStatus("action", action_type=action_type, amount=amount, raw=raw)
    return PlateStatus("none", raw=raw)


def _require_text_ocr():
    try:
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as exc:  # pragma: no cover - only without the ocr extra
        raise ImportError(
            "text OCR needs pytesseract + Pillow (and the tesseract binary): "
            "`pip install pytesseract Pillow` and `apt-get install -y tesseract-ocr`."
        ) from exc
    import pytesseract
    from PIL import Image

    return pytesseract, Image


def ocr_text(img_rgb: np.ndarray, *, psm: int = 7) -> str:
    """OCR a single line of text from an RGB crop (lazy tesseract)."""
    pytesseract, Image = _require_text_ocr()
    pil = Image.fromarray(np.ascontiguousarray(img_rgb).astype("uint8"))
    return pytesseract.image_to_string(pil, config=f"--psm {psm}").strip()


def _looks_like_card_tile(tile_rgb: np.ndarray, *, min_luma: float = 110.0) -> bool:
    """Card tiles are bright/white; empty slots are dark felt/background."""
    return float(tile_rgb.mean()) >= min_luma


def read_frame(
    frame_rgb: np.ndarray,
    layout: OverlayLayout | None = None,
    *,
    detector=None,
    profile=None,
    rank_templates: dict[str, np.ndarray] | None = None,
    suit_templates: dict[str, np.ndarray] | None = None,
    roster: list[str] | None = None,
    text_ocr=ocr_text,
) -> FrameReading:
    """Read pot, board, and name plates from one RGB frame.

    By default the overlay regions are **detected by appearance**
    (``broadcast.detect.detect_overlays``), which handles the compilation's
    varying crops/zoom; pass a ``profile`` (``broadcast.formats.FormatProfile``)
    to seed the banner colour for the current format. Passing a fixed ``layout``
    instead uses the legacy fractional-ROI path (manual fallback).

    Card recognition runs only if both template sets are supplied (otherwise
    cards read as ``None`` and you still get pot/names/actions). ``roster`` snaps
    noisy name OCR to the closest known surname. ``detector`` / ``text_ocr`` are
    injectable for testing.
    """
    if layout is None:
        return _read_frame_detected(
            frame_rgb, detector=detector, profile=profile,
            rank_templates=rank_templates, suit_templates=suit_templates,
            roster=roster, text_ocr=text_ocr,
        )
    active = detect_active_area(frame_rgb)
    has_templates = bool(rank_templates) and bool(suit_templates)

    def read_card(region) -> str | None:
        tile = crop(frame_rgb, region, active)
        if not has_templates or not _looks_like_card_tile(tile):
            return None
        return recognize_card(tile, rank_templates, suit_templates)

    pot = parse_pot(text_ocr(crop(frame_rgb, layout.pot, active)))

    seats: list[SeatReading] = []
    for i in range(layout.max_plates):
        regions = layout.plate_regions(i)
        name = text_ocr(crop(frame_rgb, regions["name"], active))
        if not name:
            continue  # empty slot — no plate here
        if roster:
            name = _snap_name(name, roster)
        c1 = read_card(regions["card1"])
        c2 = read_card(regions["card2"])
        hole = (c1, c2) if c1 and c2 else None
        status = parse_status(text_ocr(crop(frame_rgb, regions["status"], active)))
        seats.append(SeatReading(name=name, hole_cards=hole, status=status))

    board: list[str] = []
    for j in range(layout.board_slots):
        card = read_card(layout.board_slot(j))
        if card is None:
            break  # board fills left-to-right; stop at first empty slot
        board.append(card)

    return FrameReading(pot=pot, board=board, seats=seats)


def _read_frame_detected(
    frame_rgb, *, detector, profile, rank_templates, suit_templates, roster,
    text_ocr,
):
    """Detection-based reading: find overlay boxes, then OCR/recognize them."""
    det_fn = detector
    if det_fn is None:
        from poker_tell.broadcast.detect import detect_overlays

        det_fn = detect_overlays
    kwargs = {}
    if profile is not None:
        kwargs["red_ranges"] = profile.red_ranges()
    det = det_fn(frame_rgb[:, :, ::-1], **kwargs)  # detector expects BGR

    has_templates = bool(rank_templates) and bool(suit_templates)

    def read_card(box) -> str | None:
        if box is None or not has_templates:
            return None
        tile = box.crop(frame_rgb)
        if not _looks_like_card_tile(tile):
            return None
        return recognize_card(tile, rank_templates, suit_templates)

    pot = None
    if det.pot_amount is not None:
        pot = parse_pot(text_ocr(det.pot_amount.crop(frame_rgb)))

    seats: list[SeatReading] = []
    for s in det.seats:
        name = text_ocr(s.name.crop(frame_rgb))
        if roster and name:
            name = _snap_name(name, roster)
        c1, c2 = read_card(s.card1), read_card(s.card2)
        hole = (c1, c2) if c1 and c2 else None
        status = parse_status(text_ocr(s.status.crop(frame_rgb)))
        seats.append(SeatReading(name=name or None, hole_cards=hole, status=status))

    board: list[str] = []
    for b in det.board:
        card = read_card(b)
        if card is None:
            break
        board.append(card)

    return FrameReading(pot=pot, board=board, seats=seats)


def _snap_name(text: str, roster: list[str]) -> str:
    """Snap an OCR'd name to the closest roster entry by simple edit distance."""
    up = re.sub(r"[^A-Z]", "", text.upper())
    if not up:
        return text.strip()
    best, best_d = text.strip(), None
    for cand in roster:
        d = _levenshtein(up, cand.upper())
        if best_d is None or d < best_d:
            best, best_d = cand, d
    # only snap if reasonably close (within ~40% of the name length)
    if best_d is not None and best_d <= max(1, len(best) * 0.4):
        return best
    return text.strip()


def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
