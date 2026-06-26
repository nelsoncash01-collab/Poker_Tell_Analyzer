"""Reading hand state from broadcast on-screen graphics (overlay OCR).

Stage 1 of the broadcast-reconstruction pipeline: read a single frame's
overlays — pot, board cards, and each player's name-plate (hole cards + the
status line that shows their action / equity % / WINNER).

This is reused per player but is *graphics reading*, not behavioral modeling:
nothing learned here is shared into any per-player model. The card recognizer and
text parsers are deliberately dependency-light (numpy for image work); only the
text OCR step needs ``pytesseract`` (+ ``Pillow``), imported lazily.
"""

from poker_tell.broadcast.cards import recognize_card, normalize_card_str
from poker_tell.broadcast.layout import (
    OverlayLayout,
    POKERGO_CLASSIC_HSP,
    Region,
    detect_active_area,
)
from poker_tell.broadcast.ocr import (
    FrameReading,
    PlateStatus,
    SeatReading,
    parse_pot,
    parse_status,
    read_frame,
)

__all__ = [
    "recognize_card",
    "normalize_card_str",
    "OverlayLayout",
    "POKERGO_CLASSIC_HSP",
    "Region",
    "detect_active_area",
    "FrameReading",
    "PlateStatus",
    "SeatReading",
    "parse_pot",
    "parse_status",
    "read_frame",
]
