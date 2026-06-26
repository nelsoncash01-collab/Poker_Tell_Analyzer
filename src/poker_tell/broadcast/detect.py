"""Detect overlay regions in a frame by appearance (not fixed coordinates).

The compilation splices clips at different crops/zooms, so fixed fractional ROIs
fail. Instead we find the graphics by what they look like:

- the maroon/red **chrome** of the POT banner and the NAME strips (color mask);
- the bright-white **card tiles** (rectangular, high-value/low-saturation).

The POT banner anchors everything and sets the scale, so plate/board/status
boxes are placed relative to it and self-adjust to each clip's framing. The
banner's expected colour/location can be supplied via a ``FormatProfile`` (see
``broadcast.formats``) so detection is cheap within one format and adapts when
the format changes.

This reads game-state ground truth from broadcast graphics — it is not a
behavioral model and nothing here is shared across players. OpenCV is imported
lazily so the rest of the package works without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Default maroon/red HSV ranges (OpenCV H in [0,180]); red wraps around 0/180.
DEFAULT_RED_RANGES = (
    ((0, 90, 40), (12, 255, 255)),
    ((168, 90, 40), (180, 255, 255)),
)


def _require_cv2():
    try:
        import cv2  # noqa: F401
    except ImportError as exc:  # pragma: no cover - only without opencv
        raise ImportError(
            "overlay detection needs OpenCV: `pip install opencv-python-headless`."
        ) from exc
    return __import__("cv2")


@dataclass(frozen=True)
class Box:
    """An axis-aligned pixel box ``(x, y, w, h)`` with helpers."""

    x: int
    y: int
    w: int
    h: int

    @property
    def x1(self) -> int:
        return self.x + self.w

    @property
    def y1(self) -> int:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y:self.y1, self.x:self.x1]


@dataclass
class SeatBoxes:
    name: Box
    status: Box
    card1: Box | None = None
    card2: Box | None = None


@dataclass
class DetectedOverlays:
    pot: Box | None = None
    pot_amount: Box | None = None
    seats: list[SeatBoxes] = field(default_factory=list)
    board: list[Box] = field(default_factory=list)


# --- masks -----------------------------------------------------------------


def red_chrome_mask(frame_bgr: np.ndarray, ranges=DEFAULT_RED_RANGES) -> np.ndarray:
    """Binary mask of the graphics' maroon/red chrome (POT banner + name bars)."""
    cv2 = _require_cv2()
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    return mask


def white_tile_mask(frame_bgr: np.ndarray, *, min_v: int = 180,
                    max_s: int = 45) -> np.ndarray:
    """Binary mask of bright, low-saturation regions (white card tiles)."""
    cv2 = _require_cv2()
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array((0, 0, min_v), np.uint8),
                       np.array((180, max_s, 255), np.uint8))


def connected_boxes(
    mask: np.ndarray,
    *,
    min_area: int = 80,
    aspect_range: tuple[float, float] = (0.05, 20.0),
    min_extent: float = 0.0,
    roi: tuple[float, float, float, float] | None = None,
) -> list[Box]:
    """Connected components of ``mask`` as boxes, filtered.

    ``aspect_range`` is width/height bounds; ``min_extent`` is the minimum
    area/bbox-area (filly-filled rectangles ~1.0; circles ~0.79); ``roi`` is a
    fractional ``(x0, y0, x1, y1)`` region the box centre must fall inside.
    """
    cv2 = _require_cv2()
    H, W = mask.shape[:2]
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out: list[Box] = []
    for i in range(1, n):  # skip background label 0
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_area or w == 0 or h == 0:
            continue
        aspect = w / h
        if not aspect_range[0] <= aspect <= aspect_range[1]:
            continue
        if (area / (w * h)) < min_extent:
            continue
        if roi is not None:
            cx, cy = centroids[i]
            if not (roi[0] * W <= cx <= roi[2] * W and roi[1] * H <= cy <= roi[3] * H):
                continue
        out.append(Box(x, y, w, h))
    return out


# --- the detector ----------------------------------------------------------


def detect_overlays(
    frame_bgr: np.ndarray,
    *,
    red_ranges=DEFAULT_RED_RANGES,
    plate_roi: tuple[float, float, float, float] = (0.0, 0.0, 0.42, 1.0),
    board_roi: tuple[float, float, float, float] = (0.30, 0.66, 0.95, 1.0),
) -> DetectedOverlays:
    """Locate the pot banner, name plates (+ hole-card tiles, status), and board.

    ``red_ranges`` can come from a learned ``FormatProfile`` so the banner colour
    matches the current format. ``plate_roi`` / ``board_roi`` are generous
    fractional priors (left column / bottom-centre band) — not exact positions.
    """
    H, W = frame_bgr.shape[:2]
    red = red_chrome_mask(frame_bgr, red_ranges)
    white = white_tile_mask(frame_bgr)

    # Wide red bars in the left column: the POT banner and the name strips.
    red_bars = connected_boxes(
        red, min_area=int(0.0008 * H * W), aspect_range=(1.4, 12.0),
        min_extent=0.55, roi=plate_roi,
    )
    red_bars.sort(key=lambda b: b.y)

    result = DetectedOverlays()
    if not red_bars:
        return result

    # Topmost red bar = POT banner (anchor + scale). The amount sits in its
    # lower portion, under the "POT" label.
    pot = red_bars[0]
    result.pot = pot
    result.pot_amount = Box(pot.x, pot.y + int(0.45 * pot.h), pot.w,
                            int(0.55 * pot.h))

    # White rectangles that look like card tiles (taller than wide, well-filled).
    card_boxes = connected_boxes(
        white, min_area=int(0.0006 * H * W), aspect_range=(0.45, 0.95),
        min_extent=0.80,
    )

    # Remaining red bars below the banner are name strips. For each, the hole
    # cards are the card tiles just above it; the status line is just below.
    for strip in red_bars[1:]:
        cards = [
            b for b in card_boxes
            if b.y1 <= strip.y + 0.3 * strip.h         # sits above the strip
            and abs(b.cx - strip.cx) <= 1.2 * strip.w  # roughly same column
            and (strip.y - b.cy) <= 2.2 * strip.h      # not far above
        ]
        cards.sort(key=lambda b: b.x)
        c1 = cards[0] if cards else None
        c2 = cards[1] if len(cards) > 1 else None
        status = Box(strip.x, strip.y1, strip.w, strip.h)  # light strip below name
        result.seats.append(SeatBoxes(name=strip, status=status, card1=c1, card2=c2))

    # Board: card tiles in the bottom-centre band, left to right.
    board = [
        b for b in card_boxes
        if board_roi[1] * H <= b.cy <= board_roi[3] * H
        and board_roi[0] * W <= b.cx <= board_roi[2] * W
    ]
    board.sort(key=lambda b: b.x)
    result.board = board
    return result


def draw_overlays(frame_bgr: np.ndarray, det: DetectedOverlays) -> np.ndarray:
    """Return a copy of the frame with detected boxes drawn (for calibration)."""
    cv2 = _require_cv2()
    out = frame_bgr.copy()

    def rect(box, color, label):
        if box is None:
            return
        cv2.rectangle(out, (box.x, box.y), (box.x1, box.y1), color, 2)
        cv2.putText(out, label, (box.x, max(0, box.y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    rect(det.pot, (0, 255, 255), "POT")
    for i, seat in enumerate(det.seats):
        rect(seat.name, (255, 0, 0), f"name{i}")
        rect(seat.status, (255, 128, 0), f"status{i}")
        rect(seat.card1, (0, 255, 0), f"c{i}a")
        rect(seat.card2, (0, 255, 0), f"c{i}b")
    for j, b in enumerate(det.board):
        rect(b, (0, 0, 255), f"b{j}")
    return out
