"""Overlay region geometry for a broadcast graphics style.

Regions are expressed as fractions of the *de-pillarboxed active area* (the real
picture inside any black bars), so the same layout works regardless of the
encoded resolution or pillarbox width. A layout is calibrated once per graphics
style; ``POKERGO_CLASSIC_HSP`` is a first-pass calibration for the PokerGO
re-release of classic High Stakes Poker (the only style in the current footage).

The fractional coordinates below are a starting guess to be tuned against real
frames with the ``dump-regions`` CLI — they are not expected to be pixel-perfect
out of the box.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Region:
    """A rectangle in fractional [0,1] coordinates of the active area."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.x0 < self.x1 <= 1.0 and 0.0 <= self.y0 < self.y1 <= 1.0):
            raise ValueError(f"invalid fractional region {self!r}")


def detect_active_area(
    frame_rgb: np.ndarray, *, black_thresh: int = 24, min_fraction: float = 0.5
) -> tuple[int, int, int, int]:
    """Find the non-letterboxed picture box as ``(x0, y0, x1, y1)`` pixels.

    Detects near-black bars on the borders by scanning rows/columns whose mean
    brightness stays under ``black_thresh``. Returns the inclusive-exclusive
    bounding box of the active picture. Falls back to the full frame if no clear
    bars are found.
    """
    if frame_rgb.ndim != 3:
        raise ValueError("expected an (H, W, 3) RGB frame")
    h, w, _ = frame_rgb.shape
    lum = frame_rgb.mean(axis=2)
    col_active = lum.mean(axis=0) >= black_thresh
    row_active = lum.mean(axis=1) >= black_thresh

    def _span(active: np.ndarray, n: int) -> tuple[int, int]:
        idx = np.flatnonzero(active)
        if idx.size < max(1, int(min_fraction * n)):
            return 0, n  # not enough active span; treat whole axis as active
        return int(idx[0]), int(idx[-1]) + 1

    x0, x1 = _span(col_active, w)
    y0, y1 = _span(row_active, h)
    return x0, y0, x1, y1


@dataclass(frozen=True)
class OverlayLayout:
    """Named overlay regions for one graphics style.

    ``plate0_top`` / ``plate_height`` describe the vertically-stacked name plates
    under the POT banner; ``plate_sub`` gives each plate's internal rows
    (two card tiles, name bar, status bar) as fractions *within* one plate.
    Board card slots are evenly spaced across ``board_band``.
    """

    name: str
    pot: Region
    plate_x0: float
    plate_x1: float
    plate0_top: float
    plate_height: float
    max_plates: int
    board_band: Region
    board_slots: int

    # internal plate row fractions (of one plate's height) and card x-splits
    plate_card_bottom: float = 0.55
    plate_name_bottom: float = 0.78
    plate_card1: tuple[float, float] = (0.02, 0.49)
    plate_card2: tuple[float, float] = (0.51, 0.98)

    def plate_regions(self, i: int) -> dict[str, Region]:
        """Sub-regions (card1, card2, name, status) for plate index ``i``."""
        if not 0 <= i < self.max_plates:
            raise IndexError(i)
        top = self.plate0_top + i * self.plate_height
        bot = top + self.plate_height
        card_b = top + self.plate_card_bottom * self.plate_height
        name_b = top + self.plate_name_bottom * self.plate_height
        cx0, cx1 = self.plate_x0, self.plate_x1
        span = cx1 - cx0

        def _cardx(frac: tuple[float, float]) -> tuple[float, float]:
            return cx0 + frac[0] * span, cx0 + frac[1] * span

        c1x0, c1x1 = _cardx(self.plate_card1)
        c2x0, c2x1 = _cardx(self.plate_card2)
        return {
            "card1": Region(c1x0, top, c1x1, card_b),
            "card2": Region(c2x0, top, c2x1, card_b),
            "name": Region(cx0, card_b, cx1, name_b),
            "status": Region(cx0, name_b, cx1, bot),
        }

    def board_slot(self, j: int) -> Region:
        """Region for board card slot ``j`` (left to right)."""
        if not 0 <= j < self.board_slots:
            raise IndexError(j)
        b = self.board_band
        w = (b.x1 - b.x0) / self.board_slots
        # slight inset so adjacent slots don't bleed into each other
        x0 = b.x0 + j * w + 0.1 * w
        x1 = b.x0 + (j + 1) * w - 0.1 * w
        return Region(x0, b.y0, x1, b.y1)


def region_to_pixels(
    region: Region, active: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """Map a fractional region into absolute pixel coords within ``active``."""
    ax0, ay0, ax1, ay1 = active
    aw, ah = ax1 - ax0, ay1 - ay0
    return (
        int(ax0 + region.x0 * aw),
        int(ay0 + region.y0 * ah),
        int(ax0 + region.x1 * aw),
        int(ay0 + region.y1 * ah),
    )


def crop(
    frame_rgb: np.ndarray, region: Region, active: tuple[int, int, int, int]
) -> np.ndarray:
    """Crop ``frame_rgb`` to a fractional ``region`` of the active area."""
    x0, y0, x1, y1 = region_to_pixels(region, active)
    return frame_rgb[y0:y1, x0:x1]


# First-pass calibration for the PokerGO classic-HSP graphics package.
# TUNE these against real frames via `poker-tell dump-regions`.
POKERGO_CLASSIC_HSP = OverlayLayout(
    name="pokergo_classic_hsp",
    pot=Region(0.02, 0.08, 0.18, 0.21),
    plate_x0=0.02,
    plate_x1=0.20,
    plate0_top=0.24,
    plate_height=0.135,
    max_plates=4,
    board_band=Region(0.61, 0.86, 0.87, 0.95),
    board_slots=5,
)
