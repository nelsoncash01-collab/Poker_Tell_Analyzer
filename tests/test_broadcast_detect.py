"""Detection tests against synthetic frames (needs OpenCV, not the real video).

We draw the graphics' signatures — a maroon POT banner, maroon name strips,
white card tiles, plus chip/cash decoys — at chosen positions and assert
detect_overlays finds them. A second frame uses a different banner colour and
position to confirm the color prior (FormatProfile) makes detection adaptive.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from poker_tell.broadcast.detect import detect_overlays  # noqa: E402
from poker_tell.broadcast.formats import FormatProfile  # noqa: E402
from poker_tell.broadcast.detect import Box  # noqa: E402


def _draw_card(frame, x, y, w=36, h=52):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (240, 240, 240), -1)


def make_frame(H=480, W=854, banner_bgr=(40, 40, 160), banner_x=30, banner_y=18):
    """Synthetic broadcast-style frame (BGR) with two seats and a 5-card board."""
    frame = np.full((H, W, 3), 30, np.uint8)  # dark felt
    # POT banner (top-left)
    cv2.rectangle(frame, (banner_x, banner_y), (banner_x + 160, banner_y + 44),
                  banner_bgr, -1)
    # Two seats: two white card tiles, then a name strip just below them.
    for sy in (70, 170):
        _draw_card(frame, 40, sy)
        _draw_card(frame, 82, sy)
        cv2.rectangle(frame, (36, sy + 54), (36 + 150, sy + 54 + 28),
                      banner_bgr, -1)  # name strip
    # Board: five white tiles, bottom-centre.
    for j in range(5):
        _draw_card(frame, 300 + j * 46, 400)
    # Decoys: a white chip (round) and green cash (wrong colour).
    cv2.circle(frame, (700, 120), 25, (240, 240, 240), -1)
    cv2.rectangle(frame, (640, 360), (720, 410), (60, 140, 60), -1)
    return frame


def test_detects_pot_plates_board():
    det = detect_overlays(make_frame())
    assert det.pot is not None
    assert det.pot.y < 60  # topmost red bar
    assert len(det.seats) == 2
    # each seat got its two hole-card tiles, ordered left to right
    for seat in det.seats:
        assert seat.card1 is not None and seat.card2 is not None
        assert seat.card1.x < seat.card2.x
        assert seat.status is not None
    assert len(det.board) == 5
    # board tiles are ordered left to right
    xs = [b.x for b in det.board]
    assert xs == sorted(xs)


def test_round_chip_and_cash_are_not_cards():
    det = detect_overlays(make_frame())
    # 2 seats x 2 cards + 5 board = 9 card tiles; the chip/cash must be excluded.
    n_cards = sum(c is not None for s in det.seats for c in (s.card1, s.card2))
    assert n_cards + len(det.board) == 9


def test_pot_anchor_is_topmost_left_bar():
    det = detect_overlays(make_frame(banner_x=30, banner_y=18))
    # the name strips sit below the banner
    assert all(s.name.y > det.pot.y for s in det.seats)


def test_color_prior_makes_detection_adaptive():
    # A blue banner is NOT found with the default maroon ranges...
    blue = make_frame(banner_bgr=(170, 40, 40))  # BGR: strong blue
    assert detect_overlays(blue).pot is None

    # ...but IS found when seeded with a profile whose hue matches blue (~120).
    profile = FormatProfile(pot_color_hsv=(120, 200, 170),
                            pot_box=Box(30, 18, 160, 44), scale=160.0,
                            frame_shape=(480, 854))
    det = detect_overlays(blue, red_ranges=profile.red_ranges())
    assert det.pot is not None
    assert len(det.seats) == 2
