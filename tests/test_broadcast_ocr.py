"""OCR-layer tests: pure string parsers + read_frame with an injected fake OCR.

No tesseract needed. We paint each overlay region with a unique sentinel gray
value and give read_frame a fake text-OCR that maps that value to canned text.
Card templates are omitted, so card reading is skipped (returns None) and never
calls the text OCR.
"""

import numpy as np

from poker_tell.broadcast.layout import (
    POKERGO_CLASSIC_HSP,
    detect_active_area,
    region_to_pixels,
)
from poker_tell.broadcast.ocr import FrameReading, parse_pot, parse_status, read_frame


def test_parse_pot():
    assert parse_pot("POT $203,800") == 203800
    assert parse_pot("$8,300") == 8300
    assert parse_pot("$12,200") == 12200
    assert parse_pot("no number here") is None


def test_parse_status_actions():
    s = parse_status("RAISE TO $3,000")
    assert (s.kind, s.action_type, s.amount) == ("action", "raise", 3000)
    s = parse_status("CALL $3,000")
    assert (s.kind, s.action_type, s.amount) == ("action", "call", 3000)
    assert parse_status("CHECK").action_type == "check"
    assert parse_status("FOLD").action_type == "fold"
    assert parse_status("BET $5,000").amount == 5000
    s = parse_status("ALL IN $50,000")
    assert (s.action_type, s.amount) == ("all_in", 50000)


def test_parse_status_check_fold_have_no_amount():
    assert parse_status("CHECK").amount is None
    assert parse_status("FOLD").amount is None


def test_parse_status_equity_winner_none():
    assert parse_status("65%").kind == "equity"
    assert parse_status("65%").equity_pct == 65
    assert parse_status("WINNER").kind == "winner"
    assert parse_status("").kind == "none"
    assert parse_status("   ").kind == "none"


def _frame_with_regions(layout, region_text: dict):
    """Build a pillarboxed frame; paint each named region a unique sentinel
    value and return (frame, fake_ocr) where fake_ocr maps value -> text."""
    frame = np.full((400, 700, 3), 150, dtype=np.uint8)
    frame[:, :60] = 0
    frame[:, 640:] = 0
    active = detect_active_area(frame)

    def region_of(key):
        if key == "pot":
            return layout.pot
        kind = "".join(c for c in key if c.isalpha())
        idx = int("".join(c for c in key if c.isdigit()))
        return layout.plate_regions(idx)[kind]

    value_to_text = {}
    value = 60
    for key, text in region_text.items():
        x0, y0, x1, y1 = region_to_pixels(region_of(key), active)
        frame[y0:y1, x0:x1] = value
        value_to_text[value] = text
        value += 7

    def fake_ocr(crop_img):
        return value_to_text.get(int(round(float(crop_img.mean()))), "")

    return frame, fake_ocr


def test_read_frame_assembles_pot_names_and_status():
    layout = POKERGO_CLASSIC_HSP
    frame, fake_ocr = _frame_with_regions(layout, {
        "pot": "POT $8,300",
        "name0": "IVEY",
        "status0": "RAISE TO $3,000",
        "name1": "DAGOSTINO",
        "status1": "CALL $3,000",
    })

    reading = read_frame(frame, layout, text_ocr=fake_ocr)
    assert isinstance(reading, FrameReading)
    assert reading.pot == 8300
    assert [s.name for s in reading.seats] == ["IVEY", "DAGOSTINO"]
    ivey = reading.seats[0]
    assert ivey.status.action_type == "raise" and ivey.status.amount == 3000
    assert reading.seats[1].status.action_type == "call"
    # No templates -> hole cards not read.
    assert all(s.hole_cards is None for s in reading.seats)
    # No board (slots are mid-gray, not card-bright, and no templates).
    assert reading.board == []


def test_read_frame_skips_empty_plates():
    layout = POKERGO_CLASSIC_HSP
    frame, fake_ocr = _frame_with_regions(layout, {"name0": "NEGREANU"})
    reading = read_frame(frame, layout, text_ocr=fake_ocr)
    assert len(reading.seats) == 1
    assert reading.seats[0].name == "NEGREANU"


def test_roster_snaps_noisy_name():
    layout = POKERGO_CLASSIC_HSP
    frame, fake_ocr = _frame_with_regions(layout, {"name0": "NEGREANI"})
    reading = read_frame(frame, layout, roster=["NEGREANU", "IVEY"],
                         text_ocr=fake_ocr)
    assert reading.seats[0].name == "NEGREANU"
