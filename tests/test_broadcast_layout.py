import numpy as np
import pytest

from poker_tell.broadcast.layout import (
    POKERGO_CLASSIC_HSP,
    OverlayLayout,
    Region,
    crop,
    detect_active_area,
    region_to_pixels,
)


def test_region_validates_bounds():
    Region(0.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        Region(0.5, 0.0, 0.5, 1.0)  # zero width
    with pytest.raises(ValueError):
        Region(0.0, 0.0, 1.2, 1.0)  # out of range


def test_detect_active_area_finds_pillarbox():
    # 100x200 frame with 40px black bars left and right -> active x in [40,160)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[:, 40:160] = 180
    x0, y0, x1, y1 = detect_active_area(frame)
    assert (x0, x1) == (40, 160)
    assert (y0, y1) == (0, 100)


def test_detect_active_area_full_frame_when_no_bars():
    frame = np.full((50, 50, 3), 200, dtype=np.uint8)
    assert detect_active_area(frame) == (0, 0, 50, 50)


def test_region_to_pixels_maps_into_active_area():
    active = (40, 0, 160, 100)  # 120 wide, 100 tall
    px = region_to_pixels(Region(0.0, 0.0, 0.5, 1.0), active)
    assert px == (40, 0, 100, 100)


def test_crop_extracts_subarray():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[:, 40:160] = 180
    active = detect_active_area(frame)
    sub = crop(frame, Region(0.0, 0.0, 1.0, 1.0), active)
    assert sub.shape == (100, 120, 3)


def test_plate_regions_stack_without_overlap():
    lay = POKERGO_CLASSIC_HSP
    r0 = lay.plate_regions(0)
    r1 = lay.plate_regions(1)
    # plate 1 starts below plate 0
    assert r1["card1"].y0 >= r0["status"].y1 - 1e-9
    # each plate has the four expected sub-regions
    assert set(r0) == {"card1", "card2", "name", "status"}
    # card2 is to the right of card1
    assert r0["card2"].x0 > r0["card1"].x0


def test_board_slots_are_ordered_and_disjoint():
    lay = POKERGO_CLASSIC_HSP
    slots = [lay.board_slot(j) for j in range(lay.board_slots)]
    for a, b in zip(slots, slots[1:]):
        assert b.x0 > a.x0
        assert a.x1 <= b.x0 + 1e-9


def test_plate_and_board_index_bounds():
    lay = POKERGO_CLASSIC_HSP
    with pytest.raises(IndexError):
        lay.plate_regions(lay.max_plates)
    with pytest.raises(IndexError):
        lay.board_slot(lay.board_slots)
