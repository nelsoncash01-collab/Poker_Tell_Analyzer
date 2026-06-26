"""Card recognizer tests — numpy-only, no real video or OCR engine needed.

We build distinct synthetic glyph templates and synthetic card tiles (a rank
glyph on the left, a colored suit glyph on the right) and assert the recognizer
returns the right normalized card string.
"""

import numpy as np
import pytest

from poker_tell.broadcast.cards import (
    RANKS,
    load_templates,
    match_template,
    normalize_card_str,
    recognize_card,
    save_templates,
    suit_is_red,
)

rng = np.random.RandomState(0)


def _glyph(seed: int, size=(24, 24)) -> np.ndarray:
    """A deterministic distinct grayscale pattern for a template key."""
    r = np.random.RandomState(seed)
    return r.rand(*size) * 255


def _rank_templates() -> dict:
    return {r: _glyph(i + 1) for i, r in enumerate(RANKS)}


def _suit_glyphs() -> dict:
    # one distinct shape per suit (shared shape, color added at tile build time)
    return {s: _glyph(100 + i) for i, s in enumerate("cdhs")}


def test_normalize_card_str():
    assert normalize_card_str("10H") == "Th"
    assert normalize_card_str("ah") == "Ah"
    assert normalize_card_str(" Kd ") == "Kd"
    with pytest.raises(ValueError):
        normalize_card_str("ZZ")


def test_suit_is_red():
    red = np.zeros((10, 10, 3), dtype=np.uint8)
    red[..., 0] = 200
    black = np.full((10, 10, 3), 20, dtype=np.uint8)
    assert suit_is_red(red) is True
    assert suit_is_red(black) is False


def test_match_template_picks_closest():
    templates = {"A": _glyph(1), "K": _glyph(2), "Q": _glyph(3)}
    query = templates["K"].copy()  # exact match should win
    key, score = match_template(query, templates)
    assert key == "K"
    assert score > 0.99


def test_match_template_empty():
    assert match_template(np.zeros((4, 4)), {}) == (None, -1.0)


def _make_tile(rank_glyph, suit_glyph, red: bool, h=24, w=48):
    """Build a tile: rank glyph on left half, colored suit glyph on right half."""
    tile = np.zeros((h, w, 3), dtype=float)
    # left half = rank (gray)
    rg = np.repeat(rank_glyph[:, :, None], 3, axis=2)
    tile[:, : w // 2] = _resize_rgb(rg, h, w // 2)
    # right half = suit shape, colored
    sg = suit_glyph / 255.0
    sub = np.zeros((h, w - w // 2, 3))
    sg_r = _resize_gray(sg, h, w - w // 2)
    if red:
        sub[..., 0] = sg_r * 255
    else:
        sub[..., :] = (sg_r * 255)[..., None]
    tile[:, w // 2 :] = sub
    return tile


def _resize_gray(a, h, w):
    H, W = a.shape
    ys = np.clip((np.arange(h) * H / h).astype(int), 0, H - 1)
    xs = np.clip((np.arange(w) * W / w).astype(int), 0, W - 1)
    return a[ys][:, xs]


def _resize_rgb(a, h, w):
    return np.stack([_resize_gray(a[..., c], h, w) for c in range(3)], axis=2)


def test_recognize_card_red_and_black():
    ranks = _rank_templates()
    suits = _suit_glyphs()
    # Build an Ace of hearts tile: rank 'A' glyph + red 'h' suit glyph
    tile_ah = _make_tile(ranks["A"], suits["h"], red=True)
    assert recognize_card(tile_ah, ranks, suits) == "Ah"

    # King of spades: rank 'K' + black 's'
    tile_ks = _make_tile(ranks["K"], suits["s"], red=False)
    assert recognize_card(tile_ks, ranks, suits) == "Ks"


def test_recognize_card_color_restricts_suit():
    ranks = _rank_templates()
    suits = _suit_glyphs()
    # A red tile must never resolve to a black suit even if shapes are close.
    tile = _make_tile(ranks["7"], suits["d"], red=True)
    result = recognize_card(tile, ranks, suits)
    assert result[0] == "7"
    assert result[1] in "dh"  # red suit only


def test_recognize_card_without_templates_returns_none():
    assert recognize_card(np.zeros((10, 10, 3)), {}, {}) is None


def test_templates_round_trip(tmp_path):
    templates = {"rank_A": _glyph(1), "suit_h": _glyph(2)}
    save_templates(templates, tmp_path)
    loaded = load_templates(tmp_path)
    assert set(loaded) == {"rank_A", "suit_h"}
    assert np.allclose(loaded["rank_A"], templates["rank_A"])
