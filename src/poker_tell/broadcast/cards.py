"""Recognize a playing card from its on-screen tile (rank + suit).

The broadcast renders each card as a clean tile: a rank glyph and a suit symbol,
the suit colored red (hearts/diamonds) or black (spades/clubs). Recognition is:

1. classify the suit *color* (red vs black) from the suit sub-region — halves the
   suit candidates;
2. template-match the suit *shape* among the two same-color suits;
3. template-match the rank glyph against the 13 rank templates.

Templates are small grayscale arrays calibrated from real frames (see
``load_templates`` / ``save_templates``). This is numpy-only on purpose so the
recognizer is testable without OpenCV. It reads broadcast graphics — it is not a
behavioral model and nothing here is shared across players.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

RANKS = "23456789TJQKA"
SUITS = "cdhs"
RED_SUITS = "dh"
BLACK_SUITS = "cs"

_RANK_ALIASES = {"10": "T", "1": "T"}


def normalize_card_str(s: str) -> str:
    """Normalize a textual card like ``'10H'`` or ``'ah'`` to ``'Th'`` / ``'Ah'``.

    Rank is upper-cased (ten -> ``T``), suit lower-cased. Raises on anything that
    isn't a valid rank+suit.
    """
    s = s.strip()
    if len(s) >= 2:
        suit = s[-1].lower()
        rank = s[:-1].upper()
        rank = _RANK_ALIASES.get(rank, rank)
        if rank in RANKS and suit in SUITS:
            return rank + suit
    raise ValueError(f"not a valid card string: {s!r}")


def to_gray(img_rgb: np.ndarray) -> np.ndarray:
    return img_rgb.astype(float).mean(axis=2)


def _resize(a: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest-neighbour resize (numpy only)."""
    H, W = a.shape
    ys = np.clip((np.arange(h) * H / h).astype(int), 0, H - 1)
    xs = np.clip((np.arange(w) * W / w).astype(int), 0, W - 1)
    return a[ys][:, xs]


def _normvec(a: np.ndarray, size: tuple[int, int] = (32, 32)) -> np.ndarray:
    v = _resize(a, *size).ravel().astype(float)
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n else v


def match_template(
    img_gray: np.ndarray, templates: dict[str, np.ndarray],
    *, size: tuple[int, int] = (32, 32)
) -> tuple[str | None, float]:
    """Return the best-matching template key and its correlation score.

    Score is the normalized cross-correlation (dot of zero-mean unit vectors),
    in [-1, 1]; higher is a better match. Empty ``templates`` yields ``(None, -1)``.
    """
    if not templates:
        return None, -1.0
    q = _normvec(img_gray, size)
    best_key, best = None, -np.inf
    for key, tmpl in templates.items():
        score = float(q @ _normvec(tmpl, size))
        if score > best:
            best_key, best = key, score
    return best_key, float(best)


def suit_is_red(suit_rgb: np.ndarray, *, margin: float = 12.0) -> bool:
    """True if the suit sub-region is predominantly red (hearts/diamonds)."""
    r = float(suit_rgb[..., 0].mean())
    g = float(suit_rgb[..., 1].mean())
    b = float(suit_rgb[..., 2].mean())
    return r > g + margin and r > b + margin


def _crop_frac(img: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def recognize_card(
    tile_rgb: np.ndarray,
    rank_templates: dict[str, np.ndarray],
    suit_templates: dict[str, np.ndarray],
    *,
    rank_box: tuple[float, float, float, float] = (0.0, 0.0, 0.62, 1.0),
    suit_box: tuple[float, float, float, float] = (0.55, 0.0, 1.0, 1.0),
) -> str | None:
    """Recognize a card tile, returning a normalized string like ``'Ah'``.

    ``rank_box`` / ``suit_box`` locate the rank glyph and suit symbol within the
    tile (fractions). Returns ``None`` if either template set is empty.
    """
    if not rank_templates or not suit_templates:
        return None
    gray = to_gray(tile_rgb)
    rank_img = _crop_frac(gray, rank_box)
    suit_rgb = _crop_frac(tile_rgb, suit_box)
    suit_gray = to_gray(suit_rgb)

    wanted = RED_SUITS if suit_is_red(suit_rgb) else BLACK_SUITS
    suit_candidates = {k: v for k, v in suit_templates.items() if k in wanted}
    if not suit_candidates:  # fall back to all if templates miss this color
        suit_candidates = suit_templates

    rank, _ = match_template(rank_img, rank_templates)
    suit, _ = match_template(suit_gray, suit_candidates)
    if rank is None or suit is None:
        return None
    return rank + suit


# --- template persistence (grayscale .npy) ---------------------------------


def save_templates(templates: dict[str, np.ndarray], out_dir: Path | str) -> None:
    """Save each template as ``<out_dir>/<key>.npy`` (grayscale float array)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for key, arr in templates.items():
        np.save(out / f"{key}.npy", np.asarray(arr, dtype=float))


def load_templates(in_dir: Path | str) -> dict[str, np.ndarray]:
    """Load ``<in_dir>/<key>.npy`` files into a ``{key: array}`` dict."""
    in_dir = Path(in_dir)
    return {p.stem: np.load(p) for p in sorted(in_dir.glob("*.npy"))}
