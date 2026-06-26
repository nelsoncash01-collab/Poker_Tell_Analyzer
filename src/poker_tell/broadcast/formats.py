"""Learn the POT banner's colour + location, and detect when they change.

The compilation can switch graphics formats/eras, where the POT banner changes
colour and/or moves. Rather than hardcode "maroon, top-left", we *learn* the
banner from each clip (anchored on the format-invariant ``$#,###`` amount, so it
works whatever colour the banner is) and flag a **format change** when the
learned colour or location drifts. That coarse segmentation sits above per-hand
segmentation and lets the per-frame detector use the right colour prior.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from poker_tell.broadcast.detect import Box

_POT_AMOUNT_RE = re.compile(r"\$?\s*\d[\d,]{2,}")


@dataclass
class FormatProfile:
    """The learned overlay format for one clip/era."""

    pot_color_hsv: tuple[int, int, int]
    pot_box: Box
    scale: float                 # banner width in px (the overlay scale unit)
    frame_shape: tuple[int, int]  # (H, W)

    def red_ranges(self, *, hue_tol: int = 12, s_min: int = 90, v_min: int = 40):
        """HSV ranges around the learned banner hue, for ``detect_overlays``."""
        h = int(self.pot_color_hsv[0])
        lo_h, hi_h = h - hue_tol, h + hue_tol
        ranges = []
        # handle hue wrap-around at 0/180
        if lo_h < 0:
            ranges.append(((180 + lo_h, s_min, v_min), (180, 255, 255)))
            lo_h = 0
        if hi_h > 180:
            ranges.append(((0, s_min, v_min), (hi_h - 180, 255, 255)))
            hi_h = 180
        ranges.append(((lo_h, s_min, v_min), (hi_h, 255, 255)))
        return tuple(ranges)


@dataclass
class FormatSegment:
    start_frame: int
    end_frame: int
    profile: FormatProfile


def _hue_dist(h1: float, h2: float) -> float:
    d = abs(h1 - h2)
    return min(d, 180 - d)


def profiles_differ(
    a: FormatProfile, b: FormatProfile, *, color_tol: float = 30.0,
    loc_tol: float = 0.06,
) -> bool:
    """True if two profiles differ enough to be considered a format change.

    Colour distance weights hue (×2, to put its 0-180 range on par with the
    0-255 sat/val) and location is the banner-centre distance normalized by the
    frame diagonal.
    """
    ha, sa, va = a.pot_color_hsv
    hb, sb, vb = b.pot_color_hsv
    color_d = ((2 * _hue_dist(ha, hb)) ** 2 + (sa - sb) ** 2 + (va - vb) ** 2) ** 0.5
    diag = (a.frame_shape[0] ** 2 + a.frame_shape[1] ** 2) ** 0.5 or 1.0
    loc_d = (((a.pot_box.cx - b.pot_box.cx) ** 2
              + (a.pot_box.cy - b.pot_box.cy) ** 2) ** 0.5) / diag
    return color_d > color_tol or loc_d > loc_tol


def segment_profiles(
    observations: list[tuple[int, FormatProfile | None]], **differ_kwargs
) -> list[FormatSegment]:
    """Collapse per-frame banner observations into format segments.

    ``observations`` is ``[(frame_index, profile_or_None)]`` in time order. A new
    segment starts whenever a profile differs from the current segment's. ``None``
    observations (banner not found) just extend the current segment.
    """
    segments: list[FormatSegment] = []
    cur: FormatSegment | None = None
    for idx, prof in observations:
        if prof is None:
            if cur is not None:
                cur.end_frame = idx
            continue
        if cur is None:
            cur = FormatSegment(idx, idx, prof)
        elif profiles_differ(cur.profile, prof, **differ_kwargs):
            segments.append(cur)
            cur = FormatSegment(idx, idx, prof)
        else:
            cur.end_frame = idx
    if cur is not None:
        segments.append(cur)
    return segments


def _dominant_hsv(banner_bgr: np.ndarray) -> tuple[int, int, int]:
    """Median HSV of a banner crop, ignoring near-white (text) pixels."""
    import cv2

    hsv = cv2.cvtColor(banner_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    # drop bright low-saturation (white text) pixels
    keep = hsv[(hsv[:, 1] > 60) & (hsv[:, 2] > 40)]
    use = keep if len(keep) else hsv
    med = np.median(use, axis=0)
    return int(med[0]), int(med[1]), int(med[2])


def observe_banner(frame_bgr: np.ndarray, *, top_frac: float = 0.35) -> FormatProfile | None:
    """Find the POT banner colour-agnostically via its ``$#,###`` amount text.

    Scans the top band with tesseract, locates the word reading a pot amount (or
    "POT"), boxes the banner around it, and samples its dominant colour. Returns
    ``None`` if no banner-like text is found. (Needs pytesseract; used by
    ``track_formats`` but injectable there for testing.)
    """
    from poker_tell.broadcast.ocr import _require_text_ocr

    pytesseract, Image = _require_text_ocr()
    H, W = frame_bgr.shape[:2]
    band = frame_bgr[: int(top_frac * H)]
    rgb = band[:, :, ::-1]  # BGR -> RGB for tesseract
    data = pytesseract.image_to_data(
        Image.fromarray(np.ascontiguousarray(rgb)),
        output_type=pytesseract.Output.DICT,
    )
    best = None
    for i, text in enumerate(data["text"]):
        t = (text or "").strip()
        if not t:
            continue
        if _POT_AMOUNT_RE.fullmatch(t) or t.upper() == "POT":
            x, y, w, h = (data["left"][i], data["top"][i],
                          data["width"][i], data["height"][i])
            if best is None or w * h > best[2] * best[3]:
                best = (x, y, w, h)
    if best is None:
        return None
    x, y, w, h = best
    # widen to the banner around the text, clamped to the frame
    bx0 = max(0, x - int(0.4 * w))
    bx1 = min(W, x + int(1.4 * w))
    by0 = max(0, y - int(0.6 * h))
    by1 = min(int(top_frac * H), y + int(1.6 * h))
    banner = Box(bx0, by0, bx1 - bx0, by1 - by0)
    color = _dominant_hsv(banner.crop(frame_bgr))
    return FormatProfile(pot_color_hsv=color, pot_box=banner, scale=float(banner.w),
                         frame_shape=(H, W))


def track_formats(
    source, *, sample_seconds: float = 2.0, observe=observe_banner, timeline=None,
) -> list[FormatSegment]:
    """Sample the video and return its format segments.

    ``source`` is a ``VideoSource`` (its path is read); ``timeline`` is its
    ``VideoTimeline`` (re-probed if not given). ``observe`` is injectable for
    testing.
    """
    from poker_tell.ingest import extract_frames, read_pts
    from poker_tell.video import VideoTimeline

    if timeline is None:
        pts, _ = read_pts(source.path)
        timeline = VideoTimeline(pts)
    n = timeline.n_frames
    fps = timeline.median_fps()
    stride = max(1, int(sample_seconds * fps))
    indices = list(range(0, n, stride))

    observations: list[tuple[int, FormatProfile | None]] = []
    for idx, frame in zip(indices, extract_frames(source, indices)):
        # extract_frames yields RGB; detection/observation expect BGR
        observations.append((idx, observe(frame[:, :, ::-1])))
    return segment_profiles(observations)
