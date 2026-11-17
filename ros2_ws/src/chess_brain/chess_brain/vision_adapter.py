"""Turn a camera image into a placement FEN by calling one detector function. Pure Python
(numpy is only needed when an image is actually converted).

This is the vision team's plug-in point. The detector is the ONE function they write:

    def detect(image_bgr):
        '''Return the piece-placement FEN seen in the image.

        image_bgr : numpy array, shape (height, width, 3), dtype uint8, BGR order (OpenCV style).

        Return the placement, rank 8 first, a1 at the bottom-left, e.g.
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
        Return None if the board cannot be read (occluded, low confidence). Raising also counts
        as "cannot read". A full FEN is accepted too; only the placement field is used.
        '''

Everything else (subscribing to the camera, the detect button, publishing) is done by
vision_adapter_node.py.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

from .game import canonical_placement

Detector = Callable[[object], Optional[str]]

START_PLACEMENT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

# encoding -> (channels, index of B, G, R inside one pixel)
_COLOR_ENCODINGS = {
    "bgr8": (3, (0, 1, 2)),
    "rgb8": (3, (2, 1, 0)),
    "bgra8": (4, (0, 1, 2)),
    "rgba8": (4, (2, 1, 0)),
}


def image_to_bgr(height: int, width: int, encoding: str, step: int, data):
    """Convert the fields of a ``sensor_msgs/Image`` to a (height, width, 3) uint8 BGR array.

    Supports bgr8, rgb8, bgra8, rgba8 and mono8, including row padding (``step`` > width*channels).
    """
    import numpy as np

    enc = str(encoding).strip().lower()
    height, width, step = int(height), int(width), int(step)
    if height <= 0 or width <= 0:
        raise ValueError(f"empty image ({width}x{height})")
    if enc == "mono8":
        channels = 1
    elif enc in _COLOR_ENCODINGS:
        channels = _COLOR_ENCODINGS[enc][0]
    else:
        raise ValueError(f"unsupported image encoding {encoding!r} (use bgr8, rgb8, bgra8, rgba8 or mono8)")
    if step < width * channels:
        raise ValueError(f"step {step} is too small for width {width} with {channels} channel(s)")

    buffer = np.frombuffer(bytes(data), dtype=np.uint8)
    if buffer.size < height * step:
        raise ValueError(f"image data has {buffer.size} bytes, expected {height * step}")
    rows = buffer[: height * step].reshape(height, step)[:, : width * channels]
    if channels == 1:
        gray = rows.reshape(height, width)
        return np.stack([gray, gray, gray], axis=-1).copy()
    pixels = rows.reshape(height, width, channels)
    return pixels[:, :, list(_COLOR_ENCODINGS[enc][1])].copy()


def run_detector(detector: Detector, image) -> Tuple[Optional[str], str]:
    """Call the detector and validate its answer. Returns ``(placement or None, reason)``."""
    try:
        result = detector(image)
    except Exception as exc:  # the detector failed: report it, never crash the node
        return None, f"detector raised {type(exc).__name__}: {exc}"
    if result is None:
        return None, "detector returned None (board not readable)"
    placement = canonical_placement(result) if isinstance(result, str) else None
    if placement is None:
        return None, f"detector returned something that is not a placement FEN: {str(result)[:80]!r}"
    return placement, "ok"


def example_detector(image_bgr) -> str:
    """Placeholder that always "sees" the start position. For testing the plumbing only."""
    return START_PLACEMENT
