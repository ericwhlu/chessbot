"""Turn a camera image into what chess_brain reads, by calling one detector function: a placement
FEN, or just the occupied squares. Pure Python (numpy is only needed when an image is actually
converted).

This is the vision team's plug-in point. The detector is the ONE function they write:

    def detect(image_bgr):
        '''Return the piece-placement FEN seen in the image.

        image_bgr : numpy array, shape (height, width, 3), dtype uint8, BGR order (OpenCV style).

        Return the placement, rank 8 first, a1 at the bottom-left, e.g.
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
        Return None if the board cannot be read (occluded, low confidence). Raising also counts
        as "cannot read". A full FEN is accepted too; only the placement field is used.
        '''

A detector that cannot tell the pieces apart returns the occupied squares instead (the node is then
started with ``output:=occupancy`` and publishes on ``chessbot/detected_occupancy``):

    def detect(image_bgr):
        '''Return the names of the squares that hold a piece, e.g. ["a1", "b1", "e4"], in any
        order, or None if the board cannot be read.'''

Either kind of detector may take a second parameter, the camera:

    def detect(image_bgr, camera):
        '''camera : {"k": 3x3 intrinsic matrix (rows), "d": distortion coefficients,
                     "width", "height", "frame_id", "config": dict}, from the
                     sensor_msgs/CameraInfo topic given to the node; "config" is the node's
                     ``detector_config`` parameter (JSON) for the detector's own settings.
                     None if no camera info has arrived.'''

Everything else (subscribing to the camera, the detect button, publishing) is done by
vision_adapter_node.py.
"""

from __future__ import annotations

import inspect
from typing import Callable, List, Optional, Tuple

from .game import canonical_placement
from .occupancy import normalise

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


def wants_camera(detector) -> bool:
    """Does the detector take a second positional parameter (the camera)?"""
    try:
        parameters = list(inspect.signature(detector).parameters.values())
    except (TypeError, ValueError):
        return False
    positional = [p for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return len(positional) >= 2 or any(p.kind is p.VAR_POSITIONAL for p in parameters)


def camera_summary(width: int, height: int, k, d, frame_id: str = "", config: Optional[dict] = None) -> dict:
    """The fields of a ``sensor_msgs/CameraInfo`` a detector needs, as plain Python."""
    k = [float(v) for v in k]
    if len(k) != 9:
        raise ValueError(f"camera matrix has {len(k)} entries, expected 9")
    return {
        "k": [k[0:3], k[3:6], k[6:9]],
        "d": [float(v) for v in d],
        "width": int(width),
        "height": int(height),
        "frame_id": str(frame_id),
        "config": dict(config or {}),
    }


def _call(detector, image, camera):
    return detector(image, camera) if wants_camera(detector) else detector(image)


def run_occupancy_detector(detector, image, camera: Optional[dict] = None) -> Tuple[Optional[List[str]], str]:
    """Call a detector that reports occupied squares and validate its answer. Returns
    ``(sorted square names or None, reason)``."""
    try:
        result = _call(detector, image, camera)
    except Exception as exc:  # the detector failed: report it, never crash the node
        return None, f"detector raised {type(exc).__name__}: {exc}"
    if result is None:
        return None, "detector returned None (board not readable)"
    if isinstance(result, (str, bytes)) or not hasattr(result, "__iter__"):
        return None, f"detector returned {str(result)[:60]!r}, not a list of squares (is this a FEN detector? use output:=fen)"
    squares = normalise(result)
    if squares is None:
        return None, f"detector returned something that is not a list of square names: {str(list(result))[:80]}"
    return sorted(squares), "ok"


def run_detector(detector: Detector, image, camera: Optional[dict] = None) -> Tuple[Optional[str], str]:
    """Call the detector and validate its answer. Returns ``(placement or None, reason)``."""
    try:
        result = _call(detector, image, camera)
    except Exception as exc:  # the detector failed: report it, never crash the node
        return None, f"detector raised {type(exc).__name__}: {exc}"
    if result is None:
        return None, "detector returned None (board not readable)"
    placement = canonical_placement(result) if isinstance(result, str) else None
    if placement is None:
        return None, f"detector returned something that is not a placement FEN: {str(result)[:80]!r}"
    return placement, "ok"


def superseded(image_at: float, request_waiting: bool, request_at: float) -> bool:
    """Is a finished detection out of date? Yes if a request is still waiting that was made
    after the picture was taken: the asker wants the board as it is now (for instance after
    the arm has left it), and the request will produce the next detection."""
    return bool(request_waiting) and image_at < request_at


def example_detector(image_bgr) -> str:
    """Placeholder that always "sees" the start position. For testing the plumbing only."""
    return START_PLACEMENT
