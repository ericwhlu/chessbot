"""Parsing of operator input for operator_node.py. Pure Python, so it can be tested without ROS."""

from __future__ import annotations

from typing import Tuple

from .game import canonical_placement

COMMANDS = ("start", "reset", "retry")

HELP = """\
    (just Enter)        press the detect button: asks vision for one detection
                        (publishes "button" on chessbot/detect_request)
    start               start the game without waiting for the start position
    reset               start a new game
    retry               re-send the robot's move after an arm failure / retry the engine
    <placement FEN>     act as vision yourself: publish this placement on
                        chessbot/detected_fen (e.g. pasted from the detector's web demo)
    help                show this text
    quit                leave"""


def classify(line: str) -> Tuple[str, str]:
    """Map one line of operator input to ``(kind, payload)``.

    kind is one of: "detect", "cmd", "fen", "help", "quit", "unknown".
    """
    text = line.strip()
    low = text.lower()
    if text == "":
        return "detect", "button"
    if low in COMMANDS:
        return "cmd", low
    if low in ("q", "quit", "exit"):
        return "quit", ""
    if low in ("h", "help", "?"):
        return "help", ""
    if "/" in text:
        placement = canonical_placement(text)
        if placement is not None:
            return "fen", placement
    return "unknown", text
