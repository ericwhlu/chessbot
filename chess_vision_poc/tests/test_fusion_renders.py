"""Test independent Fusion top-view silhouettes on a calibrated overhead board."""
from pathlib import Path

import cv2
import numpy as np

from chess_vision.vision import Detector


def test_fusion_top_views():
    # The six Fusion exports use different zooms. These STEP widths calibrate
    # the fixture to a 50 mm tile. The detector only receives the assembled RGB.
    widths_mm = {"pawn": 30, "rook": 30, "knight": 43.3013, "bishop": 35.2671,
                 "queen": 47.5528, "king": 40}
    renders = Path(__file__).resolve().parents[2] / "models/chessbot/renders"
    size = 96
    board = np.full((8 * size, 8 * size, 3), 140, np.uint8)
    for col, (name, width) in enumerate(widths_mm.items(), start=1):
        gray = cv2.imread(str(renders / name / f"{name}_top.png"), cv2.IMREAD_GRAYSCALE)
        assert gray is not None
        contours, _ = cv2.findContours((gray < 220).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        outline = max(contours, key=cv2.contourArea)
        moments = cv2.moments(outline)
        center = np.array([moments["m10"], moments["m01"]]) / moments["m00"]
        scale = size * width / (50 * cv2.boundingRect(outline)[2])
        for row, level in [(2, 235), (5, 50)]:
            origin = size * np.array([col + 0.5, row + 0.5])
            points = np.rint((outline - center) * scale + origin).astype(np.int32)
            cv2.fillPoly(board, [points], (level,) * 3)
    end = size * 8
    result = Detector().detect(board, {"corners": [[0,0],[end,0],[end,end],[0,end]]})
    expected = ["........"] * 8
    expected[2], expected[5] = ".PRNBQK.", ".prnbqk."
    assert ["".join(rank) for rank in result["matrix"]] == expected
