"""camera_detector: the detector plug-in around chessbot_sim's perception.

The tag geometry is checked anywhere numpy is installed. The comparison with the simulation
needs MuJoCo, OpenCV and the YOLO weights, so it is skipped where they are missing (it runs in
the simulation's Python environment and in the containers).
"""

import random
import tempfile

import chess
import pytest

np = pytest.importorskip("numpy")

from chessbot_sim_ros.camera_detector import detect, tag_corners  # noqa: E402

TAGS = [
    {"id": 0, "center_xy_m": [0.030, 0.460], "size_m": 0.040, "rotation_deg": 0},
    {"id": 2, "center_xy_m": [0.370, -0.060], "size_m": 0.040, "rotation_deg": 90},
]


def test_tag_corners_start_at_the_top_left_and_go_clockwise():
    corners = tag_corners(TAGS, 0.0067)
    expected = [[0.01, 0.48, 0.0067], [0.05, 0.48, 0.0067], [0.05, 0.44, 0.0067], [0.01, 0.44, 0.0067]]
    assert np.allclose(corners[0], expected, atol=1e-12)


def test_a_rotated_tag_has_rotated_corners():
    corners = tag_corners(TAGS, 0.0)[2]
    # turned a quarter turn counter-clockwise, the tag's own top-left corner is at the bottom left
    assert np.allclose(corners[0], [0.35, -0.08, 0.0], atol=1e-12)
    assert np.allclose(corners[1], [0.35, -0.04, 0.0], atol=1e-12)


def test_the_detector_refuses_to_guess_the_camera():
    with pytest.raises(ValueError, match="camera matrix"):
        detect(np.zeros((4, 4, 3), dtype=np.uint8), None)


def test_same_squares_as_the_simulation_sees():
    """Frames rendered by the simulation, through the plug-in (image + camera matrix only) and
    through the simulation's own pipeline: the occupied squares must be the same, and they must
    be the squares the pieces really stand on."""
    pytest.importorskip("mujoco")
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("ultralytics")
    from chessbot_sim_ros.paths import use_repo

    use_repo()
    from chess_brain.occupancy import squares_of
    from chessbot_sim.world import MujocoWorld

    with tempfile.TemporaryDirectory() as out:
        world = MujocoWorld(out=out, robot="black", seed=3, script="", video=False)
        try:
            sim, camera = world.sim, world.match.camera
            board = chess.Board()
            rng = random.Random(5)
            for ply in range(9):
                if ply:                       # move the pieces by hand, both colours
                    move = rng.choice(list(board.legal_moves))
                    world.play_human(board, move)
                    board.push(move)
                rgb, _ = sim.render_camera(camera)
                image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                info = {"k": sim.intrinsics.tolist(), "d": [], "width": rgb.shape[1], "height": rgb.shape[0]}
                seen = detect(image_bgr, info)
                assert seen is not None, f"ply {ply}: the board was not found"
                assert frozenset(seen) == world.scan(), f"ply {ply}: differs from the simulation's own pipeline"
                assert frozenset(seen) == squares_of(board), f"ply {ply}: differs from the pieces on the board"
        finally:
            world.close()
