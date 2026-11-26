"""The perception of chessbot_sim as a detector for chess_brain's vision_adapter_node:
camera image in, occupied squares out.

    ros2 run chess_brain vision_adapter_node --ros-args \\
        -p detector:=chessbot_sim_ros.camera_detector:detect -p output:=occupancy \\
        -p image_topic:=/camera/image_raw -p camera_info_topic:=/camera/camera_info

It uses chessbot_sim unchanged: ``vision.ChessVision`` finds the board from its four AprilTags,
``piece_pose.PieceDetector`` (a one-class YOLO model) finds the pieces on the board plane. What
this file adds is everything the simulation used to supply: the tag corners from
``chessbot_sim/tags.json``, the camera matrix from ``sensor_msgs/CameraInfo``, the colour order.

**Status: checked against frames rendered by the MuJoCo simulation only** (the same squares as
the simulation's own pipeline). It has never seen a real camera. On a real board it needs the
four tags printed and placed as in tags.json, and the YOLO weights were trained on rendered
images; whether they work on photographs is not known.

Settings, all optional, come from the node's ``detector_config`` parameter (a JSON object):

    tags          path of the tag layout file            (default: chessbot_sim/tags.json)
    square_size   size of one square in metres            (default 0.05)
    origin_xy     the a1 corner in the tag layout's frame (default [0, 0])
    surface_z     height of the playing surface in that frame (default 0.0)
    tag_height    how far the tags lie above the playing surface (default 0.0)
    weights       path of the YOLO weights                (default: chessbot_sim/assets/chess-pieces.pt)
    confidence    detection threshold                     (default 0.25)
    repo_root     where chessbot_sim is, if it is not found automatically
"""

from __future__ import annotations

from typing import List, Optional

from .paths import use_repo

_UNIT_CORNERS = ((-0.5, 0.5), (0.5, 0.5), (0.5, -0.5), (-0.5, -0.5))   # top left first, as the decoder reports them
_state: dict = {}   # one detector per set of settings; loading the YOLO model takes seconds


def tag_corners(tags: list, z: float) -> dict:
    """Corner coordinates (4 x 3, metres) of every tag in the layout, keyed by tag id, in the
    order the AprilTag decoder reports them. Same convention as chessbot_sim/scene.py."""
    import math

    import numpy as np

    corners = {}
    for tag in tags:
        cx, cy = (float(v) for v in tag["center_xy_m"])
        size = float(tag["size_m"])
        yaw = math.radians(float(tag.get("rotation_deg", 0.0)))
        c, s = math.cos(yaw), math.sin(yaw)
        points = [(cx + size * (c * ux - s * uy), cy + size * (s * ux + c * uy), z) for ux, uy in _UNIT_CORNERS]
        corners[int(tag["id"])] = np.array(points, dtype=float)
    return corners


def _build(config: dict) -> dict:
    """Everything that does not change from frame to frame."""
    import json
    from pathlib import Path

    import cv2

    root = use_repo(str(config.get("repo_root", "")))
    from chessbot_sim.piece_pose import WEIGHTS, BoardPlane, PieceDetector

    layout_file = Path(config.get("tags") or Path(root) / "chessbot_sim" / "tags.json")
    layout = json.loads(layout_file.read_text())
    dictionary_id = getattr(cv2.aruco, "DICT_APRILTAG_" + str(layout["tag_family"]), None)
    if dictionary_id is None:
        raise ValueError(f"unknown AprilTag family {layout['tag_family']!r} in {layout_file}")
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    surface = float(config.get("surface_z", 0.0))
    plane = BoardPlane(
        float(config.get("square_size", 0.05)), surface, tuple(config.get("origin_xy", (0.0, 0.0)))
    )
    return {
        "tags": cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary_id), parameters),
        "corners": tag_corners(layout["tags"], surface + float(config.get("tag_height", 0.0))),
        "pieces": PieceDetector(
            plane, weights=config.get("weights") or WEIGHTS, confidence=float(config.get("confidence", 0.25))
        ),
    }


def detect(image_bgr, camera) -> Optional[List[str]]:
    """The squares that hold a piece, or None if the board's tags cannot be found in the image.

    ``camera`` is what vision_adapter_node hands over: the intrinsic matrix under ``"k"`` (required),
    distortion coefficients under ``"d"``, the settings above under ``"config"``.
    """
    if not camera or "k" not in camera:
        raise ValueError("this detector needs the camera matrix: start the node with camera_info_topic")

    import json

    import cv2
    import numpy as np

    config = dict(camera.get("config") or {})
    key = json.dumps(config, sort_keys=True, default=str)
    if key not in _state:
        _state.clear()
        _state[key] = _build(config)
    built = _state[key]

    from chessbot_sim.vision import ChessVision

    intrinsics = np.array(camera["k"], dtype=float).reshape(3, 3)
    rgb = cv2.cvtColor(np.asarray(image_bgr), cv2.COLOR_BGR2RGB)
    distortion = np.array(camera.get("d") or [], dtype=float)
    if distortion.size and np.any(distortion != 0.0):
        rgb = cv2.undistort(rgb, intrinsics, distortion)   # chessbot_sim expects an undistorted image
    frame = ChessVision(built["tags"], intrinsics, built["corners"], {}).infer(rgb, built["pieces"])
    if frame.pieces is None or frame.pieces.get("state") != "ready":
        return None
    return sorted(detection["square"] for detection in frame.pieces["detections"])
