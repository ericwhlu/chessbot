"""MuJoCo scene, movable camera, and simulation-only ground-truth evaluation."""
from dataclasses import dataclass
import json
from pathlib import Path

import mujoco
import numpy as np

from .piece_pose import compare_to_sim
from .scene import HEIGHT, WIDTH, make_scene
from .vision import ChessVision, inverse_pose, rotation_about, transform


TAG_CONFIG = Path(__file__).with_name("tags.json")
ARM_CONFIG = Path(__file__).with_name("arm.json")
CV_TO_MJ = np.diag([1.0, -1.0, -1.0])


def look_at(offset, up=(0, 0, 1)):
    """Return camera-to-world rotation with MuJoCo camera axes."""
    back = offset / np.linalg.norm(offset)
    right = np.cross(up, back)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    return np.column_stack((right, np.cross(back, right), back))


@dataclass
class Camera:
    position: np.ndarray
    rotation: np.ndarray

    @classmethod
    def initial(cls, view_center):
        offset = np.array([-0.20, -0.33, 1.48])
        return cls(view_center + offset, look_at(offset, up=(0, 1, 0)))

    def drag(self, dx, dy, orbit_center):
        yaw_axis = np.array([0.0, 0.0, 1.0]) if orbit_center is not None else self.rotation[:, 1]
        delta = rotation_about(yaw_axis, -dx * 0.006) @ rotation_about(
            self.rotation[:, 0], -dy * 0.006
        )
        self.rotation = delta @ self.rotation
        if orbit_center is not None:
            self.position = orbit_center + delta @ (self.position - orbit_center)


def calibration_matrix(fovy):
    focal = (HEIGHT / 2) / np.tan(np.deg2rad(fovy) / 2)
    # OpenGL samples the first pixel at (0.5, 0.5); OpenCV numbers its center (0, 0).
    return np.array([[focal, 0, (WIDTH - 1) / 2],
                     [0, focal, (HEIGHT - 1) / 2],
                     [0, 0, 1]], dtype=float)


@dataclass
class PosePair:
    truth: np.ndarray
    estimate: np.ndarray | None

    def inverse(self):
        estimate = inverse_pose(self.estimate) if self.estimate is not None else None
        return PosePair(inverse_pose(self.truth), estimate)

    def errors(self):
        position_delta = self.estimate[:3, 3] - self.truth[:3, 3]
        delta = self.truth[:3, :3].T @ self.estimate[:3, :3]
        skew = np.array([delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0],
                         delta[1, 0] - delta[0, 1]]) / 2
        angle = np.arctan2(np.linalg.norm(skew), (np.trace(delta) - 1) / 2)
        return position_delta, float(np.rad2deg(angle))


@dataclass
class Observation:
    pose: PosePair
    ids: list[int]
    reprojection_px: float | None


@dataclass
class Frame:
    bgr: np.ndarray
    board: Observation
    base: Observation
    base_to_board: PosePair
    pieces: dict | None = None


class Simulation:
    def __init__(self, config, arm_config, fovy, *, movable_pieces=False):
        self.model, tag_detector, board_corners, base_corners, self.view_center = make_scene(
            config, arm_config, fovy, movable_pieces=movable_pieces
        )
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.keyframe("home").id)
        # Turn the parked arm away from the playing area.
        self.data.joint("joint_1").qpos[0] = -0.6
        self.data.actuator("joint_1").ctrl[0] = -0.6
        self.camera_id = self.model.camera("rgb").id
        self.intrinsics = calibration_matrix(self.model.cam_fovy[self.camera_id])
        self.renderer = mujoco.Renderer(self.model, height=HEIGHT, width=WIDTH)
        self.vision = ChessVision(tag_detector, self.intrinsics, board_corners, base_corners)

    @classmethod
    def from_files(cls, tags=TAG_CONFIG, arm=ARM_CONFIG, fovy=49, *, movable_pieces=False):
        return cls(json.loads(Path(tags).read_text()), json.loads(Path(arm).read_text()), fovy,
                   movable_pieces=movable_pieces)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.renderer.close()

    def render_camera(self, camera):
        self.data.mocap_pos[0] = camera.position
        mujoco.mju_mat2Quat(self.data.mocap_quat[0], camera.rotation.ravel())
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera_id)
        rgb = self.renderer.render()
        camera_to_world = transform(
            self.data.cam_xmat[self.camera_id].reshape(3, 3) @ CV_TO_MJ,
            self.data.cam_xpos[self.camera_id],
        )
        return rgb, inverse_pose(camera_to_world)

    def capture(self, camera, piece_detector=None, piece_layout=None):
        """Render from camera and pair image estimates with independent MuJoCo ground truth."""
        rgb, world_to_camera = self.render_camera(camera)
        observation = self.vision.infer(rgb, piece_detector)

        def evaluate(body_name, detected):
            body = self.data.body(body_name)
            truth = world_to_camera @ transform(body.xmat.reshape(3, 3), body.xpos)
            pose = detected.pose
            pair = PosePair(truth, pose.target_to_camera if pose is not None else None)
            return Observation(pair, detected.ids, pose.reprojection_px if pose is not None else None)

        board = evaluate("chessboard", observation.board)
        base = evaluate("base_link", observation.arm_base)
        truth_relative = inverse_pose(board.pose.truth) @ base.pose.truth
        relative = PosePair(truth_relative, observation.base_to_board)
        pieces = observation.pieces
        if pieces is not None and piece_layout is not None:
            pieces = compare_to_sim(pieces, piece_layout.truth())
        return Frame(observation.bgr, board, base, relative, pieces)
