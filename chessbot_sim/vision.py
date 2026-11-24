"""Single-frame AprilTag and YOLO perception, with no simulator state input."""
from dataclasses import dataclass

import cv2
import numpy as np


def transform(rotation, translation):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(translation).reshape(3)
    return matrix


def inverse_pose(matrix):
    rotation = matrix[:3, :3].T
    return transform(rotation, -rotation @ matrix[:3, 3])


def rotation_about(axis, angle):
    return cv2.Rodrigues(np.asarray(axis, dtype=float) * angle)[0]


@dataclass
class Estimate:
    target_to_camera: np.ndarray
    reprojection_px: float


def estimate_pose(corners, ids, corner_map, intrinsics):
    if not ids:
        return None
    target_points = np.concatenate([corner_map[tag_id] for tag_id in ids])
    plane_origin = target_points.mean(axis=0)
    plane_points = target_points - plane_origin
    # IPPE expects planar coordinates. Keep the target height in the final transform.
    plane_points[:, 2] = 0
    image_points = np.concatenate(corners).reshape(-1, 2).astype(np.float64)
    _, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        plane_points, image_points, intrinsics, None, flags=cv2.SOLVEPNP_IPPE
    )
    best = None
    for rvec, tvec in zip(rvecs, tvecs):
        if not (np.isfinite(rvec).all() and np.isfinite(tvec).all()):
            continue
        rvec, tvec = cv2.solvePnPRefineLM(plane_points, image_points, intrinsics, None, rvec, tvec)
        rotation = cv2.Rodrigues(rvec)[0]
        extrinsic = transform(rotation, tvec.ravel() - rotation @ plane_origin)
        depth = (plane_points @ rotation.T)[:, 2] + tvec[2, 0]
        if np.any(depth <= 0) or inverse_pose(extrinsic)[2, 3] <= target_points[0, 2]:
            continue
        projected = cv2.projectPoints(plane_points, rvec, tvec, intrinsics, None)[0]
        residuals = projected.reshape(-1, 2) - image_points
        rms = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))
        if np.isfinite(rms) and (best is None or rms < best.reprojection_px):
            best = Estimate(extrinsic, rms)
    return best


def draw_axes(image, intrinsics, pose, length):
    axes = np.vstack((np.zeros(3), np.eye(3) * length))
    camera_points = axes @ pose[:3, :3].T + pose[:3, 3]
    if np.any(camera_points[:, 2] <= 0):
        return
    pixels = np.rint((camera_points @ intrinsics.T)[:, :2] / camera_points[:, 2:])
    if np.any(pixels < 0) or np.any(pixels >= [image.shape[1], image.shape[0]]):
        return
    origin, *ends = pixels.astype(int)
    for end, color in zip(ends, [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        cv2.line(image, tuple(origin), tuple(end), color, 2, cv2.LINE_AA)


@dataclass
class TargetDetection:
    ids: list[int]
    pose: Estimate | None


@dataclass
class VisionFrame:
    bgr: np.ndarray
    board: TargetDetection
    arm_base: TargetDetection
    base_to_board: np.ndarray | None
    pieces: dict | None


class ChessVision:
    """Process a calibrated, undistorted RGB image using known metric tag corners."""

    def __init__(self, detector, intrinsics, board_corners, base_corners):
        self.detector = detector
        self.intrinsics = intrinsics
        self.board_corners = board_corners
        self.base_corners = base_corners

    def infer(self, rgb, piece_detector=None):
        """Return estimates and a BGR overlay from the supplied frame."""
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        detected = dict(zip(ids.ravel(), corners)) if ids is not None else {}
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        def observe(corner_map, color, axis_length):
            matched = sorted(int(tag_id) for tag_id in detected.keys() & corner_map.keys())
            points = [detected[tag_id] for tag_id in matched]
            pose = estimate_pose(points, matched, corner_map, self.intrinsics)
            if matched:
                cv2.aruco.drawDetectedMarkers(bgr, points, np.asarray(matched, dtype=np.int32), color)
            if pose is not None:
                draw_axes(bgr, self.intrinsics, pose.target_to_camera, axis_length)
            return TargetDetection(matched, pose)

        board = observe(self.board_corners, (95, 220, 120), 0.065)
        base = observe(self.base_corners, (225, 190, 60), 0.085)
        board_pose = board.pose.target_to_camera if board.pose is not None else None
        base_to_board = None
        if board_pose is not None and base.pose is not None:
            base_to_board = inverse_pose(board_pose) @ base.pose.target_to_camera

        pieces = None
        if piece_detector is not None:
            pieces = piece_detector.scan(rgb, bgr, self.intrinsics, board_pose)
            if base_to_board is not None:
                board_to_base = inverse_pose(base_to_board)
                for detection in pieces["detections"]:
                    detection["xyz_arm_m"] = (
                        board_to_base[:3, :3] @ detection["xyz_m"] + board_to_base[:3, 3]
                    ).tolist()
        return VisionFrame(bgr, board, base, base_to_board, pieces)
