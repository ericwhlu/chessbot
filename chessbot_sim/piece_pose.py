"""One-class YOLO piece detection and metric board-plane localization."""
from pathlib import Path

import cv2
import numpy as np


RECT_SIZE = 384
WEIGHTS = Path(__file__).with_name("assets") / "chess-pieces.pt"


class BoardPlane:
    """Metric geometry of an eight-by-eight board in its own coordinate frame."""

    def __init__(self, square_size_m, surface_z_m, origin_xy_m=(0, 0)):
        self.square_size = float(square_size_m)
        self.surface = float(surface_z_m)
        self.minimum = np.asarray(origin_xy_m, dtype=float)
        span = 8 * self.square_size
        self.maximum = self.minimum + span
        padding = (0.065 / 0.4) * span
        self.rect_min = self.minimum - padding
        self.rect_max = self.maximum + padding
        self.rect_scale = (RECT_SIZE - 1) / (span + 2 * padding)

    @classmethod
    def from_model(cls, model):
        a1 = model.geom("square_a1")
        return cls(2 * a1.size[0], a1.pos[2] + a1.size[2], a1.pos[:2] - a1.size[:2])

    def square_center(self, square):
        indices = np.array([ord(square[0]) - ord("a"), int(square[1]) - 1])
        xy = self.minimum + (indices + 0.5) * self.square_size
        return np.array([*xy, self.surface])

    def square_name(self, xyz):
        if not np.isfinite(xyz).all() or not np.all((xyz[:2] >= self.minimum) & (xyz[:2] < self.maximum)):
            return None
        file, rank = np.floor((xyz[:2] - self.minimum) / self.square_size).astype(int)
        return f"{chr(ord('a') + file)}{rank + 1}"

    def projection(self, intrinsics, board_to_camera):
        # H maps [x_board, y_board, 1] on the known piece support plane to image pixels.
        rotation = board_to_camera[:3, :3]
        translation = board_to_camera[:3, 3] + self.surface * rotation[:, 2]
        return intrinsics @ np.column_stack((rotation[:, :2], translation))

    def image_to_rect(self, shape, intrinsics, board_to_camera):
        s = self.rect_scale
        low, high = self.rect_min, self.rect_max
        corners = np.array([[x, y, self.surface] for x in (low[0], high[0]) for y in (low[1], high[1])])
        camera = corners @ board_to_camera[:3, :3].T + board_to_camera[:3, 3]
        if np.any(camera[:, 2] < 0.025):
            return None
        image_xy = (camera @ intrinsics.T)[:, :2] / camera[:, 2:]
        size = np.ptp(image_xy, axis=0)
        if np.min(size) < 36:
            return None
        width, height = shape[1], shape[0]
        if (image_xy[:, 0].max() < 0 or image_xy[:, 0].min() >= width
                or image_xy[:, 1].max() < 0 or image_xy[:, 1].min() >= height):
            return None
        board_to_rect = np.array([[s, 0, -s * low[0]], [0, -s, s * high[1]], [0, 0, 1]])
        return board_to_rect @ np.linalg.inv(self.projection(intrinsics, board_to_camera))

    def rect_to_board(self, points):
        pixels = np.asarray(points)
        return np.column_stack((self.rect_min[0] + pixels[:, 0] / self.rect_scale,
                                self.rect_max[1] - pixels[:, 1] / self.rect_scale,
                                np.full(len(pixels), self.surface)))

    def board_to_rect(self, points):
        points = np.asarray(points)
        return np.column_stack(((points[:, 0] - self.rect_min[0]) * self.rect_scale,
                                (self.rect_max[1] - points[:, 1]) * self.rect_scale))


class PieceDetector:
    def __init__(self, plane, weights=WEIGHTS, confidence=0.25):
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(min(torch.get_num_threads(), 4))
        self.plane = plane
        self.model = YOLO(str(weights), task="pose")
        self.confidence = confidence

    def scan(self, rgb, overlay, intrinsics, board_to_camera):
        if board_to_camera is None:
            return {"state": "board_pose_unavailable", "detections": []}
        h = self.plane.image_to_rect(rgb.shape, intrinsics, board_to_camera)
        if h is None:
            return {"state": "board_pose_unavailable", "detections": []}
        rectified = cv2.warpPerspective(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), h,
                                        (RECT_SIZE, RECT_SIZE))
        result = self.model.predict(rectified, imgsz=RECT_SIZE, device="cpu", verbose=False,
                                    conf=self.confidence, iou=0.5, max_det=64)[0]
        if result.keypoints is None or not len(result.keypoints):
            return {"state": "ready", "detections": []}

        keypoints = result.keypoints.data.cpu().numpy()[:, 0, :]
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        xyz = self.plane.rect_to_board(keypoints[:, :2])
        rect_to_image = np.linalg.inv(h)
        detections = {}
        for i in np.argsort(-scores):
            square = self.plane.square_name(xyz[i])
            if square is None or square in detections or keypoints[i, 2] < 0.25:
                continue
            point = keypoints[i, :2].reshape(1, 1, 2)
            raw_point = cv2.perspectiveTransform(point, rect_to_image).ravel()
            if not np.all((raw_point >= 0) & (raw_point < [rgb.shape[1], rgb.shape[0]])):
                continue
            detections[square] = {
                "square": square,
                "xyz_m": xyz[i].tolist(),
                "confidence": float(scores[i]),
                "keypoint_confidence": float(keypoints[i, 2]),
                "image_xy_px": raw_point.tolist(),
            }
            x0, y0, x1, y1 = boxes[i]
            rect = np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], np.float32)
            outline = np.rint(cv2.perspectiveTransform(rect, rect_to_image)[0]).astype(np.int32)
            mark = tuple(np.rint(raw_point).astype(int))
            cv2.polylines(overlay, [outline], True, (72, 226, 252), 1, cv2.LINE_AA)
            cv2.circle(overlay, mark, 3, (72, 226, 252), -1, cv2.LINE_AA)
            cv2.putText(overlay, square, (mark[0] + 4, mark[1] + 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.36, (72, 226, 252), 1, cv2.LINE_AA)

        return {"state": "ready", "detections": sorted(detections.values(), key=lambda d: d["square"])}


class SimPieceLayout:
    """Place simulated pieces and read their support points for validation."""

    def __init__(self, model):
        self.model = model
        self.plane = BoardPlane.from_model(model)
        self.ids = np.array([i for i in range(model.ngeom) if model.geom(i).name.startswith("piece_")])
        self.initial = np.array([self.plane.square_center(model.geom(i).name[-2:]) for i in self.ids])
        self.original_geom_pos = model.geom_pos[self.ids].copy()
        self.square_centers = np.array([self.plane.square_center(f"{file}{rank}")
                                        for file in "abcdefgh" for rank in range(1, 9)])

    def randomize(self, rng, count):
        centers = self.square_centers[rng.choice(64, len(self.ids), replace=False)].copy()
        centers[:, :2] += rng.uniform(-0.004, 0.004, (len(self.ids), 2))
        active = rng.permutation(len(self.ids)) < count
        self.place(centers, active)
        return centers, active

    def place(self, centers, active):
        self.model.geom_pos[self.ids] = self.original_geom_pos + centers - self.initial
        self.model.geom_pos[self.ids[~active], 2] = -2

    def truth(self):
        centers = self.initial + self.model.geom_pos[self.ids] - self.original_geom_pos
        truth = {}
        for xyz in centers:
            square = self.plane.square_name(xyz)
            if square is not None and abs(xyz[2] - self.plane.surface) < 0.001:
                truth[square] = xyz
        return truth


def compare_to_sim(frame, truth):
    detections = frame["detections"]
    position_errors = []
    for detection in detections:
        true_position = truth.get(detection["square"])
        if true_position is None:
            continue
        error = float(np.linalg.norm(np.array(detection["xyz_m"]) - true_position) * 1000)
        detection["error_mm"] = error
        position_errors.append(error)
    matched = len(position_errors)
    return {
        **frame,
        "metrics": {
            "expected": len(truth),
            "matched": matched,
            "missed": len(truth) - matched,
            "false_positives": len(detections) - matched,
            "mean_error_mm": float(np.mean(position_errors)) if matched else None,
            "max_error_mm": max(position_errors) if matched else None,
        } if frame["state"] == "ready" else None,
    }
