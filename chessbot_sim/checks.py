"""Deterministic simulation checks for AprilTags and piece localization."""
import numpy as np

from .piece_pose import SimPieceLayout
from .simulation import Camera, look_at
from .vision import rotation_about


def verify(sim):
    # All four occupied ranks should contain every piece in the starting lineup.
    pieces = [sim.model.geom(i).name for i in range(sim.model.ngeom)
              if sim.model.geom(i).name.startswith("piece_")]
    expected = {f"piece_{file}{rank}" for rank in (1, 2, 7, 8) for file in "abcdefgh"}
    if set(pieces) != expected:
        raise RuntimeError("The model does not contain the 32 starting pieces")
    print("32 Chessbot pieces on their starting squares")

    initial = Camera.initial(sim.view_center)
    views = [("Default", initial, (4, 4))]
    for name, offset, roll, counts in [
        ("Top-down", [0, 0, 1.45], 0, (4, 4)),
        ("Oblique with roll", [-0.30, -0.38, 1.32], 0.16, (4, 4)),
        ("Partially obscured arm fixture", [0.12, -0.24, 1.50], -0.35, (4, 3)),
    ]:
        rotation = look_at(offset, up=(0, 1, 0)) @ rotation_about([0, 0, 1], roll)
        views.append((name, Camera(sim.view_center + offset, rotation), counts))
    views.extend([
        ("Board only", Camera(np.array([0.09, 0.20, 0.65]), np.eye(3)), (4, 0)),
        ("Arm only", Camera(np.array([0.82, 0.20, 0.65]), np.eye(3)), (0, 3)),
        ("Looking away", Camera(initial.position, initial.rotation @ rotation_about([0, 1, 0], np.pi)),
         (0, 0)),
        ("Returning to the scene", initial, (4, 4)),
    ])

    for name, camera, expected_counts in views:
        frame = sim.capture(camera)
        counts = len(frame.board.ids), len(frame.base.ids)
        if counts != expected_counts:
            raise RuntimeError(f"{name}: expected tag counts {expected_counts}, got {counts}")
        print(f"{name}: {counts[0]} board tags, {counts[1]} arm tags")
        checks = [
            ("Board in camera", frame.board.pose, counts[0] > 0),
            ("Camera in board", frame.board.pose.inverse(), counts[0] > 0),
            ("Arm base in camera", frame.base.pose, counts[1] > 0),
            ("Arm base in board", frame.base_to_board, all(counts)),
        ]
        for label, pair, expected_pose in checks:
            if (pair.estimate is not None) != expected_pose:
                raise RuntimeError(f"{name}: unexpected availability of {label}")
            if pair.estimate is None:
                continue
            delta, angle = pair.errors()
            distance_mm = np.linalg.norm(delta) * 1000
            print(f"  {label}: {distance_mm:.2f} mm, {angle:.3f} deg")
            if not (distance_mm <= 12 and angle <= 1.4):
                raise RuntimeError(f"{name}: {label} exceeds the pose error limit")
        for observed in (frame.board, frame.base):
            if observed.reprojection_px is not None and observed.reprojection_px > 1.0:
                raise RuntimeError(f"{name}: excessive reprojection error")


def verify_pieces(sim, detector):
    layout = SimPieceLayout(sim.model)
    rng = np.random.default_rng(9241)
    metrics = []
    initial = Camera.initial(sim.view_center)

    def capture(camera):
        return sim.capture(camera, detector, layout).pieces

    for index, (count, offset) in enumerate([
        (32, None), (32, [-0.30, -0.38, 1.32]), (25, [0.12, -0.24, 1.50]),
        (20, [-0.20, -0.33, 1.48]), (30, [0, 0, 1.45]), (0, None),
    ], 1):
        if index > 2:
            layout.randomize(rng, count)
        camera = initial if offset is None else Camera(
            sim.view_center + offset, look_at(offset, up=(0, 1, 0)))
        result = capture(camera)["metrics"]
        if result is None:
            raise RuntimeError(f"View {index}: no AprilTag board pose")
        error = result["mean_error_mm"]
        description = f"{error:.2f} mm mean position error" if error is not None else "no matched pieces"
        print(f"View {index}: {result['matched']}/{count} squares, "
              f"{result['false_positives']} false positives, {description}")
        if (index == 1 and result["matched"] != 32) or (count == 0 and result["false_positives"]):
            raise RuntimeError(f"Incorrect detection count in view {index}")
        metrics.append(result)

    rng = np.random.default_rng(28841)
    for index in range(30):
        count = 32 if index % 4 == 0 else int(rng.integers(6, 32))
        layout.randomize(rng, count)
        azimuth = rng.uniform(-np.pi, np.pi)
        slope = rng.uniform(0.08, 0.72)
        offset = rng.uniform(0.84, 1.65) * np.array(
            [slope * np.cos(azimuth), slope * np.sin(azimuth), 1])
        target = np.array([0.20, 0.20, 0.03]) + rng.uniform(-0.06, 0.06, 3)
        rotation = look_at(offset, up=(0, 1, 0)) @ rotation_about([0, 0, 1], rng.uniform(-0.5, 0.5))
        result = capture(Camera(target + offset, rotation))["metrics"]
        if result is None:
            raise RuntimeError(f"Random view {index + 1}: no AprilTag board pose")
        metrics.append(result)

    away = Camera(initial.position, initial.rotation @ rotation_about([0, 1, 0], np.pi))
    result = capture(away)
    if result["metrics"] is not None or result["detections"]:
        raise RuntimeError("Pieces were reported without a board pose")

    expected = sum(row["expected"] for row in metrics)
    matched = sum(row["matched"] for row in metrics)
    false_positives = sum(row["false_positives"] for row in metrics)
    mean_error = sum(row["matched"] * (row["mean_error_mm"] or 0) for row in metrics) / matched
    print(f"Total across {len(metrics)} views: {matched}/{expected} occupied squares, "
          f"{false_positives} false positives, {mean_error:.2f} mm mean 3D position error")
    if matched / expected < 0.97 or false_positives > 3 or mean_error > 8:
        raise RuntimeError("Piece recognition was below the simulated acceptance threshold")
