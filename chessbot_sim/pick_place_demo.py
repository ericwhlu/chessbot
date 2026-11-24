"""Record YOLO-guided pick and place in MuJoCo."""
import argparse
from functools import partial
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from .manipulation import CLOSED_GRIP, FPS, OPEN_GRIP, Arm, plan_pick_place
from .piece_pose import BoardPlane, PieceDetector
from .simulation import Camera, Simulation


TRIALS = (("h2", "h4"), ("g2", "g4"), ("f2", "f4"),
          ("h7", "h5"), ("g7", "g5"), ("f7", "f5"))
class Video:
    def __init__(self, path, sim, camera, detector):
        self.sim = sim
        self.camera = camera
        self.detector = detector
        self.trial = 0
        self.source = ""
        self.destination = ""
        self.pick_error = None
        self.placed = []
        self.target = None
        self.jaw_forces = None
        self.tallies = []
        self.process = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.process = subprocess.Popen([
                "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pixel_format", "bgr24",
                "-video_size", "1280x720", "-framerate", str(FPS), "-i", "-", "-an",
                "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(path),
            ], stdin=subprocess.PIPE)

    @staticmethod
    def write_label(image, text, x, y, size=0.55, color=(226, 232, 231), thickness=1):
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)

    def render(self, phase):
        if self.process is None:
            return
        frame = self.sim.capture(self.camera, self.detector)
        piece_count = len(frame.pieces["detections"])
        board_tags, arm_tags = len(frame.board.ids), len(frame.base.ids)
        self.tallies.append((piece_count, board_tags, arm_tags))
        view = frame.bgr
        board_to_camera = frame.board.pose.estimate
        if self.target is not None and board_to_camera is not None:
            p = board_to_camera[:3, :3] @ self.target + board_to_camera[:3, 3]
            if p[2] > 0:
                pixel = tuple(np.rint((self.sim.intrinsics @ p)[:2] / p[2]).astype(int))
                cv2.circle(view, pixel, 16, (245, 205, 108), 2, cv2.LINE_AA)

        image = np.full((720, 1280, 3), (28, 26, 24), np.uint8)
        image[:, :960] = cv2.resize(view, (960, 720), interpolation=cv2.INTER_AREA)
        label = partial(self.write_label, image)
        muted = (172, 183, 184)
        accent = (132, 215, 247)
        label("Chessbot", 980, 40, 0.96, thickness=2)
        label("Perception camera", 980, 69, 0.62, muted)
        cv2.line(image, (980, 92), (1260, 92), (76, 72, 68), 1)

        label("YOLO", 980, 125, 0.61, accent, 2)
        if frame.pieces["state"] == "ready":
            label(f"{piece_count} pieces detected", 980, 153, 0.58)
        else:
            label("Board pose unavailable", 980, 153, 0.55)
        label("AprilTag", 980, 194, 0.61, (185, 224, 149), 2)
        label(f"Board  {board_tags}/4      Arm base  {arm_tags}/4", 980, 221, 0.54)
        if frame.base_to_board.estimate is not None:
            delta, angle = frame.base_to_board.errors()
            label(f"Base vs GT  {np.linalg.norm(delta) * 1000:.2f} mm  {angle:.3f} deg",
                  980, 249, 0.49, muted)
        cv2.line(image, (980, 271), (1260, 271), (76, 72, 68), 1)

        label(f"TRIAL {self.trial} / {len(TRIALS)}", 980, 310, 0.60, accent, 2)
        label(f"{self.source}  >  {self.destination}", 980, 358, 1.0, thickness=2)
        label(phase, 980, 391, 0.60)
        if self.pick_error is not None:
            label(f"Pickup error vs GT  {self.pick_error:.1f} mm", 980, 428, 0.51, muted)
        if self.jaw_forces is not None:
            left, right = self.jaw_forces
            label(f"Jaw contact L/R  {left:.1f} / {right:.1f} N", 980, 454, 0.50, muted)
        cv2.line(image, (980, 478), (1260, 478), (76, 72, 68), 1)
        label("Placement error vs ground truth", 980, 504, 0.50, muted)
        for i, (source, dest, error) in enumerate(self.placed):
            label(f"{source} > {dest}     {error:.1f} mm", 980, 529 + 24 * i, 0.55, (164, 224, 183))
        label("MuJoCo | Contact-only grasp", 980, 701, 0.48, muted)
        self.process.stdin.write(image.tobytes())

    def still(self, phase, seconds):
        for _ in range(round(seconds * FPS)):
            self.render(phase)

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            if self.process.wait():
                raise RuntimeError("Video encoding failed")
            if self.tallies:
                pieces, board, arm = zip(*self.tallies)
                print(f"Processed {len(pieces)} RGB frames through perception. "
                      f"Visible pieces {min(pieces)}–{max(pieces)}, board tags {min(board)}–{max(board)}, "
                      f"arm tags {min(arm)}–{max(arm)}.", flush=True)


def run(sim, detector, output):
    plane = detector.plane
    arm = Arm(sim)
    overhead = Camera.initial(sim.view_center)
    video = Video(output, sim, overhead, detector)
    objects = [sim.data.body(f"object_{file}{rank}") for rank in (1, 2, 7, 8) for file in "abcdefgh"]
    piece = None

    def record(phase):
        if video.process is not None and piece is not None:
            video.jaw_forces = arm.jaw_forces(piece)
        video.render(phase)

    try:
        arm.hold(0.8, OPEN_GRIP, lambda _: None, "")
        for index, (source, dest) in enumerate(TRIALS, 1):
            frame = sim.capture(overhead, detector)
            found = {piece["square"]: piece for piece in frame.pieces["detections"]}
            if source not in found or dest in found or frame.base_to_board.estimate is None:
                raise RuntimeError(f"Could not plan {source} to {dest} from the YOLO scan")
            detection = found[source]
            pickup = np.array(detection["xyz_m"])
            plan = plan_pick_place(arm, pickup, plane.square_center(dest), frame.base_to_board.estimate)
            before = {item.id: item.xpos[:2].copy() for item in objects}
            # Simulation ground truth is used only to evaluate the pickup and landing.
            piece = min(objects, key=lambda item: np.linalg.norm(item.xpos[:2] - pickup[:2]))
            pickup_error = float(np.linalg.norm(piece.xpos - pickup) * 1000)
            video.trial, video.source, video.destination = index, source, dest
            video.target = plane.square_center(dest)
            video.pick_error = pickup_error
            video.jaw_forces = None
            video.still("Scan with YOLO", 0.8)

            arm.play([arm.parked, plan.approach], 1.55, OPEN_GRIP, record, "Move above detection")
            arm.play(plan.descent, 1.05, OPEN_GRIP, record, "Lower gripper")
            arm.hold(0.24, OPEN_GRIP, record, "Settle above piece")
            arm.hold(0.52, CLOSED_GRIP, record, "Close gripper")
            closed_forces = arm.jaw_forces(piece)
            arm.play(plan.descent[::-1], 1.0, CLOSED_GRIP, record, "Lift piece")
            arm.hold(0.12, CLOSED_GRIP, record, "Verify lift")
            lifted_forces = arm.jaw_forces(piece)
            if piece.xpos[2] < 0.11 or min(lifted_forces) < 0.05:
                raise RuntimeError(f"Trial {index}: contact grasp failed; height "
                                   f"{1000*piece.xpos[2]:.1f} mm, jaw forces closed "
                                   f"{closed_forces}, lifted {lifted_forces}")
            arm.play(plan.transfer, 0.90, CLOSED_GRIP, record, "Carry to empty square")
            carried_forces = arm.jaw_forces(piece)
            if piece.xpos[2] < 0.11 or min(carried_forces) < 0.05:
                raise RuntimeError(f"Trial {index}: lost contact during transfer ({carried_forces})")
            arm.play(plan.placement, 1.02, CLOSED_GRIP, record, "Lower onto destination")
            arm.hold(0.36, OPEN_GRIP, record, "Open gripper")
            arm.play(plan.placement[::-1], 0.70, OPEN_GRIP, record, "Retract")
            # Reverse the safe outbound route after lifting the tool clear of the pieces.
            arm.play(plan.transfer[::-1], 0.5, OPEN_GRIP, record, "Retract to scan pose")
            arm.play([plan.approach, arm.parked], 1.50, OPEN_GRIP, record, "Return to scan pose")
            arm.hold(0.25, OPEN_GRIP, record, "Check placement")
            after = sim.capture(overhead, detector)
            confirmed = {p["square"] for p in after.pieces["detections"]}
            error = float(np.linalg.norm(piece.xpos[:2] - plane.square_center(dest)[:2]) * 1000)
            stable = abs(piece.xpos[2] - plane.surface) < 0.003 and piece.xmat[8] > 0.97
            disturbed = max(np.linalg.norm(item.xpos[:2] - before[item.id]) * 1000
                            for item in objects if item.id != piece.id)
            print(f"Trial {index} {source} -> {dest}: pickup {pickup_error:.2f} mm, "
                  f"placement {error:.2f} mm, neighbors {disturbed:.2f} mm, "
                  f"min jaw force at lift/carry {min(*lifted_forces, *carried_forces):.1f} N, "
                  f"YOLO confirmed {dest in confirmed and source not in confirmed}", flush=True)
            if plane.square_name(piece.xpos) != dest or not stable or disturbed > 6:
                raise RuntimeError(f"Trial {index} did not settle cleanly: {piece.xpos}")
            if source in confirmed or dest not in confirmed:
                raise RuntimeError(f"YOLO did not confirm {source} -> {dest}")
            video.placed.append((source, dest, error))
            video.still("Placement verified", 0.82)
        video.still("6 of 6 placements verified", 1.5)
    finally:
        video.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("chessbot-perception-camera.mp4"))
    parser.add_argument("--check", action="store_true", help="validate the six trials without recording")
    args = parser.parse_args()
    with Simulation.from_files(movable_pieces=True) as sim:
        detector = PieceDetector(BoardPlane.from_model(sim.model))
        if args.check:
            run(sim, detector, None)
        else:
            output = args.output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=output.parent) as temp:
                recording = Path(temp) / "recording.mp4"
                run(sim, detector, recording)
                recording.replace(output)
            print(f"Video saved to {output}", flush=True)


if __name__ == "__main__":
    main()
