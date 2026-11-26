# MuJoCo simulation

From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r chessbot_sim/requirements.txt
.venv/bin/python -m chessbot_sim.live_demo
```

The desktop viewer shows the movable RGB camera, AprilTag and YOLO detections, and live pose errors against MuJoCo ground truth, including the arm base relative to the board. Left-drag to orbit, right-drag to look around, and scroll to dolly. Use **WASD** to fly, **Q/E** to change height, and **Z/X** to roll. **Tab** switches between piece positions and calibration, **R** resets the view, and **P** prints the current estimates and errors.

Record six pawn moves from the perception camera with per-frame detections. Recording requires FFmpeg with libx264.

```sh
.venv/bin/python -m chessbot_sim.pick_place_demo --output chessbot-perception-camera.mp4
```

The demo rescans the board before each move. It plans from the RGB estimates and moves the arm and gripper through MuJoCo actuators. Jaw contact and friction carry each piece. Contact forces, the settled position, and an RGB rescan are used to check each move.

## Python API

```python
from chessbot_sim.piece_pose import BoardPlane, PieceDetector
from chessbot_sim.simulation import Camera, Simulation

with Simulation.from_files() as sim:
    detector = PieceDetector(BoardPlane.from_model(sim.model))
    rgb, _ = sim.render_camera(Camera.initial(sim.view_center))
    frame = sim.vision.infer(rgb, detector)
    for piece in frame.pieces["detections"]:
        print(piece["square"], piece["xyz_m"], piece.get("xyz_arm_m"))
```

`ChessVision.infer` accepts an undistorted RGB frame with known camera intrinsics and the tag geometry in [tags.json](tags.json) and [arm.json](arm.json). The target transforms map the board and arm base into the OpenCV camera frame. When both targets are detected, `base_to_board` maps the arm base into the board frame. Board coordinates are in meters, from the a1 corner, with +X toward h, +Y toward rank 8, and +Z up. Piece coordinates come from projecting the YOLO base-center keypoint onto the playing surface. They describe resting pieces. Occlusion can interrupt detection.

## Checks

```sh
.venv/bin/python -m chessbot_sim.live_demo --verify
.venv/bin/python -m chessbot_sim.live_demo --verify-pieces
.venv/bin/python -m chessbot_sim.pick_place_demo --check
```

## Assets

- The piece meshes come from the repository's [STEP designs](../models/chessbot/meshes). To regenerate them, install `cadquery-ocp-novtk==7.9.3.0` and run `python -m chessbot_sim.export_pieces`.
- The Kinova Gen3 and Robotiq 2F-85 models come from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/bf04290ac7911fa7a39339c7e507792fd464c438). See the [Kinova](assets/kinova_gen3/LICENSE) and [Robotiq](assets/robotiq_2f85/LICENSE) licenses.
- The one-class checkpoint is fine-tuned from Ultralytics YOLO11n-pose on rendered Chessbot scenes. Run `python -m chessbot_sim.train_pieces --train-count 480 --val-count 72 --epochs 24` to rebuild it. See [Ultralytics licensing](https://www.ultralytics.com/license) before redistributing the checkpoint. The checkpoint has been evaluated in simulation.
