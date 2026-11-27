# chess_brain

Chess rules, game-state tracking and Stockfish move selection for Chessmate.
It sits between vision and the arm and is the only component that knows chess:

```
vision ──placement FEN──▶ chess_brain ──pick/place actions──▶ arm
                              ▲                                 │
                              └────────── arm_done ─────────────┘
```

Design notes: `docs/CHESS_BRAIN_PLAN.md` at the repo root.

## Status

- [x] `game.py` — full game state, move inference from a placement FEN, Stockfish wrapper
- [x] `actions.py` — move → arm pick/place actions (captures, castling, en passant; queen promotion behind a flag)
- [x] `machine.py` — turn-taking state machine (pure Python, fully unit-tested)
- [x] `brain_node.py` — ROS 2 Jazzy node (thin shell around the state machine)
- [x] `cli.py` — play against the real detector by pasting FENs: `python -m chess_brain.cli`
- [x] built and hand-driven inside the course container (`docs/evidence/`)
- [x] `sim_world_node.py` — simulated human + camera + arm: plays a whole game with no hardware
- [x] `operator_node.py` — the detect button and operator commands, from a terminal
- [x] `vision_adapter_node.py` / `arm_adapter_node.py` — plug-in points for the real detector and the real arm
- [x] `kinova_executor.py` / `kinova_arm.py` — pick-and-place for the course's Kinova Gen3 Lite through MoveIt (unit-tested with a fake arm; run in Gazebo with blocks standing in for pieces; **not yet on the real arm**)
- [x] `reach_check` — asks MoveIt which squares the arm can reach for a given board position
- [x] the Kinova executor run in Gazebo: single move, capture, missed grasp (`docs/evidence/2026-11-26_gazebo_executor_moves.txt`)
- [x] `grip_direction.py` — the gripper turns to stay clear of the other pieces (run with MoveIt on mock hardware, a whole 48-ply game included, and in Gazebo; never with a piece in the gripper: `docs/evidence/2026-11-26_integration_container_runs.txt`, `2026-11-26_selftest_whole_games_discard_reach.txt`, `2026-11-26_gazebo_grip_direction.txt`)
- [x] occupancy on the robot side: `vision_adapter_node` can publish the occupied squares, `arm_adapter_node` can look at the board before it moves (run against a simulated camera: `docs/evidence/2026-11-26_integration_container_runs.txt`)
- [ ] the Kinova executor on the real arm, and with real pieces
- [ ] a real detector plugged in (the wrapper around the simulation's perception, `chessbot_sim_ros.camera_detector`, has only seen rendered frames)

## Interface contract

All topics are `std_msgs/String`.

| Topic | Direction | Payload |
|---|---|---|
| `/chessbot/detected_fen` | vision → brain | piece-placement FEN, rank 8 → rank 1, a1 bottom-left in the image, e.g. `rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR` |
| `/chessbot/detected_occupancy` | vision → brain | alternative to `detected_fen` for a detector that only knows which squares hold a piece: JSON list of square names, e.g. `["a1", "b1", "e4"]`. Use one of the two topics. See `occupancy.py` for what this cannot tell apart |
| `/chessbot/arm_actions` | brain → arm | `{"id": 7, "move": "e7e5", "actions": [{"pick": "e7", "place": "e5"}], "occupied": ["a1", "a2", …]}`. `occupied` = every square holding a piece before the first action; the arm side may use it to keep the gripper clear of the other pieces, or ignore it |
| `/chessbot/arm_done` | arm → brain | `{"id": 7, "status": "ok"}` or `{"id": 7, "status": "failed", "reason": "grasp"}` |
| `/chessbot/status` | brain → anyone | `{"state", "status", "detail", "fen", "expected_placement", "turn", "check", "result", "move_id", "verified", "candidates"}` (latched) |
| `/chessbot/board_state` | brain → anyone | full 6-field FEN after every applied move (latched) |
| `/chessbot/cmd` | operator → brain | `start` \| `reset` \| `retry` \| `move <uci>` (answers status `ambiguous_move`) |
| `/chessbot/detect_request` | button or brain → vision | "please detect once now". `button` from the operator, or the reason from the brain: `setup` \| `after_robot_move` \| `arm_failed` \| `board_changed` \| `retry` \| `confirm_capture` |

**Vision**
- Publishes the placement only; chess_brain owns turn, rules and move selection.
- Either one message per detection (button press) or a continuous stream works: a placement is
  believed once it has been unchanged for `stable_time` seconds.
- Button mode: subscribe to `/chessbot/detect_request` and publish one detection for every message on
  it, whatever its text. The human presses the button (`operator_node`) after moving; the brain sends
  the request itself when the robot's move has finished, so nobody has to press for the robot.
- A detection taken after `arm_done` lets the brain verify the arm really made the move.
- Publish nothing when the board is occluded.

**Arm**
- Execute `actions` in order. `pick`/`place` are square names (`a1`…`h8`); square → metres is
  `docs/board_setup.md`. Off-board names: `DISCARD` (captured pieces; the arm chooses the slot) and,
  only when promotion is enabled, `SPARE_Q_W` / `SPARE_Q_B`.
- `place` is never an occupied square: a capture always sends `(captured square → DISCARD)` first.
- Reply exactly once per `id`, after the arm is parked out of the camera's view. Any other id is ignored.
- `arm_actions` is not latched: subscribe before the game starts. A missed command shows up as
  `arm_timeout` on `/chessbot/status`.
- The move is applied to the game only after `"status": "ok"`. After `failed` (or a timeout) the brain
  waits for a detection: board back in the pre-move position → it re-sends; board in the post-move
  position (e.g. a human completed the move) → it carries on.

**Humans**
- When castling, move the king first (as the rules of chess require). A rook moved first and left
  standing is a legal rook move and will be read as one.
- `status: illegal_move` means the board shows no legal move; put the pieces right and it continues.

### Status values

`setup` · `waiting_human` · `illegal_move` · `arm_mismatch` · `vision_lost` · `thinking` · `executing` ·
`arm_failed` · `arm_timeout` · `board_changed` · `game_over` · `engine_error` · `unsupported_move`

Only with `detected_occupancy`: `confirming_capture` (a capture is applied after a second frame shows
the same squares) · `ambiguous_move` (several moves look the same from above; `candidates` lists them,
answer with `move <uci>`).

## Parameters

| Name | Default | Meaning |
|---|---|---|
| `robot_color` | `black` | `white` or `black` |
| `stockfish_path` | `stockfish` | also found automatically in `/usr/games` |
| `think_time` | `0.5` | seconds per move |
| `skill_level` | `5` | Stockfish Skill Level, 0–20 |
| `stable_time` | `1.0` | seconds a placement must stay unchanged |
| `arm_timeout` | `120.0` | seconds to wait for `arm_done` |
| `vision_timeout` | `0.0` | `> 0` enables `vision_lost` (continuous vision only) |
| `allow_promotion` | `false` | needs spare queens on the table |
| `start_fen` | start position | full 6-field FEN (a placement-only FEN is rejected) |
| `first_action_id` | `-1` | `-1` seeds ids from the clock; use `0` when driving by hand so the first id is 1 |

## Nodes in this package

| Executable | What it is |
|---|---|
| `brain_node` | the chess logic node (the real thing) |
| `operator_node` | terminal console: Enter = detect button; `start` / `reset` / `retry`; paste a placement FEN to act as vision; prints the brain's status |
| `sim_world_node` | stand-in for the human, the camera and the arm, for testing and demos without hardware |
| `vision_adapter_node` | camera image → your `detect()` function → `/chessbot/detected_fen` |
| `arm_adapter_node` | `/chessbot/arm_actions` → your `pick_and_place()` function → `/chessbot/arm_done` |
| `reach_check` | prints which squares the Kinova arm can reach for a board position; the arm does not move |

## Plugging in the real detector and the real arm

The two adapter nodes already speak chess_brain's topics. Each team writes **one Python function**;
nothing in chess_brain has to change.

### Vision: one function

```python
def detect(image_bgr):
    """image_bgr: numpy array (height, width, 3), uint8, BGR (OpenCV style).
    Return the piece-placement FEN, rank 8 first, a1 bottom-left, e.g.
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR".
    Return None (or raise) if the board cannot be read."""
```

```bash
ros2 run chess_brain vision_adapter_node --ros-args \
    -p detector:=your_package.your_module:detect -p image_topic:=/your/camera/image
```

The node keeps the latest camera frame, runs `detect` once for every message on
`/chessbot/detect_request` (the operator's button, or chess_brain asking after the robot's move) and
publishes the answer. `-p period:=1.0` adds continuous detection. Images may be `bgr8`, `rgb8`,
`bgra8`, `rgba8` or `mono8` (a Gazebo RGB camera bridged to ROS arrives as `rgb8`).

The FEN must be in board coordinates whatever the camera orientation: if the camera looks from
black's side, rotate before returning.

**A detector that cannot tell the pieces apart** returns the occupied squares instead, and the node
is started with `-p output:=occupancy` (it then publishes on `/chessbot/detected_occupancy`):

```python
def detect(image_bgr):
    """Return the squares that hold a piece, e.g. ["a1", "b1", "e4"], or None."""
```

Either kind of detector may take a second parameter, `camera`: the intrinsic matrix `k`, the
distortion `d`, `width`, `height` and `frame_id` from the `sensor_msgs/CameraInfo` topic named by
`-p camera_info_topic:=...`, plus `config`, the JSON object in `-p detector_config:='{...}'`.
`chessbot_sim_ros.camera_detector:detect` is such a detector: the AprilTag + YOLO perception of
`chessbot_sim/`, unchanged (see that package's README for what it needs and what was checked).

A detection whose picture is older than a request that is still waiting is not published; the
request gets the next one.

### Arm: one function

```python
def pick_and_place(pick_xyz, place_xyz, info):
    """Move the piece standing at pick_xyz to place_xyz and return when it is there.
    pick_xyz, place_xyz: (x, y, z) in metres, world frame; z is the board surface.
    info: {"id", "move", "index", "count", "last", "pick", "place"}. When info["last"] is
    True, finish by parking the arm out of the camera's view.
    Return None/True on success; return False or a reason string, or raise, on failure
    (chess_brain.arm_adapter.PlacedButFailed(reason) if the piece arrived all the same)."""
```

```bash
ros2 run chess_brain arm_adapter_node --ros-args \
    -p executor:=your_package.your_module:pick_and_place \
    -p board_x:=0.30 -p board_y:=-0.20 -p board_yaw_deg:=0.0
```

The node converts square names to coordinates (`docs/board_setup.md`: 50 mm squares, origin at the a1
corner), hands out discard slots for captured pieces (a column beside the a-file, `discard_x` /
`discard_y` in the board frame), calls your function once per action, and answers on
`/chessbot/arm_done`. "World" is whatever frame `board_x`, `board_y`, `board_yaw_deg` and `surface_z`
were measured in, normally the arm's planning frame. The z it passes is the board surface; the grasp
height above it is yours to add.

`pick_and_place` runs inside `arm_adapter_node`'s process, in a worker thread (not a ROS callback), so
it may block for as long as the motion takes. If it uses ROS itself:

1. never call `rclpy.init()` or `rclpy.shutdown()` (the adapter already did);
2. create your own node once and your own executor, and wait only with that executor, never with
   `rclpy.spin_until_future_complete(node, ...)` or `rclpy.spin_once(node)`;
3. use the `_async` calls, never the blocking `client.call()` or `action_client.send_goal()`;
4. always wait with a timeout, and return a reason string when it expires.

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor

_node = _exec = _client = None

def _ros():
    global _node, _exec, _client
    if _node is None:                                    # first call
        _node = rclpy.create_node("chess_arm_motion", use_global_arguments=False)
        _exec = SingleThreadedExecutor()
        _exec.add_node(_node)
        _client = ActionClient(_node, YourAction, "your_action_name")

def _wait(future, seconds):
    _exec.spin_until_future_complete(future, timeout_sec=seconds)   # OUR executor
    if not future.done():
        raise TimeoutError(f"no answer within {seconds:g} s")
    return future.result()

def pick_and_place(pick_xyz, place_xyz, info):
    _ros()
    if not _client.wait_for_server(timeout_sec=5.0):
        return "action server not available"
    handle = _wait(_client.send_goal_async(goal), 10.0)  # build `goal` from pick_xyz / place_xyz
    if not handle.accepted:
        return "goal rejected"
    _wait(handle.get_result_async(), 60.0)
    return None                                           # or a reason string on failure
```

Without `executor` the adapter runs dry: it logs every pick and place with its coordinates, waits,
and reports success. That is enough to drive chess_brain today:

```bash
ros2 launch chess_brain robot.launch.py          # brain + vision adapter + arm adapter (dry run)
ros2 run chess_brain operator_node               # second shell: paste placement FENs as the "camera"
```

As the "camera", wait for `[brain] wait_human` before pasting the next position, and read the robot's
move from `[brain] executing - <move>` to know what the board looks like afterwards.

`your_package` must be importable by the node's interpreter (`/usr/bin/python3`, not the
`/opt/ros_python` virtualenv), together with everything it imports (OpenCV, etc.). Build it too
(`colcon build --symlink-install --packages-select chess_brain your_package`), re-source
`install/setup.bash` in every shell, and check with `/usr/bin/python3 -c "import your_package.your_module"`.
A single file works as well:
`PYTHONPATH=/dir/of/file:$PYTHONPATH ros2 launch chess_brain robot.launch.py detector:=my_file:detect`.

Do not run `arm_adapter_node` and `sim_world_node` together: both answer `arm_actions`. If you restart
chess_brain, restart `arm_adapter_node` too (it ignores command ids it has already seen).

### With a vision that only reports occupied squares: look before moving

From the occupied squares alone, the board after a capture looks the same whether or not the
captured piece was really taken away: the square is occupied either way. So with
`-p check_squares:=occupancy` the arm adapter asks the camera itself (`/chessbot/detect_request`,
then the next `/chessbot/detected_occupancy`):

- before the first action of a command: the pick square must hold a piece, the place square must be empty;
- again before an action that puts a piece where an earlier action of the same command took one
  away (the second half of a capture). The arm parks first, so that it is not in the picture.

If the camera disagrees, or gives no picture within `check_timeout` (15 s), the action does not start
and the command fails with the reason; chess_brain then waits for the operator as after any arm
failure. `robot.launch.py` turns the check on by itself when `observation:=occupancy` and a detector
is set (`check_squares:=off` to overrule). The parked arm must not hide the board from the camera:
at the WORK pose the gripper hangs over the ranks nearest the arm in the board placements used so
far, so an overhead camera needs `park:=retract` or another parking pose.

## Running on the course's Kinova Gen3 Lite

`chess_brain.kinova_executor:pick_and_place` is a ready executor for the arm used in Lab 6 and the
midterm (Gen3 Lite, MoveIt through `pymoveit2`). It moves a piece with the Lab 6 block-stacking
sequence: above the square → open → straight down → close → straight up → straight across at the
carrying height → straight down → open → straight up, then parks at the WORK pose after the last
action of a move. It runs in Gazebo and on the real arm with the same command, like the labs.

**Status: unit tests, and in Gazebo on 2026-11-26 it moved the Lab 6 blocks as stand-in pieces (a
single move, a two-action capture, a missed grasp reported as a failure; see
`docs/evidence/2026-11-26_gazebo_executor_moves.txt`). Never run on the real arm or with chess
pieces. On the real arm only with a TA present, one operator, a hand on the emergency stop.**
Ctrl-C does not stop a motion the arm has already been sent.

In Gazebo, start the container from `kinova-jazzy-20260924`: in `kinova-jazzy-latest` (the build
of 2026-10-15) the simulated gripper does not move
(`docs/evidence/2026-11-26_gazebo_gripper_image_regression.txt`).

1. Start the arm the way Lab 6 does (Gazebo station or the real arm, then MoveIt), each in its own
   container shell.
2. Decide where the board is. The frame is the arm's `base_link`: +x forward from the arm, +y to
   its left, z = 0 at the bottom of the arm's base, which is 0.0627 m above the tabletop. With the
   board flat on the table, `surface_z` = -0.0627 + the board's thickness (≈ **-0.056** for the
   6.56 mm board of `docs/board_setup.md`). `board_x`, `board_y` are the a1 corner; `board_yaw_deg`
   turns the board about z. The examples below use this layout, with the robot playing black:

   ```
   board_x:=0.60 board_y:=-0.20 board_yaw_deg:=90     (rank 8 towards the arm, rank 1 towards the human)
   squares: x 0.20 … 0.60, y -0.20 … 0.20;  captured pieces: a column on the table at y = -0.275
   ```

   `offboard_drop:=0.0066` says the captured pieces and spare queens stand on the table, one board
   thickness below the squares.
3. Ask MoveIt which squares are reachable there. Nothing moves:

   ```bash
   ros2 run chess_brain reach_check --ros-args -p use_sim_time:=true \
       -p board_x:=0.60 -p board_y:=-0.20 -p board_yaw_deg:=90 -p surface_z:=-0.056 -p offboard_drop:=0.0066
   ```

   `#` = reachable for carrying and for gripping, `^` = only at the carrying height, `.` = not at
   all. Move the board until all 64 squares are `#` (if that is impossible, the board is too big
   for this arm and the team has to know now). The check takes several minutes and prints its
   progress.

   Close to the arm's base a `#` can be too optimistic: the check asks for joint angles at the
   carrying and at the gripping height, not whether the arm can travel straight down between
   them. In Gazebo the arm could not go down onto squares 0.12 and 0.14 m from its base axis that
   the map showed as `#`; at 0.17 m and beyond it could
   (`docs/evidence/2026-11-27_gazebo_reach_by_moving.txt`).

   Below the map it shows the places for captured pieces in the order the arm fills them (8 to a
   column, further columns away from the board) and says how many pieces fit before the first
   place that is out of reach. A game can capture 30 pieces, so it checks 30 places
   (`-p discard_slots:=N` for another number). If fewer fit, move the first place with
   `discard_x` / `discard_y` (board frame) and run the check again: the far corner of that area
   is the first thing to leave the arm's reach, so it usually has to sit towards the arm.

   Result for the 400 mm board (50 mm squares) as close as the mount plate allows, the command
   above with `board_x:=0.572`: **54 of 64**; the far rank and two more squares are out of reach
   (`docs/evidence/2026-11-25_gazebo_reach_check_50mm.txt`). With the gripper straight down this
   arm covers about 0.15 to 0.52 m from its base axis. Pushing that board towards the arm, onto
   the mount plate, brings the far rank into reach, but then the arm cannot go down onto the
   middle squares of the near rank: no position gives all 64
   (`docs/evidence/2026-11-27_gazebo_reach_by_moving.txt`).

   Result for a 320 mm board (40 mm squares) in the same place, `square_size:=0.04 board_x:=0.492
   board_y:=-0.16` with the captured pieces from `discard_x:=-0.05 discard_y:=0.14`: **64 of 64**
   and room for all 30 captured pieces (`docs/evidence/2026-11-26_gazebo_reach_check_40mm.txt`).
   The places for captured pieces then run past rank 8 beside the arm's base.
4. Play, with you as the camera until a detector is plugged in:

   ```bash
   ros2 launch chess_brain robot.launch.py use_sim_time:=true \
       executor:=chess_brain.kinova_executor:pick_and_place \
       board_x:=0.60 board_y:=-0.20 board_yaw_deg:=90 surface_z:=-0.056 offboard_drop:=0.0066 \
       arm_timeout:=300 grasp_check:=false
   ros2 run chess_brain operator_node           # second shell: paste placement FENs
   ```

   **The arm moves to WORK as soon as the launch starts.** `grasp_check:=false` is for Gazebo
   without pieces (the fingers close on nothing); leave it on with real pieces. Drop `use_sim_time`
   on the real arm. A move takes roughly a minute at these speeds, hence `arm_timeout:=300`.
   In Gazebo a move with an object in the gripper took 4 to 5 minutes of wall time (the simulation
   slows down), so a capture there needs `arm_timeout:=900`.

| Parameter | Default | Meaning |
|---|---|---|
| `speed` / `slow_speed` | `0.2` / `0.1` | velocity and acceleration scaling of the moves at carrying height and of every return to the parking pose / of the descents and lifts. Above `max_speed` is refused |
| `max_speed` | `0` | upper limit for both. `0` = the default limit: 0.2 on the real arm (what the midterm required of every motion there), 1.0 when `use_sim_time` is true. A value from 0.01 to 1.0 is the limit wherever the arm runs, e.g. `max_speed:=0.6 speed:=0.6 slow_speed:=0.3`; the arm node warns when it is above 0.2 outside simulation. The first move after start-up (to WORK, from wherever the arm was left) stays at 0.2 at most. With MoveIt on mock hardware one move (a1 to h1 on 40 mm squares) took about 53 s at 0.2 / 0.1 and about 32 s at 0.6 / 0.3; nothing above 0.2 has run on the real arm |
| `grip_height` | `0.02` | fingertips above the square when closing on a piece (m) |
| `travel_clearance` | `0.15` | fingertips above the board while carrying (m); must clear the tallest piece plus the carried one |
| `place_clearance` | `0.003` | a piece is released this much higher than it was gripped (m) |
| `open_position` | `0.5` | finger joint when open over the board; 0 = fully open, wider than a square allows. **How wide 0.5 is has not been measured on the real gripper.** A straight line through the Lab 6 readings (0.80 closed, about 0.44 on a 50 mm block in Gazebo) gives about 42 mm; the manufacturer's finger meshes give about 60 mm, which touches the neighbouring pieces. Set it from a measured width |
| `grasp_check` | `true` | check every finger command on the measured finger position: opened before descending, closed on *something*, still holding before the release, opened at the release |
| `missed_reading` | `0.75` | a finger reading at or above this after closing means "no piece" |
| `park` | `work` | where the arm waits between moves: `work` or `retract` (it must not hide the board from the camera) |
| `add_table` | `true` | add the lab table and mount to MoveIt's planning scene |
| `floor_z` | `-0.0627` | the tabletop; a `surface_z` that would send the fingertips below it is refused |
| `gripper_length` | `0.157` | fingertips below `end_effector_link` (m), measured in Lab 6 with the fingers fully open |

### Which way the gripper opens

Seen from above, each finger is a block about 20 mm thick and 24 mm wide. Opening along the board's
grid, the fingers reach towards the two nearest neighbours and pass a few millimetres from them;
turned 45°, they reach towards the diagonal neighbours, which are 1.41 squares away. A 3D study with
the manufacturer's finger meshes and the piece models (pieces on all eight neighbouring squares) gave
about 4 to 6 mm along the grid and 8 to 9 mm on the diagonal, on 50 mm and on 40 mm squares alike.
Those figures are for the opening that suits each case best, not for `open_position`'s default, whose
width has not been measured. The plan-view model used here is cruder and more pessimistic along the
grid: with a 40 mm opening on 50 mm squares it gives -2.5 mm (touching) along the grid and about 11 mm
on the diagonal. Both say the same thing about which direction is better.

So the direction is chosen for every action. chess_brain sends the occupied squares with each command;
`arm_adapter_node` picks, out of four directions (along a rank, along a file, the two diagonals), the one
that keeps the open fingers farthest from the other pieces at the pick and at the place
(`grip_direction.py`), and the executor turns the gripper before it descends. Rules of the choice:

- more than 20 mm of room counts as enough: the gripper stays along the grid unless that is tight;
- the piece is turned by a quarter turn or not at all while it is carried, so a square stem keeps its
  faces along the grid;
- filled discard slots count as pieces in the way.

The model is a plan view (two rectangles, discs for the pieces) and only ranks directions; it is not a
collision check. On the diagonal a square stem is held by two corners: round stems would be better.

Two tool yaws, half a turn apart, open the fingers along the same line. Whether a pose has joint
angles at all does not depend on the direction, as long as either yaw may be used (the last joint
turns about the vertical), so `reach_check` needs no change. Whether the arm can get there in a
straight line from where it is does depend on it: the last joint turns 148° either way and no further.
For every move `kinova_arm` therefore ranks the two yaws from the measured joint angles: first the
one the arm can turn to without changing its configuration, and of two such the one that does not end
within 20° of that limit; if the first cannot be planned and nothing has moved, the second is tried.
The yaw the gripper already has is kept all the same, even next to the limit: the other one is then
half a turn away, and MoveIt's straight line does not reliably turn the right way round (on mock
hardware it stopped after 11 % of such a line). A descent or a lift keeps the yaw the gripper has: it
never turns among the pieces.

| Parameter of `arm_adapter_node` | Default | Meaning |
|---|---|---|
| `grip_direction` | `auto` | `auto` = chosen per action as above; `fixed` = always the Lab 6 orientation (fingers opening along the arm's y axis) |
| `grip_gap` | `0.040` | free width between the open fingers that the model assumes (m); set it to the measured width at `open_position` |
| `grip_piece_radius` | `0.0125` | radius of another piece at finger height that the model assumes (m); scale it with the pieces |

A hand-written command without `occupied` gets no direction and the Lab 6 orientation.

What the sequence guarantees, whatever fails (each failure is reported on `arm_done` with its reason):

| Failure | What the arm does |
|---|---|
| cannot reach or descend onto the piece; gripper does not open or close; fingers close on nothing | lifts straight up, parks: "no piece was moved" |
| cannot reach or descend onto the target while carrying | puts the piece back on its square, parks |
| cannot put it back, cannot lift it, or cannot open over the target | stops, **still holding the piece**, and refuses every further move until the gripper is empty: take the piece out, put it back on its square, type `retry` in `operator_node` |
| the piece slips out on the way | parks and says so |
| the piece arrived but the arm cannot lift away or park | says so; a discard slot it filled is not reused |
| a lift fails | the arm stays where it is (never sideways from among the pieces) and lifts first on the next move |

Between squares the arm moves in a straight line at the carrying height; only if MoveIt cannot plan
that line (or when leaving the parking pose) may it choose its own path.

To be measured on the real arm before a game: `open_position` (the open fingers must fit between
neighbouring pieces), `grip_height`, `gripper_length` with the fingers at `open_position`, and the
finger reading with each piece type in the gripper (it must lie between `open_position` and
`missed_reading`).

### Without Gazebo: MoveIt on mock hardware

`ros2 launch chess_brain mock_gen3_lite.launch.py` starts the Gen3 Lite with ros2_control mock
hardware and MoveIt: the joints follow their commands exactly, there is no physics and nothing to
grip (a finger always ends where it was told to go, so use `grasp_check:=false`). `reach_check` and
the executor run against it as against Gazebo, without `use_sim_time`. `docker/jazzy-moveit/Dockerfile`
builds an image for it on Apple Silicon, where the course image does not run:

```bash
docker build -f docker/jazzy-sim/Dockerfile -t chessbot-jazzy-sim ros2_ws/src
docker build -t chessbot-jazzy-moveit docker/jazzy-moveit
docker run --rm -it -v "$PWD":/chessbot chessbot-jazzy-moveit
# inside: colcon build ..., source install/setup.bash, then in three shells:
#   ros2 launch chess_brain mock_gen3_lite.launch.py
#   ros2 run chess_brain reach_check --ros-args -p board_x:=0.572 -p board_y:=-0.20 -p board_yaw_deg:=90 -p surface_z:=-0.056
#   ros2 launch chess_brain robot.launch.py executor:=chess_brain.kinova_executor:pick_and_place grasp_check:=false board_x:=...
```

That image has pymoveit2 5.1 and the released kortex packages; the course image builds both from
source (pymoveit2 4.0). The reach map it gives for the 400 mm board is the same as in Gazebo.

## Run the tests (any machine, no ROS)

```bash
cd ros2_ws/src/chess_brain
python3 -m venv .venv && source .venv/bin/activate
touch .venv/COLCON_IGNORE          # so colcon never crawls the venv
pip install -r requirements-dev.txt
pytest -q
```

One test needs a real Stockfish (`brew install stockfish` / `apt-get install stockfish`) and is skipped without it.

## Build and run in the course container

Start the container as in the repo README (the repo is mounted at `/chessbot`), then:

```bash
source /opt/ros/jazzy/setup.bash
bash /chessbot/ros2_ws/src/chess_brain/scripts/setup_deps.sh   # stockfish (apt) + python-chess (pip, for /usr/bin/python3); once per container
cd /chessbot/ros2_ws
colcon build --symlink-install --packages-select chess_brain
source install/setup.bash
ros2 run chess_brain brain_node --ros-args -p robot_color:=black -p first_action_id:=0
```

`build/` and `install/` stay on the bind mount, so a new container only needs `setup_deps.sh` and the two `source` lines.
They are created as root; to hand them back to your user, run inside the container:
`chown -R "$(stat -c %u:%g /chessbot)" /chessbot/ros2_ws/{build,install,log}`.

### Watch a whole game with no hardware

`sim_world_node` plays the human (a weak Stockfish), the camera and the arm on the real topics, and checks
every arm command against the rules the real arm relies on. One command, no Gazebo needed:

```bash
ros2 launch chess_brain sim_game.launch.py                     # full game, ends by itself
ros2 launch chess_brain sim_game.launch.py max_plies:=10       # short demo
ros2 launch chess_brain sim_game.launch.py arm_fail_every:=4   # every 4th arm command fails: shows recovery
ros2 launch chess_brain sim_game.launch.py robot_color:=white human:=random
```

At the end it prints the moves, the result, and whether the brain and the simulated referee agree.
After pulling new code, rebuild first (`colcon build --symlink-install --packages-select chess_brain`).

### Be the vision yourself

With `brain_node` running, in a second shell:

```bash
source /opt/ros/jazzy/setup.bash && source /chessbot/ros2_ws/install/setup.bash
ros2 run chess_brain operator_node
```

Paste a placement FEN (for example from the detector's web demo) and press Enter to publish it as a
detection; plain Enter presses the detect button; `start`, `reset`, `retry` are sent as commands.

### Drive one turn by hand

Open a second shell in the same container (from the VM host):

```bash
docker exec -it "$(docker ps -q --filter ancestor=ghcr.io/mems-intro-to-robotics/mems-robotics-toolkit:kinova-jazzy-latest | head -n1)" bash
source /opt/ros/jazzy/setup.bash && source /chessbot/ros2_ws/install/setup.bash
bash /chessbot/ros2_ws/src/chess_brain/scripts/demo_turn.sh
```

Every new shell in the container needs those two `source` lines before `ros2 run` / `ros2 launch`.

`demo_turn.sh` publishes the start position, then the position after 1.e4, then `arm_done` for id 1.
In the node's terminal you should see:

```
status: setup → wait_human → (human played e2e4) → thinking → arm_actions: {'id': 1, 'move': ...} → executing → wait_human
```

The same by hand:

```bash
ros2 topic pub --once /chessbot/detected_fen std_msgs/msg/String "{data: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'}"
ros2 topic pub --once /chessbot/detected_fen std_msgs/msg/String "{data: 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR'}"
ros2 topic pub --once /chessbot/arm_done std_msgs/msg/String '{data: "{\"id\": 1, \"status\": \"ok\"}"}'
ros2 topic echo --once --full-length /chessbot/status std_msgs/msg/String --qos-reliability reliable --qos-durability transient_local
```

If a `--once` message does not register, send it a few times (`--times 3 -r 2`); repeats are harmless.
