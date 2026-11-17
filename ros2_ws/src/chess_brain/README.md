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
- [ ] a real detector and a real pick-and-place plugged in

## Interface contract

All topics are `std_msgs/String`.

| Topic | Direction | Payload |
|---|---|---|
| `/chessbot/detected_fen` | vision → brain | piece-placement FEN, rank 8 → rank 1, a1 bottom-left in the image, e.g. `rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR` |
| `/chessbot/arm_actions` | brain → arm | `{"id": 7, "move": "e7e5", "actions": [{"pick": "e7", "place": "e5"}]}` |
| `/chessbot/arm_done` | arm → brain | `{"id": 7, "status": "ok"}` or `{"id": 7, "status": "failed", "reason": "grasp"}` |
| `/chessbot/status` | brain → anyone | `{"state", "status", "detail", "fen", "expected_placement", "turn", "check", "result", "move_id"}` (latched) |
| `/chessbot/board_state` | brain → anyone | full 6-field FEN after every applied move (latched) |
| `/chessbot/cmd` | operator → brain | `start` \| `reset` \| `retry` |
| `/chessbot/detect_request` | button or brain → vision | "please detect once now". `button` from the operator, or the reason from the brain: `setup` \| `after_robot_move` \| `arm_failed` \| `board_changed` \| `retry` |

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

### Arm: one function

```python
def pick_and_place(pick_xyz, place_xyz, info):
    """Move the piece standing at pick_xyz to place_xyz and return when it is there.
    pick_xyz, place_xyz: (x, y, z) in metres, world frame; z is the board surface.
    info: {"id", "move", "index", "count", "last", "pick", "place"}. When info["last"] is
    True, finish by parking the arm out of the camera's view.
    Return None/True on success; return False or a reason string, or raise, on failure."""
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
