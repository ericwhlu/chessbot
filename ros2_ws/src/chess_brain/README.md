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
- [ ] built and hand-driven inside the course container (commands below)
- [ ] connected to the real detector / the real arm

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

**Vision**
- Publishes the placement only; chess_brain owns turn, rules and move selection.
- Either one message per detection (button press) or a continuous stream works: a placement is
  believed once it has been unchanged for `stable_time` seconds.
- Detect once after the human moves and once after the robot's move has finished. A detection taken
  after `arm_done` lets the brain verify the arm really made the move.
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
bash /chessbot/ros2_ws/src/chess_brain/scripts/setup_deps.sh   # stockfish (apt) + python-chess (pip); once per container
cd /chessbot/ros2_ws
colcon build --symlink-install --packages-select chess_brain
source install/setup.bash
ros2 run chess_brain brain_node --ros-args -p robot_color:=black -p first_action_id:=0
```

`build/` and `install/` stay on the bind mount, so a new container only needs `setup_deps.sh` and the two `source` lines.
They are created as root; to hand them back to your user, run inside the container:
`chown -R "$(stat -c %u:%g /chessbot)" /chessbot/ros2_ws/{build,install,log}`.

### Drive one turn by hand

Open a second shell in the same container (from the VM host):

```bash
docker exec -it "$(docker ps -q --filter ancestor=ghcr.io/mems-intro-to-robotics/mems-robotics-toolkit:kinova-jazzy-latest | head -n1)" bash
source /opt/ros/jazzy/setup.bash
bash /chessbot/ros2_ws/src/chess_brain/scripts/demo_turn.sh
```

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
