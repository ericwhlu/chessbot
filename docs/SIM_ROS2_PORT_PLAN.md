# Plan: the MuJoCo game on ROS 2

Status (2026-11-25, night): **built and verified on the Mac (steps 1 to 4), then run in the course
image on the VM (`docs/evidence/2026-11-25_vm_ros2_mujoco_game_12plies.txt`); on the branch
`feature/sim-game`, not merged.** Section 8 says what was built, what differs from the plan, and what was
checked how. The plan itself was reviewed twice before the work and the code twice after.

Goal: the game that `python -m chessbot_sim.play_game` plays in one process runs as ROS 2 nodes on
the existing `chessbot/*` topics: `brain_node` decides, a bridge node owns Hao's MuJoCo
`Simulation` and plays the camera, the arm and the opponent. Nothing of Hao's is modified.

Out of scope: the real arm (Gen3 Lite / MoveIt), `feature/adapters`, colour from vision, video.

## 1. Where it runs

The target is the course container on the VM, used the way the midterm was: the repository
under `/root/workspaces/`, `colcon build --symlink-install --packages-select ...`,
`source install/setup.bash`, then `ros2 run` / `ros2 launch`. David runs it there and pastes
the output.

It is developed in a mirror of that on the Mac: a native arm64 ROS 2 Jazzy container
(`ros:jazzy-ros-base`, Ubuntu 24.04, `/usr/bin/python3` 3.12), where each step can be run and
checked directly. The nodes use only `rclpy` and `std_msgs`. The probe (§6) showed that ROS 2,
MuJoCo rendering, the perception and a whole short game work in the mirror; the course image
(amd64) is not measured yet.

## 2. Part A: chess_brain accepts occupancy

Hao's perception reports occupied squares, not piece types. Today the brain only understands a
placement FEN on `chessbot/detected_fen`.

### 2.1 The change

- New topic `chessbot/detected_occupancy` (`std_msgs/String`, JSON list of square names).
  `detected_fen` keeps working. There is one observation slot: a message on either topic
  replaces the last one, so a setup uses one of the two topics, not both.
- `machine.py`: the stored detection becomes an *observation*, a placement or an occupancy set.
  New input `on_occupancy(text, now)`; anything that is not a list of square names is ignored
  and logged. The six places that compare placements today (setup, the frame from just before
  `arm_done`, the human's move, the check while thinking, RECOVER "after" and "before") go
  through two helpers, `_shows(observation, board)` and `_infer(observation)`. The machine keeps
  the board after the pending robot move (today only its FEN text) for the RECOVER comparison.
- `brain_node.py`: one more subscription. No new parameter.

### 2.2 Where occupancy is weaker, and what the machine does about it

The review demonstrated three cases on a prototype. The first two would send the arm onto a
wrong board, so they are part of the design, not later hardening.

1. **A missing piece looks like a capture.** After 1.e4 d5, the e4 pawn in the opponent's hand
   (or missed by the detector in one frame) gives exactly the squares of exd5; the brain would
   apply it and the arm would discard the robot's own pawn. Reproduced; in random games about
   15 % of single missing pieces read as a unique capture. So a capture inferred from occupancy
   is not applied at once: the machine publishes a `detect_request` (`confirm_capture`) and
   applies the move only when a frame received after that request, unchanged for `stable_time`,
   shows the same squares. This is two frames instead of one, not a proof: vision should scan
   with hands out of view, and the real fix remains colour from vision (question for Hao).
2. **A frame from just before `arm_done`.** For a placement the machine trusts such a frame if
   it shows the finished robot move plus at most one legal human move, because an arm in view
   never produces exactly that. With occupancy an arm hiding one piece does. For occupancy only
   "no change" is accepted there.
3. **A robot capture cannot be verified from the final frame alone.** Captured piece never
   removed and capturing piece lost gives the same squares as success. The check belongs to
   the executor: before the second action the target square must scan empty.
   `Match.pick_and_place` already does this; it is a stated requirement for any arm node used
   with occupancy.

Limits that stay, written down and pinned by tests: SETUP accepts any 32 pieces on the four
home ranks (colours or king and queen swapped go unnoticed; the operator checks the set-up).

### 2.3 Ambiguity and the `move` command

- The review enumerated 119,400 positions: the only moves occupancy cannot tell apart are one
  piece capturing on two or more squares, and what a pawn promotes to (3 to 5 % of played moves).
  `occupancy.py` assumes a queen today; a human promotion becomes ambiguous with four candidates
  instead, so an under-promotion cannot silently desynchronise the game.
- In WAIT_HUMAN an ambiguous observation gives status `ambiguous_move` with the candidates in
  `detail`, and the machine waits.
- New operator command `move <uci>` on `chessbot/cmd`: accepted only in WAIT_HUMAN and only if
  it is one of the candidates of the current stable observation. A refusal and its reason go
  into `detail` (not only the log); malformed text never raises. `operator_node` / `console.py`
  learn the command.

### 2.4 Tests

`test_machine.py` on occupancy: setup; a quiet human move; a capture needs the confirming
frame, and a different second frame cancels it; the missing-piece case of 2.2.1 sends no arm
command; the pre-`arm_done` frame of 2.2.2; robot move confirmed; RECOVER before / after, and
half-done capture and castling give no re-send; ambiguous + `move` accepted, not a candidate,
malformed, wrong state; human promotion; messages with duplicates or wrong types; alternating
topics; the whole-game test of the suite on occupancy with an operator. The existing 284 stay
green.

## 3. Part B: the bridge

Pure core first, ROS shell second (the split of `machine.py` / `brain_node.py`). The review
showed the difficult part is the order of messages, so that order lives in a class that pytest
can drive against the real `BrainMachine`.

New ROS 2 package `ros2_ws/src/chessbot_sim_ros` (ament_python). Hao's files and
`play_game.py` are not modified.

### 3.1 `session.py`: who does what, when (no ROS, no MuJoCo)

`WorldSession(world, config)` with inputs `on_status`, `on_arm_actions`, `on_detect_request` and
`tick(now)`, which returns the messages to publish. It keeps its own `chess.Board` (what the
opponent knows). One job per tick, in this order:

1. **Arm command.** A repeated id is ignored; a move that is not legal on its own board is
   answered `failed`. Otherwise execute it. On success: push the move, publish `arm_done ok`.
   On failure: the simulated operator finishes the move by hand **first**, the move is pushed,
   and only then `arm_done failed` + reason is published. Every later scan therefore shows the
   finished move and the brain leaves RECOVER through "board shows the post-move position". (If
   a pre-move scan could slip out before the operator acts, the brain would re-send the command
   and the arm would run the move twice.)
2. **Pending `detect_request`**: scan, publish `chessbot/detected_occupancy`.
3. **Opponent's move**, only when the brain's latest status is `wait_human` with the same FEN
   as the session's board, no detect request is pending, and `human_delay` (1.5 s, longer than
   the brain's `stable_time`) has passed since the last published scan. This lets the brain
   verify the robot's move before the opponent moves. Then scan and publish.
4. **Heartbeat** (scan mode `continuous`, the default): when idle, scan and publish every
   2 s. The brain asks for the first scan exactly once, on a volatile topic, while this node
   is still loading YOLO; without the heartbeat both sides wait forever. It also covers any
   lost message.

Both ways of telling the brain "look now" are kept (David, 2026-11-25); parameter `scan_mode`:

- `continuous` (default): as above. The brain decides the opponent has moved when the squares
  have been unchanged for `stable_time` and fit exactly one legal move.
- `on_request`: no heartbeat. The node scans only when asked on `chessbot/detect_request`, by
  the brain or by a button (`operator_node` already publishes `button` there; there is no
  physical button yet), plus once at start-up and once after the simulated opponent's move,
  which stands for the opponent pressing the button.

Operator and result:

- While the status is `ambiguous_move` the operator sends `move <uci>` (the move the opponent
  played), again every 2 s until the status changes: a command that arrives before the
  observation is stable is refused.
- A ply counts as done when the brain's FEN (from its status) equals the session's FEN. The
  summary records per ply: done by arm / finished by operator, inferred / ambiguous.
- A ply that is not done within `ply_timeout`, or brain state `error`, ends the game: summary
  written, non-zero exit. The game also ends after `plies` half-moves or at game over.

pytest (pure, with a fake world): a short whole game; first detect request lost; arm failure
(exactly one `arm_actions` is sent, the game continues); ambiguous capture answered with
`move`; duplicate `arm_actions`; the opponent never moves before the robot's move is verified.
The 12-ply run reaches neither the failure nor the ambiguity path, so these tests are what
covers them.

### 3.2 `MujocoWorld`: the world the session drives (MuJoCo, no ROS)

A thin adapter over the existing `Match` of `play_game.py` (constructed without an engine):
`scan()`, `execute(actions)` (`Match.pick_and_place` per action, then the confirming scan),
`play_human(board, move)`, `finish_by_hand(...)`, `snapshot(...)`. MuJoCo ground truth is used
only by the opponent's hand, the operator and the evaluation columns, as in `play_game.py`.

### 3.3 `mujoco_world_node`: the ROS shell

- Subscribes `chessbot/arm_actions`, `chessbot/detect_request`, `chessbot/status`; publishes
  `chessbot/detected_occupancy`, `chessbot/arm_done`, `chessbot/cmd`.
- One thread: callbacks only store, a timer calls `session.tick`. The node is busy while the
  arm moves (the brain sends nothing during that time and accepts a late `ok`). MuJoCo's
  renderer stays on the thread that created it and is closed in `finally`.
- Parameter `repo_root` (default `/chessbot`): the node puts it on `sys.path` itself before
  importing `chessbot_sim`, so `ros2 run` works as well as the launch file. Sets
  `MUJOCO_GL=egl` if it is unset (without it MuJoCo tries to open a window).
- Writes `summary.json` and one image per ply to `out_dir`.

## 4. Part C: launch, container, evidence

- `chessbot_sim_ros/launch/mujoco_game.launch.py`: `brain_node` + `mujoco_world_node`; arguments
  `plies`, `robot_color`, `seed`, `script`, `out_dir`, `repo_root`; `arm_timeout:=300`; ends when
  the game does. `allow_promotion` stays false (there are no spare queens in the scene).
- `chessbot_sim_ros/scripts/setup_sim_deps.sh`: what a fresh course container needs, installed
  for `/usr/bin/python3` (the interpreter `ros2 run` uses there): mujoco, ultralytics with
  CPU-only torch, opencv-headless, scipy, with numpy held at Ubuntu's 1.26 (what the ROS
  bindings use); apt `libegl1` and friends. Same role as chess_brain's `setup_deps.sh`.
- `docker/jazzy-sim/Dockerfile`: the Mac mirror, built from `ros:jazzy-ros-base` with the same script.
- A README section with the commands for the VM, terminal by terminal, in the style of the
  midterm's install notes.
- `use_sim_time` stays off: MuJoCo publishes no `/clock`, and `brain_node` already uses the
  wall clock for its timers.
- Evidence in `docs/evidence/`: pytest log, the launch log of a 12-ply game, `summary.json`.

## 5. Order of work and checks

1. Part A + tests (pytest on the Mac).
2. `session.py` + tests (pytest on the Mac).
3. `MujocoWorld`; a run without ROS on the Mac (session + `BrainMachine` + MuJoCo in one
   process, 12 plies), so the message order is checked against the real simulation first.
4. Node, launch, Dockerfile; in the container: all pytest, then the 12-ply game over ROS 2.
5. Evidence; David decides on branch / commit / PR.
6. David runs `setup_sim_deps.sh`, the pytest suites and the 12-ply launch in the course
   container on the VM and pastes the output; what differs there is fixed before the PR.

Code is reviewed independently after steps 2 and 4.

Left out on purpose: a refactor of `play_game.py`; a ROS-level failure-injection run (pytest
covers that path); games longer than 12 plies (a later check, not part of this port).

## 6. Container probe (2026-11-25, Mac M1 Pro, Docker Desktop, linux/arm64)

Image: `ros:jazzy-ros-base` + apt (`python3-numpy`, `stockfish`, `libegl1`, `libgl1`, ...) + pip
for `/usr/bin/python3` (mujoco 3.17, ultralytics 8.4, opencv-headless, scipy, chess).

| Check | Result |
|---|---|
| `rclpy` publish/subscribe of `std_msgs/String` with all pip packages imported | ok (numpy 1.26.4) |
| Offscreen render + perception, `MUJOCO_GL=egl` | 32 of 32 pieces; 0.6 to 0.75 s per scan (first 3.3 s) |
| Same with `MUJOCO_GL=osmesa` | no result: the process ends without output; not investigated |
| Hao's `pick_place_demo --check` | 6 of 6 trials pass, 41 s |
| `play_game --plies 6` | 3 of 3 robot moves by the arm incl. one capture, 3 of 3 human moves inferred, 34 s |

Found on the way: the default pip torch on arm64 is a CUDA build (3.3 GB of unused nvidia
libraries; image 7.56 GB), hence the CPU-only torch above. Not measured: the course image on
the VM (amd64); whether EGL works there is the first thing to check.

## 7. How this leads to the demo on the real arm

The demo runs on ROS 2 with the midterm's stack: the Kinova Gen3 Lite through MoveIt
(`pymoveit2`, the `ArmNode` pattern of Lab 6 and the midterm), in the course container. The
bridge of this plan speaks exactly the topics the real nodes will speak, so going to the real
arm replaces the bridge by a vision node and an arm node and changes nothing in `chess_brain`:

| Topic | In this plan (MuJoCo) | On the real arm |
|---|---|---|
| `chessbot/detected_occupancy` | `mujoco_world_node` (Hao's perception on the rendered camera) | a vision node: Hao's perception on the real camera |
| `chessbot/arm_actions` -> `chessbot/arm_done` | `mujoco_world_node` (Hao's IK on MuJoCo actuators) | an arm node on MoveIt: `arm_adapter_node` + `kinova_executor` (parked on `feature/adapters`, simulated arm only so far) |
| the opponent, the operator | simulated inside `mujoco_world_node` | people; `operator_node` for `move <uci>` and `retry` |

What carries over unchanged: `chess_brain` including Part A, the launch and container setup,
the message order rules of §3.1 (they describe what any camera + arm pair must do). What does
not: Hao's arm motion code. The MoveIt side is not part of this plan.

## 8. As built (2026-11-25)

| Step | State |
|---|---|
| 1. chess_brain accepts occupancy | done: `machine.py`, `occupancy.py`, `brain_node.py`, `console.py`; `test_machine_occupancy.py` |
| 2. `WorldSession` | done: `ros2_ws/src/chessbot_sim_ros/chessbot_sim_ros/session.py`, `test/test_session.py` |
| 3. `MujocoWorld`, game without ROS | done: `chessbot_sim/world.py`, `chessbot_sim_ros/dry_run.py` |
| 4. node, launch, set-up script, Dockerfile | done: `mujoco_world_node.py`, `launch/mujoco_game.launch.py`, `scripts/setup_sim_deps.sh`, `docker/jazzy-sim/Dockerfile` |
| 5. evidence | written to `docs/evidence/2026-11-25_ros2_mujoco_game_*.txt`; branch / commit / PR wait for David |
| 6. the course container on the VM | done 2026-11-25: 12 plies, arm 6/6, opponent 6/6, EGL works there (`docs/evidence/2026-11-25_vm_ros2_mujoco_game_12plies.txt`) |

What differs from sections 2 to 4 (found by the code reviews or while running):

- **The second frame is also needed in RECOVER** (2.2.1). "The robot's capture is done" is,
  from above, the position before it minus the capturing piece, so a frame that misses that
  piece read as "done". Reproduced and fixed.
- **chess_brain's status has two more fields**: `verified` (false from the robot's move until
  a frame taken after it shows the board in step) and `candidates` (the moves `ambiguous_move`
  asks about). The opponent waits for `verified` instead of for a delay (3.1 rule 3), and the
  operator answers only a question that names the move (rule 4).
- **Heartbeat in `on_request` mode**: the node still scans every 2 s while chess_brain is
  waiting for the camera (game not started, a ply open, robot move not verified, RECOVER,
  any status but `waiting_human`). Without it one lost frame ended the game.
- **An arm command waits while the opponent's ply is open**: topics are not ordered among
  themselves, and the command can overtake the status that closes the ply.
- **`idle_timeout`** (60 s): the session ends with an error when nothing happens, instead of
  hanging (a lost arm command, a chess_brain that read a different move).
- `setup_sim_deps.sh` holds setuptools as well as numpy (pip had raised it past what colcon accepts).
- `play_game.py` is unchanged. `dry_run.py` is new: the same game in one process, for a
  machine without ROS 2.

What was checked, and how:

| Check | Where | Result |
|---|---|---|
| pytest: chess_brain / chessbot_sim_ros / `test_gameplay.py` | Mac, and `/usr/bin/python3` in the container | 333 / 51 / 11 passed |
| Each sequencing rule of `session.py` removed in turn (11 mutants) | Mac, scratch copy | every one fails a test |
| `dry_run --plies 12` (black, continuous), after the review fixes | Mac, MuJoCo | arm 6 of 6, opponent 6 of 6 |
| `dry_run` white + `on_request` 12 plies; black 24 plies (before the review fixes) | Mac, MuJoCo | 6 of 6 + 6 of 6; 12 of 12 + 12 of 12 |
| `ros2 launch ... plies:=12` (black, continuous) | container, ROS 2 Jazzy | arm 6 of 6, opponent 6 of 6, exit 0 |
| `ros2 launch ... robot_color:=white scan_mode:=on_request` | container, ROS 2 Jazzy | arm 6 of 6, opponent 6 of 6, exit 0 |

The arm executed, with the camera confirming: quiet moves, captures (including a1, the far
corner, and captures by knight and bishop) and castling (before the review fixes; king then rook).

Not verified: in the course image on the VM only the default game was run (black, continuous,
12 plies); a failing arm and an
ambiguous capture in the real simulation (pytest covers both paths with a fake world); en
passant and promotion executed by the arm; games longer than 24 plies; a second frame that is
wrong in the same way as the first (a piece held, or missed, across both).
