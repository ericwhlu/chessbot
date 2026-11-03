# chess_brain — Implementation Plan

Owner: David Zeng · Role: chess rules, Stockfish integration, game-state tracking, robot move selection
Repo: `ericwhlu/chessbot` · Stack: ROS 2 Jazzy + Gazebo Harmonic, Kinova arm, Docker image `mems-robotics-toolkit:kinova-jazzy-latest`
Status: DRAFT v0.1 — 2026-11-02

---

## 1. What this module is

`chess_brain` is the node that sits between vision and the arm. It is the only component that knows the rules of chess.

```
Atharva (vision)                     David (chess_brain)                      Hao / Julian (arm)
detector → piece-placement FEN ──▶  track game state                   ──▶  pick/place action list
                                     infer the human's move                   (square names only)
                                     validate it                         ◀──  "done" / "failed"
                                     ask Stockfish for the reply
                                     translate reply → arm actions
```

What it does **not** do: image processing (vision), square → world coordinates (Eric's board frame), motion planning or grasping (arm team).

## 2. Facts the design depends on

| Fact | Source |
|---|---|
| Vision publishes **piece-placement FEN** (field 1 of FEN only) + 8×8 matrix, e.g. `rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR` | Atharva's demo site |
| Detector is 100 % accurate when camera tilt < 35° | group chat, Oct 31 |
| Turn / castling rights / en-passant are **not** in the detector output; the demo sets "Turn" manually | demo site Edit panel |
| Pieces: geometric base → identical 0.30-unit gripping cube → piece top; square width = 1 unit | demo site |
| Team env: Ubuntu 24.04 VM, Docker `--rm`, repo mounted at `/chessbot` | repo README |
| No ROS 2 on David's Mac | — |
| A-level target: captures, castling, checks, full game unassisted | HW8 |

## 3. Architecture

Two layers so the chess logic can be developed and tested on a Mac without ROS:

```
chess_brain/
├── chess_brain/
│   ├── game.py            # pure Python: GameState, infer_move, engine, move→actions
│   ├── actions.py         # pure Python: ArmAction dataclass + serialisation
│   ├── brain_node.py      # ROS 2 node: topics, state machine, parameters
│   └── fake_vision.py     # dev tool: publishes placement FEN from a PGN or keyboard
│   └── fake_arm.py        # dev tool: echoes "done" after a delay
├── test/
│   ├── test_infer_move.py
│   ├── test_actions.py
│   └── fixtures/*.json    # (before_fen, after_fen, expected_move) cases
├── scripts/setup_deps.sh  # apt install stockfish; pip install chess
├── package.xml / setup.py / setup.cfg
└── README.md              # interface contract (copy of §4)
```

### 3.1 `game.py` — core

```python
class GameState:
    board: chess.Board                 # full state incl. turn, castling, ep
    robot_color: chess.Color

    def infer_human_move(self, placement: str) -> InferResult
    def engine_reply(self) -> chess.Move
    def apply(self, move: chess.Move) -> None
    def status(self) -> dict           # fen, turn, check, game_over, result
```

**infer_human_move** — the key function. Vision gives us *positions*, not *moves*:

```python
def infer_human_move(self, placement):
    if placement == self.board.board_fen():
        return InferResult.NO_CHANGE
    matches = []
    for mv in self.board.legal_moves:
        self.board.push(mv)
        if self.board.board_fen() == placement:
            matches.append(mv)
        self.board.pop()
    if len(matches) == 1: return InferResult.MOVE(matches[0])
    if len(matches) == 0: return InferResult.ILLEGAL_OR_MISREAD
    return InferResult.AMBIGUOUS(matches)   # should not happen with piece types known
```

Because the detector reports piece *types*, promotion is covered automatically (the detected square holds a Q/R/B/N), castling and en passant are covered because `legal_moves` includes them and `push()` moves the rook / removes the pawn.

**engine_reply** — `chess.engine.SimpleEngine.popen_uci(path)`, `Limit(time=think_time)`, Stockfish option `Skill Level` exposed as a parameter so the robot is beatable for the demo.

### 3.2 `actions.py` — move → arm primitives

The arm team receives a flat list; each item is one pick-and-place between named squares. Off-board locations are named too so the arm team maps them once.

```python
@dataclass
class ArmAction:
    pick: str    # "e2", "e4", ... or "SPARE_Q_W" (spare queen, white)
    place: str   # square or "DISCARD"
```

| Move type | Action list |
|---|---|
| Normal | `[(from, to)]` |
| Capture | `[(to, DISCARD), (from, to)]` |
| Castling | `[(e1, g1), (h1, f1)]` (king first) |
| En passant | `[(captured_pawn_sq, DISCARD), (from, to)]` |
| Promotion | `[(from, DISCARD), (SPARE_x, to)]` — promotion with capture adds `(to, DISCARD)` first |

Serialised as JSON on a `std_msgs/String` for now (no custom msg → no extra build step for teammates). Can be upgraded to a custom `.msg` once the interface is frozen.

### 3.3 `brain_node.py` — ROS 2 node and state machine

```
            ┌──────────────┐  placement differs,   ┌──────────┐
   start ──▶│ WAIT_HUMAN   │──── stable N frames ──▶│ VALIDATE │
            └──────────────┘                        └────┬─────┘
                  ▲        illegal/misread: publish status, stay   │ 1 match
                  │                                                 ▼
            ┌─────┴────────┐        arm_done          ┌──────────┐
            │ EXECUTING    │◀──── publish actions ────│  THINK   │
            └──────────────┘                          └──────────┘
                  │ game over after either side's move
                  ▼
            ┌──────────────┐
            │  GAME_OVER   │
            └──────────────┘
```

- Detection is **ignored while EXECUTING** (the arm occludes the board and pieces are mid-air).
- After `arm_done`, the node expects the next detection to match its own board; a mismatch means the arm misplaced a piece → publish `status: arm_mismatch` and stop (manual fix for now).
- If `robot_color == WHITE`, the node starts in THINK.

**Topics** (all `std_msgs/String`, namespace `/chessmate`):

| Topic | Dir | Payload |
|---|---|---|
| `/chessmate/detected_fen` | sub | piece-placement FEN from vision |
| `/chessmate/arm_done` | sub | `"ok"` or `"failed:<reason>"` |
| `/chessmate/arm_actions` | pub | JSON `{"move":"e7e5","actions":[{"pick":"e7","place":"e5"}]}` |
| `/chessmate/robot_move` | pub | UCI string (for logging / UI) |
| `/chessmate/board_state` | pub | full FEN after every applied move |
| `/chessmate/status` | pub | `waiting_human`, `thinking`, `executing`, `illegal_move`, `check`, `game_over:<result>`, `arm_mismatch` |

**Parameters:** `stockfish_path`, `think_time` (s), `skill_level` (0–20), `robot_color`, `stable_frames` (N), `start_fen` (for testing mid-game positions).

## 4. Interface contract (to agree with the team this week)

1. Vision publishes placement FEN on `/chessmate/detected_fen` continuously (or on change). FEN rank order: rank 8 → rank 1 (matches the demo site).
2. Arm subscribes to `/chessmate/arm_actions`, executes actions **in order**, publishes `/chessmate/arm_done` once per message.
3. Square naming is standard algebraic (`a1`…`h8`). Eric's board frame maps square → (x, y). `DISCARD` and `SPARE_*` are named locations the arm team defines.
4. White is on ranks 1–2 as seen from the camera, same as the demo site.

## 5. Development environment

- **Mac (no ROS):** `brew install stockfish && pip install chess pytest`. Everything in `game.py` / `actions.py` and all tests run here.
- **Docker (ROS 2 Jazzy):** the container is started with `--rm`, so dependencies vanish on exit. `scripts/setup_deps.sh` installs `stockfish` + `python3-chess` and is run at container start until the team bakes a custom image. (Open question for the team.)
- Branch `feature/chess-brain`, PR into `main`. Never push to `main` directly.

## 6. Testing

| Level | What | How |
|---|---|---|
| Unit | `infer_human_move` on ~20 (before, after, move) fixtures: normal, capture, castling both sides, en passant, promotion (all 4 pieces), promotion-with-capture, no-change, illegal, misread | `pytest`, Mac |
| Unit | `move_to_actions` for every move type | `pytest`, Mac |
| Unit | Engine reply is legal; skill level changes strength | `pytest`, Mac |
| Integration (no vision, no arm) | `fake_vision.py` replays a PGN as a sequence of placement FENs; `fake_arm.py` replies `ok` after 1 s; node must play through the whole game | Docker |
| Integration (real vision) | Atharva's detector in Gazebo → node | Docker, Obj 3 |
| Integration (real arm) | node → arm pipeline | Docker, Obj 4 |

A quick manual test from day one: make a move on Atharva's site, copy the detected FEN, paste it into a CLI that wraps `GameState`.

## 7. Schedule

| When | Deliverable | Done when |
|---|---|---|
| Mon Nov 2 | `game.py`, `actions.py`, CLI, fixtures | `pytest` green on Mac; CLI plays a game against me |
| Tue Nov 3 | Team meeting: agree §4; ask Atharva for detector code / topic name | contract written into README |
| Wed Nov 4 | `brain_node.py`, `fake_vision.py`, `fake_arm.py`, `setup_deps.sh` | full game plays through in Docker with fakes |
| Thu Nov 5 | PR `feature/chess-brain`; HW9 progress section | PR open, reviewed by one teammate |
| Nov 9–13 (Obj 3) | Connect to real detector in Gazebo; handle misreads, debounce tuning | human move detected end-to-end 10/10 times |
| Nov 16–20 (Obj 4) | Connect to arm; multi-turn test; special moves on real arm | 5 consecutive turns without intervention |

## 8. Risks

| Risk | Mitigation |
|---|---|
| Interface drift (topic names, FEN orientation) | Contract in README; fake nodes let each side test alone |
| Vision fires while the arm is over the board | Gate on state machine; require N stable frames |
| Arm misplaces a piece → brain and board disagree | Post-move verification against next detection; `arm_mismatch` status |
| Human makes an illegal move / bumps a piece | `illegal_move` status; the human fixes the board; node keeps waiting |
| Stockfish missing in `--rm` container | `setup_deps.sh`; propose custom image |
| Promotion needs spare pieces on the table | Spare queens only (parameter); other promotions flagged `unsupported` |
| Hao/Julian's arm pipeline slips | Fakes keep brain testable; demo can fall back to "brain + vision + highlighted move" |

## 9. Open questions for the team

1. Does the detector publish on every frame or only on change? What rate?
2. Will Atharva's detector run as a ROS 2 node in Gazebo, or does someone need to wrap it?
3. Who owns the Gazebo camera pose? (Must keep tilt < 35°.)
4. Where do captured pieces go, and are spare queens on the table?
5. Does the robot play white or black in the demo?
