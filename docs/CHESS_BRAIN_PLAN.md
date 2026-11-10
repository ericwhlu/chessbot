# chess_brain — Implementation Plan

Owner: David Zeng · Role: chess rules, Stockfish integration, game-state tracking, robot move selection
Repo: `ericwhlu/chessbot` · Stack: ROS 2 Jazzy + Gazebo Harmonic, Kinova arm, Docker image `mems-robotics-toolkit:kinova-jazzy-latest`
Status: **v0.3 — 2026-11-09**. v0.2 was revised after three plan reviews (§11); v0.3 records what changed while implementing the node (§12).

---

## 1. What this module is

`chess_brain` sits between vision and the arm. It is the only component that knows the rules of chess.

```
Atharva (vision)                     David (chess_brain)                      Hao / Julian (arm)
detector → piece-placement FEN ──▶  track full game state                ──▶  pick/place action list
                                     infer the human's move                   (square names only)
                                     validate it                         ◀──  done {ok | failed}
                                     ask Stockfish for the reply
                                     translate reply → arm actions
```

**Ownership line for the team contract:** *vision publishes placement only; chess_brain owns turn, rules, game state and move selection; the arm owns motion.* Atharva's "Suggest Best Move" button is a stateless placement → move helper (turn set by hand); it stays as a UI feature and doubles as a test oracle for chess_brain (same FEN → same or equally legal reply).

What this module does **not** do: image processing, square → world coordinates (Eric's board frame), motion planning or grasping.

## 2. Facts the design depends on

| Fact | Source |
|---|---|
| Vision publishes **piece-placement FEN** (field 1 of FEN only), e.g. `rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR` | Atharva's demo site |
| Detector 100 % accurate when camera tilt < 35° | group chat, Oct 31 |
| Turn / castling rights / en-passant square are **not** in the detector output | demo site Edit panel sets "Turn" by hand |
| Pieces: geometric base → identical 0.30-unit gripping cube → piece top; square = 1 unit | demo site |
| Team env: Ubuntu 24.04 VM, Docker `--rm`, repo bind-mounted at `/chessbot` | repo README |
| No ROS 2 on David's Mac | — |
| A-level: captures, castling, checks, full game unassisted. B-level: end-to-end sim with simplified board | HW8 |

## 3. Architecture

Two layers so the chess logic is developed and tested on the Mac without ROS:

```
ros2_ws/src/chess_brain/
├── chess_brain/
│   ├── __init__.py
│   ├── game.py            # pure Python: GameState, infer_human_move, engine_reply
│   ├── actions.py         # pure Python: move_to_actions(board, move) -> [ArmAction]
│   ├── brain_node.py      # ROS 2 node: timer-driven state machine, topics, params
│   ├── cli.py             # dev tool: paste a placement FEN, see move + actions
│   └── fake_vision.py     # dev tool (Obj 3): replay a PGN as placement FENs
├── test/
│   ├── test_infer_move.py
│   ├── test_actions.py
│   ├── test_property.py   # random games: every ply must be inferred exactly
│   └── fixtures/*.json
├── resource/chess_brain   # ament index marker (required, or `ros2 run` can't find the package)
├── launch/brain.launch.py
├── scripts/setup_deps.sh
├── package.xml / setup.py / setup.cfg
└── README.md              # interface contract (= §4)
```

Delete the generated `test_copyright.py` / `test_flake8.py` / `test_pep257.py` (copyright test fails on headerless files). Add `build/ install/ log/` to `.gitignore`.

### 3.1 `game.py` — core

```python
@dataclass
class InferResult:
    kind: Literal["no_change", "move", "illegal_or_misread", "ambiguous"]
    moves: list[chess.Move] = field(default_factory=list)

class GameState:
    board: chess.Board            # full state: turn, castling, ep, clocks
    robot_color: chess.Color

    def __init__(self, start_fen: str, robot_color): ...
        # start_fen MUST be a 6-field FEN. Reject anything else: a placement-only
        # string is accepted by python-chess but silently drops castling rights,
        # so every later human castling move would be "illegal".

    def infer_human_move(self, placement: str) -> InferResult
    def engine_reply(self) -> chess.Move | None
    def apply(self, move: chess.Move) -> None        # push + bookkeeping
    def outcome(self): return self.board.outcome(claim_draw=True)
    def status(self) -> dict
```

**infer_human_move** — vision gives *positions*, not *moves*, so enumerate legal moves and match the resulting placement:

```python
def infer_human_move(self, placement):
    try:
        placement = chess.BaseBoard(placement.strip()).board_fen()   # canonicalise; raises on garbage
    except ValueError:
        return InferResult("illegal_or_misread")
    if placement == self.board.board_fen():
        return InferResult("no_change")
    matches = []
    for mv in list(self.board.legal_moves):      # list(): don't push/pop under a live generator
        self.board.push(mv)
        if self.board.board_fen() == placement:
            matches.append(mv)
        self.board.pop()
    if len(matches) == 1: return InferResult("move", matches)
    if not matches:       return InferResult("illegal_or_misread")
    return InferResult("ambiguous", matches)     # provably unreachable for one human move; kept as a detector-bug trap
```

Why this covers the hard cases for free: `legal_moves` includes castling (as the king's two-square move) and en passant; `push()` moves the rook / removes the captured pawn; the detector reports piece *types*, so promotion is distinguished by what sits on the 8th rank. A human under-promotion is inferred literally (`e7e8n`).

**engine_reply** — `SimpleEngine.popen_uci(path)` opened once at startup, `engine.configure({"Skill Level": n})` once, `engine.quit()` on shutdown.
- Never call the engine unless `board.turn == robot_color` and `outcome() is None`.
- `engine.play(...).move` can be `None` (`bestmove (none)`); treat as game over.
- If the engine under-promotes (`move.promotion not in (None, chess.QUEEN)`), substitute the queen promotion: it is always legal when the under-promotion is, and the table only has spare queens.

**Game-over policy** — `is_game_over()` defaults to `claim_draw=False`, which ignores threefold repetition and the 50-move rule; a low-skill Stockfish shuffling against a beginner *will* hit threefold. Use `outcome(claim_draw=True)` everywhere. Check it **after the human's move, before THINK** (a human mate must not call the engine) and again after the robot's move is applied.

### 3.2 `actions.py` — move → arm primitives

```python
@dataclass
class ArmAction:
    pick: str    # "e2", … or "SPARE_Q_W" / "SPARE_Q_B"
    place: str   # square or "DISCARD"

def move_to_actions(board: chess.Board, move: chess.Move) -> list[ArmAction]
```

**`move_to_actions` takes the PRE-move board.** `is_castling`, `is_en_passant`, `is_capture`, `board.turn` and `piece_at(to)` are all computed from the position before the move; after `push()` they are all wrong. The node computes actions, publishes them, and only `apply()`s after the arm reports `ok` (§3.3).

Check **en passant before capture**: `is_capture()` is also True for en passant, and the captured pawn is not on `to`.

| Move type | Action list (mover's back rank) | Week-1 scope |
|---|---|---|
| Normal | `[(from, to)]` | ✔ |
| Capture | `[(to, DISCARD), (from, to)]` | ✔ |
| Castling K-side | `[(e, g), (h, f)]` — king first | ✔ |
| Castling Q-side | `[(e, c), (a, d)]` — e.g. black: `[(e8,c8),(a8,d8)]` | ✔ |
| En passant | `[(captured_pawn_sq, DISCARD), (from, to)]`, captured_pawn_sq = file(to) + rank(from). Black example after `1.a3 d5 2.a4 d4 3.e4`: `d4e3` → `[(e4,DISCARD),(d4,e3)]` | ✔ (3 lines) |
| Promotion | `[(from, DISCARD), (SPARE_Q_<W|B>, to)]`; with capture, `(to, DISCARD)` first. Spare name from `move.promotion` + pre-move `board.turn` | Obj 4 — returns `unsupported` until spare queens exist on the table |

Invariant for the arm team: **`place` is never an occupied square** — the brain always emits the `(x, DISCARD)` first. `DISCARD` is one named location; the arm allocates slots.

Serialised as JSON on `std_msgs/String` (no custom msg package for teammates to build). The payload is already in action-server *shape* (goal / result with an id) so it can be upgraded to a ROS 2 action for Obj 4 if the arm team wants cancel/feedback.

### 3.3 `brain_node.py` — state machine

```
                 stable detection == start placement
                 AND arm subscriber present (or cmd:start)
   ┌────────┐ ─────────────────────────────────────────▶ ┌────────────┐
   │ SETUP  │                                            │ WAIT_HUMAN │◀────────────────────┐
   └────────┘                                            └─────┬──────┘                     │
        ▲                        placement ≠ board, stable for stable_time                   │
        │ cmd:reset                                            │                             │
        │                     ┌──────────────┐    1 match      ▼          no match, stable:  │
        │                     │   VALIDATE   │────────────▶ ┌───────┐    status illegal_move,│
        │                     └──────────────┘  apply(human) │ THINK │    stay in WAIT_HUMAN │
        │                                       game over?   └───┬───┘                       │
        │                                                        │ engine future done        │
        │                                                        ▼ publish arm_actions{id}   │
        │                     ┌──────────────┐  done{id,failed} ┌───────────┐  done{id,ok}   │
        │                     │   RECOVER    │◀─────────────────│ EXECUTING │──── apply(robot)┘
        │                     └──────┬───────┘   or arm_timeout └───────────┘   game over?
        │   stable detection == pre-move OR post-move placement → resume there
        │
   ┌────┴─────┐
   │GAME_OVER │  (from VALIDATE or after apply(robot))
   └──────────┘
```

Rules the reviewers made me write down:

1. **Timer-driven, not callback-driven.** Subscriptions only store `last_placement`, `last_change_time`, `last_arm_done`. A `create_timer(0.1, tick)` runs the state machine. Stockfish runs in a `ThreadPoolExecutor(max_workers=1)`; `tick` polls `future.done()`. Otherwise a 1 s engine search inside a callback starves every other callback and silently drops queued messages.
2. **Stability is measured in time, not frames.** "Placement unchanged for `stable_time` s since the last change" — where "unchanged" includes "no new message". Frame counting never fires if the detector publishes on change only.
3. **`apply()` the robot's move only on `done{ok}`.** Applying before publishing leaves the internal board one move ahead of reality whenever the arm fails.
4. **Arm failure → RECOVER.** Publish `status: arm_failed:<reason>`; wait for a stable detection equal to either the pre-move or the post-move placement and resume from there. Anything else stays in RECOVER (human fixes the board by hand, guided by `expected_fen` in the status).
5. **Post-move verification is folded into WAIT_HUMAN.** After `ok`, require a detection *received after* `arm_done`, stable; then `infer_human_move`: `no_change` = verified; `move` = verified and the human already moved; `illegal_or_misread` persisting → `status: arm_mismatch` (advisory, keep waiting).
6. **Detection during THINK / EXECUTING is ignored** (but `last_change_time` keeps updating so rule 2 works afterwards). A human who changes the position during THINK gets a reply to the old position; rule 5 then flags it.
7. **`illegal_move` is advisory and debounced.** A lifted piece or a half-done castling is a transient placement that matches no legal move; publish `illegal_move` only after it has been stable for `stable_time`, and never leave WAIT_HUMAN because of it. (Checked: no transient state of castling / en passant / capture / promotion matches a *different* legal move, so there is no false acceptance.)
8. **Correlation id.** `arm_actions.id` increments; any `arm_done` with a different id is logged and ignored (kills duplicate-done). `arm_timeout` (default 120 s) → RECOVER.
9. **QoS.** `arm_actions`, `status`, `board_state` published latched (`depth=1, TRANSIENT_LOCAL`) so a late-starting arm or UI gets the current value. `detected_fen` subscribed BEST_EFFORT so it matches a sensor-data publisher; a RELIABLE subscriber against a BEST_EFFORT publisher never connects and shows no error.
10. **Every incoming string is validated** (`BaseBoard(...)` in try/except) and dropped with a WARN; never crash in a callback.
11. **Persistence.** On every `apply()` write full FEN + `robot_color` to `/chessbot/.chess_brain_state.json` (the bind mount survives `--rm`); `resume:=true` reloads it.
12. **Default `robot_color = BLACK`** — human moves first, no THINK-on-boot special case, and the camera stays on the human's side.

**Topics** (all `std_msgs/String`, relative names so launch can remap; default namespace `/chessbot`):

| Topic | Dir | Payload |
|---|---|---|
| `detected_fen` | sub | placement FEN |
| `arm_done` | sub | `{"id": 7, "status": "ok"}` or `{"id": 7, "status": "failed", "failed_action_index": 1, "reason": "grasp"}` |
| `cmd` | sub | `start`, `reset`, `pause`, `resume` |
| `arm_actions` | pub (latched) | `{"id": 7, "move": "e7e5", "actions": [{"pick": "e7", "place": "e5"}]}` |
| `status` | pub (latched) | `setup`, `waiting_human`, `thinking`, `executing`, `illegal_move`, `arm_failed:<r>`, `arm_mismatch`, `arm_timeout`, `vision_lost`, `check`, `game_over:<result>` + `expected_fen` |
| `board_state` | pub (latched) | full 6-field FEN after every `apply()` |

**Parameters:** `stockfish_path` (default `/chessbot/bin/stockfish`), `think_time` (0.5 s), `skill_level` (5), `robot_color` (`black`), `stable_time` (1.5 s), `arm_timeout` (120 s), `vision_timeout` (5 s), `start_fen` (6-field, default standard), `resume` (false).

## 4. Interface contract — post in the group chat **today**, "this is what I publish/subscribe unless someone objects by Tuesday"

1. **Ownership:** vision publishes placement only; chess_brain owns turn / rules / move selection; arm owns motion.
2. **Vision → `chessbot/detected_fen`** (`std_msgs/String`): piece-placement FEN, rank 8 → rank 1, **a1 is bottom-left in the camera image** (white on ranks 1–2). Publish at a fixed rate ≥ 2 Hz even when unchanged (doubles as a heartbeat). When the board is occluded or confidence is low, publish nothing. QoS: sensor-data (BEST_EFFORT) is fine.
3. **chess_brain → `chessbot/arm_actions`**: JSON as above. Actions execute **in order**; `place` is never occupied; `DISCARD` is one named off-board location (arm allocates slots); `SPARE_Q_W` / `SPARE_Q_B` appear only from Obj 4.
4. **Arm → `chessbot/arm_done`**: exactly one message per `arm_actions.id`, sent **after the arm is parked out of the camera's view**. Schema above.
5. Square names are standard algebraic. Eric's board frame maps square → (x, y); chess_brain never emits coordinates.
6. Robot plays **black** by default.

## 5. Development environment

- **Mac (no ROS):** `brew install stockfish`, `python3 -m venv .venv && pip install chess pytest`. All of `game.py`, `actions.py`, `cli.py` and the tests run here.
- **Docker (ROS 2 Jazzy):** the container is `--rm`, so dependencies vanish on exit. Options ranked by friction for teammates:
  1. **Vendor into the bind mount (do this now):** `pip install --target /chessbot/.pydeps chess` once and a static Stockfish binary at `/chessbot/bin/stockfish` (both gitignored; `scripts/setup_deps.sh` fetches them only if absent). `PYTHONPATH=/chessbot/.pydeps`. No root, no rebuild, survives `--rm`.
  2. **`Dockerfile FROM` the course image + `apt-get install stockfish python3-chess`** — propose to the team for Obj 3; cleanest long-term.
  3. `setup_deps.sh` with `apt-get` at every start — slow, needs network, and plain `pip install` fails on Ubuntu 24.04 (PEP 668) without `--break-system-packages`.
- The README's `docker run` is root; `colcon build` then leaves root-owned `build/ install/ log/` on the host. Add `--user $(id -u):$(id -g)` to the run command if the image allows it, otherwise `chown` after builds.
- Workspace layout: `/chessbot/ros2_ws/src/chess_brain` (the repo is currently flat).
- Branch `feature/chess-brain`, PR into `main`. Never push to `main`.

## 6. Testing

**Fixture format:** `(before_FULL_fen, after_PLACEMENT, expected)` — `before` must be 6-field (see §3.1 start_fen trap).

| Level | Fixtures / checks | Where |
|---|---|---|
| Unit — inference | normal; capture; castling ×4 (W-K, W-Q, B-K, B-Q); en passant by white and by black; "forgot to remove the e.p. pawn" → misread; promotion to Q; human under-promotion → `e7e8n`; promotion-with-capture (black, rank 1); no-change; transient (one piece lifted) → misread; malformed string → no crash; robot-white *and* robot-black variants | Mac, `pytest` |
| Unit — actions | every row of the §3.2 table, both colours; en-passant-before-capture ordering; `place` never occupied | Mac |
| Unit — engine / game-over | engine on a mated board → `None` handled; under-promotion → queen substitution; human mate / stalemate → `game_over` without calling the engine; threefold & 50-move → `game_over` with `claim_draw=True`; `"Skill Level" in engine.options` | Mac |
| Property | ~200 random python-chess games: at every ply `infer_human_move(board_after.board_fen())` returns exactly the played move — mechanically proves no-ambiguity and special-move coverage | Mac |
| Integration, hand-driven | `ros2 topic pub detected_fen` → `ros2 topic echo arm_actions`; `ros2 topic pub arm_done` → next turn | Docker, Wed |
| Integration, replay (Obj 3) | `fake_vision.py` replays a PGN; node plays through | Docker |
| Integration, real vision / real arm | Obj 3 / Obj 4 | Gazebo |

Day-one oracle: make a move on Atharva's site, copy the detected FEN into `cli.py`, compare the reply with his "Suggest Best Move". **Commit one real FEN from his site as a fixture this week — it is the only way to catch a 180°-rotated board before Obj 3.**

## 7. Schedule (resequenced: unknowns first, fakes cut, one artifact per day)

| Day | Do | Artifact for HW9 |
|---|---|---|
| **Mon Nov 2** | **AM (60–90 min): boot VM → start container → `ros2 run demo_nodes_py talker` → put stockfish + python-chess in the bind mount → run `stockfish` once. Log pass/fail per step.** Post §4 in the group chat. **PM:** `game.py` (`infer_human_move`, `engine_reply`), 8 fixtures, `pytest` green on Mac, first push to `feature/chess-brain`. | env checklist; pytest summary; commit hash |
| **Tue Nov 3** | `actions.py` + tests (promotion → `unsupported`). `cli.py`; run it on a FEN copied from Atharva's site. `ros2 pkg create` skeleton, `colcon build` in container, hello-world node publishing `status`. Team meeting 4–5 pm: confirm §4, Q1–Q3, image baking. | CLI transcript; screenshot of node running in container; contract in README |
| **Wed Nov 4** | `brain_node.py`: timer tick, SETUP/WAIT_HUMAN/VALIDATE/THINK/EXECUTING, time-based stability, engine in worker thread, correlation id. Drive by hand with `ros2 topic pub`; the fake arm is `ros2 topic pub --once arm_done '{"id":1,"status":"ok"}'`. | screenshot of `ros2 topic echo arm_actions` showing a move JSON for a published FEN |
| **Thu Nov 5** | Open PR, request one review, README = contract. Write HW9 §3 from the artifacts. Buffer for Docker fallout. If ahead: RECOVER state, `fake_vision.py`. | PR URL; HW9 section |
| Nov 9–13 (Obj 3) | Real detector in Gazebo; tune `stable_time`; RECOVER; `fake_vision.py` PGN replay; persistence | 10/10 human moves detected end-to-end |
| Nov 16–20 (Obj 4) | Real arm; promotion with spare queens; optional action-server upgrade; multi-turn | 5 consecutive turns, no intervention |

Cut from week 1 (reviewers' call): `fake_arm.py`, PGN replay, `board_state` / `robot_move` topics, skill-level tuning, `SPARE_*` in the contract.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Docker/ROS/deps environment untested — the biggest dependency | Monday morning, before any chess code |
| FEN orientation rotated 180° (camera on the robot's side) | Real FEN from Atharva's site as a fixture this week; robot = black by default |
| Interface drift | §4 posted today; silence = agreement; fakes let each side test alone |
| Engine blocks the executor | Worker thread + timer tick (§3.3 rule 1) |
| Arm fails mid-sequence; brain and board disagree | apply-on-ok; RECOVER; correlation id; timeout |
| Vision fires while the arm is over the board | Ignore in THINK/EXECUTING; `arm_done` only after parking out of view; time-based stability |
| Human lifts a piece → spurious `illegal_move` | Debounced, advisory |
| Stockfish missing in `--rm` container | Vendor into bind mount; propose Dockerfile |
| Arm pipeline slips | Demo fallback: vision + brain + highlighted move on Atharva's UI |
| "Atharva already has Stockfish" confusion | Ownership line in §4; his button is the test oracle, not a competitor |

## 9. Open questions

**Block this week — ask today:**
1. Can I `apt-get` / write to `/chessbot` inside the container? Will the team bake stockfish + python3-chess into an image? (Eric / whoever owns the README)
2. Atharva: will the detector run as a ROS 2 node on a `ros_gz_bridge` camera topic in Gazebo, or does someone need to wrap it? Which Stockfish build/depth does your button use, and were you planning to publish a best move from your node? (settle ownership in writing)
3. Robot colour — propose black; confirm Tuesday.

**Can wait:**
4. Detector rate / on-change (design handles both).
5. Camera pose owner (must keep tilt < 35°) — Eric.
6. Where captured pieces go; are spare queens on the table — arm team, Obj 4.
7. **How does the human move a piece in Gazebo?** (`gz service` set_pose teleport, GUI drag, keyboard node?) Determines what `fake_vision.py` must emulate and whether transient states exist at all in sim.

## 10. HW9 §3 skeleton (fill from the artifacts)

- **(a) Objective 1** — *Correction: HW8 listed my teammate's vision objectives by mistake. My actual Objective 1, per our role split:* build the chess-logic node that consumes the detector's placement FEN, tracks full game state, infers and validates the human's move, and selects the robot's reply with Stockfish.
- **(b) Progress / setbacks** — env checklist, pytest summary, CLI transcript, container screenshot. Setbacks: a week late; Docker `--rm` dependency problem; (whatever Monday breaks).
- **(c) Help wanted** — detector as a ROS 2 node; baking deps into the image; FEN orientation confirmation; how a human moves a piece in Gazebo.
- **(d) Learnings** — FEN vs placement; inferring moves from positions; castling/e.p./promotion via python-chess; ROS 2 QoS pitfalls; keeping blocking work out of callbacks.

## 11. Change log v0.1 → v0.2 (from the three reviews)

| # | Change | Raised by |
|---|---|---|
| 1 | `apply()` only after `arm_done{ok}`; `move_to_actions` takes the pre-move board | chess-logic, ROS |
| 2 | Stability in seconds, not frames; `vision_timeout` | ROS |
| 3 | Timer-driven state machine; engine in a worker thread | ROS |
| 4 | RECOVER state for arm failure; correlation id; `arm_timeout` | ROS |
| 5 | SETUP state + `cmd` topic; latched QoS; BEST_EFFORT subscriber | ROS |
| 6 | `start_fen` must be 6-field; en passant checked before capture; castling table for all four cases; engine under-promotion → queen; `engine_reply` Optional; `claim_draw=True`; game-over check before THINK | chess-logic |
| 7 | Placement validated via `BaseBoard`; `list(legal_moves)`; `InferResult` dataclass | chess-logic |
| 8 | Fixture list expanded (castling ×4, e.p. both colours, under-promotion, draws, transient, both robot colours) + property test | chess-logic |
| 9 | Docker environment test moved to Monday morning; `fake_arm.py` and PGN replay cut from week 1; one artifact per day; Wed target = one hand-driven turn | scope |
| 10 | Contract posted today with "silence = agreement"; a1 bottom-left; robot = black; real FEN fixture to catch rotation | scope, ROS |
| 11 | Vendor deps into the bind mount; PEP 668 note; root-owned build dirs; `resource/` marker; launch file; delete lint tests | ROS |
| 12 | Ownership line vs Atharva's Stockfish button; his button = test oracle | scope |

## 12. Change log v0.2 → v0.3 (implementing the node, week 2)

Implemented: `machine.py` (state machine, pure Python), `brain_node.py` (ROS 2 shell), package files, launch file, `scripts/setup_deps.sh`, `scripts/demo_turn.sh`. Where this section disagrees with §3–§5, this section is what the code does.

| # | Change | Why |
|---|---|---|
| 1 | State machine split out of the node into `machine.py`; the node only stores inputs and calls `tick()` | Testable on a Mac without ROS (scenario tests + simulated games with vision and arm in the loop) |
| 2 | Vision may be single-shot (one detection per button press) or continuous; `vision_timeout` defaults to 0 (off) | Team decision: detect once after each move |
| 3 | Draws end the game on the third *actual* repetition / halfmove clock 100, not with `claim_draw=True` | `claim_draw=True` ends the game one ply early (as soon as a repetition is merely possible) |
| 4 | `cmd: retry` never re-sends blind: it waits for a fresh detection showing the pre-move board | A half-executed castling or capture must not be replayed from the start |
| 5 | After an arm *timeout* nothing is re-sent automatically; a late `ok` for the same id is accepted | The arm may only be slow; a second copy would be executed twice |
| 6 | `arm_actions` is published volatile, not latched; `status` and `board_state` stay latched | A restarted arm must not replay an already-executed command |
| 7 | Action ids are seeded per process (`first_action_id`, default: from the clock) and replies queued before a command are dropped | A stale `arm_done` from an earlier run must not confirm a new move |
| 8 | New status `board_changed`: if the board changes while the engine is thinking, the arm is not commanded until the board is restored | Never send the arm onto a position the move was not computed for |
| 9 | A detection taken just before `arm_done` is used if it already shows the finished robot move | Single-shot vision often fires a moment before the arm reports done |
| 10 | Promotions the arm cannot execute are excluded from the engine's search (`searchmoves`) instead of failing afterwards | The robot always has an executable move |
| 11 | `SETUP` starts on a stable detection of the start position or `cmd: start`; no pause/resume, no persistence file yet | Scope: deferred to Obj 3 |
| 12 | Dependencies: Stockfish via apt (`/usr/games/stockfish`, found automatically), python-chess via pip | `python3-chess` is not packaged for Ubuntu 24.04 |
| 13 | Numeric parameters use dynamic typing; the tick timer uses a steady clock | `-p think_time:=1` must not crash the node; `use_sim_time` must not freeze it |
| 14 | Humans must castle king-first (§3.3 rule 7 was wrong for rook-first castling: the rook move alone is legal) | Documented in the package README |
