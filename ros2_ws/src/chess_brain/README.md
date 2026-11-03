# chess_brain

Chess rules, game-state tracking and Stockfish move selection for Chessmate.
Sits between vision (piece-placement FEN in) and the arm (pick/place actions out).
Design: `docs/CHESS_BRAIN_PLAN.md` at the repo root.

## Status

- [x] `game.py` — `GameState` (full state, move inference from placement), `Engine` (UCI wrapper)
- [x] `cli.py` — play against the real detector: `python -m chess_brain.cli`
- [ ] `actions.py` — move → arm actions
- [ ] `brain_node.py` — ROS 2 node
- [ ] interface contract (see plan §4)

## Run the tests on a Mac (no ROS needed)

```bash
brew install stockfish            # optional; one test is skipped without it
cd ros2_ws/src/chess_brain
python3 -m venv .venv && source .venv/bin/activate
touch .venv/COLCON_IGNORE          # so colcon never crawls the venv
pip install -r requirements-dev.txt
pytest -q
```

Expected: everything passes; `test_real_stockfish_replies_legally` is skipped if Stockfish is missing.

## Interface (draft — see plan §4)

- in:  `chessbot/detected_fen` — piece-placement FEN from vision
- out: `chessbot/arm_actions` — `{"id": 7, "move": "e7e5", "actions": [{"pick": "e7", "place": "e5"}]}`
- in:  `chessbot/arm_done` — `{"id": 7, "status": "ok"}`
