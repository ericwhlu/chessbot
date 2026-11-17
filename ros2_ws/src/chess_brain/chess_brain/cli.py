"""Interactive CLI to exercise chess_brain against a real detector (Atharva's demo site).

Workflow (robot = black by default):

  1. On the site, make the human's (white) move, click "Detect Board", copy the
     "DETECTED PIECE-PLACEMENT FEN", paste it here.
  2. chess_brain infers your move, asks Stockfish for the robot's reply and prints it.
  3. Make the robot's move on the site, Detect Board again, paste the FEN here:
     chess_brain verifies the board matches what it expects (this mirrors the
     node's post-move verification).
  4. Repeat. Compare the printed reply with the site's "Suggest Best Move" if you like.

Run from ros2_ws/src/chess_brain with the venv active:

  python -m chess_brain.cli                  # robot black, stockfish on PATH
  python -m chess_brain.cli --robot white
  python -m chess_brain.cli --no-engine      # inference only, no Stockfish

Commands at the prompt: fen | legal | undo | quit
"""

from __future__ import annotations

import argparse
import sys

import chess

from .game import Engine, GameState


def color_name(c: chess.Color) -> str:
    return "white" if c == chess.WHITE else "black"


def show(game: GameState) -> None:
    print()
    print(game.board.unicode(borders=True, empty_square="·"))
    s = game.status()
    who = "HUMAN" if s["human_to_move"] else "ROBOT"
    print(f"{s['turn']} to move ({who}){'  — CHECK' if s['in_check'] else ''}")
    print(f"expected placement: {s['placement']}")


def read_line(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return "quit"


def handle_command(game: GameState, text: str) -> bool:
    """Return True if ``text`` was a command (and was handled)."""
    cmd = text.lower()
    if cmd in ("q", "quit", "exit"):
        raise SystemExit(0)
    if cmd == "fen":
        print(game.fen())
        return True
    if cmd == "legal":
        print(" ".join(sorted(m.uci() for m in game.board.legal_moves)))
        return True
    if cmd == "undo":
        if game.board.move_stack:
            print("undo:", game.board.pop().uci())
        else:
            print("nothing to undo")
        return True
    return False


def human_turn(game: GameState) -> None:
    while True:
        text = read_line("paste detected FEN after YOUR move > ")
        if not text or handle_command(game, text):
            continue
        result = game.infer_human_move(text)
        if result.kind == "move":
            mv = result.move
            print(f"inferred human move: {mv.uci()}  ({game.board.san(mv)})")
            game.apply(mv)
            return
        if result.kind == "no_change":
            print("no change detected — did you move and click Detect Board?")
        elif result.kind == "ambiguous":
            print("AMBIGUOUS (detector bug?):", [m.uci() for m in result.moves])
        else:
            print("illegal move or misread. Type 'legal' to list legal moves, or fix the board on the site.")


def robot_turn(game: GameState, engine: Engine | None) -> None:
    if engine is None:
        text = read_line("no engine: type the robot's move in UCI (e.g. e7e5) > ")
        if handle_command(game, text):
            return
        try:
            mv = chess.Move.from_uci(text)
            game.apply(mv)
        except ValueError as exc:
            print("bad move:", exc)
            return
    else:
        mv = engine.reply(game)
        if mv is None:
            return
        san = game.board.san(mv)
        game.apply(mv)
        print(f"\nROBOT plays: {mv.uci()}  ({san})")

    print(f"make that move on the site, Detect Board, then paste the FEN to verify.")
    print(f"expected placement: {game.placement()}")
    while True:
        text = read_line("paste detected FEN after ROBOT move > ")
        if not text or handle_command(game, text):
            continue
        res = game.infer_human_move(text)  # on a verified board this is 'no_change'
        if res.kind == "no_change":
            print("verified ✔ board matches")
            return
        if res.kind == "move":
            print(f"verified ✔ and you already moved: {res.move.uci()}")
            game.apply(res.move)
            return
        print("MISMATCH: the detected board is not what the robot expected.")
        print("expected:", game.placement())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", choices=["white", "black"], default="black")
    ap.add_argument("--fen", default=chess.STARTING_FEN, help="full 6-field start FEN")
    ap.add_argument("--stockfish", default="stockfish", help="engine path")
    ap.add_argument("--skill", type=int, default=5, help="Stockfish Skill Level 0-20")
    ap.add_argument("--time", type=float, default=0.5, help="think time in seconds")
    ap.add_argument("--no-engine", action="store_true", help="you type the robot's moves")
    args = ap.parse_args(argv)

    robot = chess.WHITE if args.robot == "white" else chess.BLACK
    game = GameState(args.fen, robot_color=robot)

    engine = None
    if not args.no_engine:
        try:
            engine = Engine(args.stockfish, think_time=args.time, skill_level=args.skill)
            print(f"engine: {engine.name}")
        except OSError as exc:  # FileNotFoundError etc.
            print(f"could not start '{args.stockfish}' ({exc}); falling back to --no-engine", file=sys.stderr)

    print(f"robot plays {args.robot}. Commands: fen | legal | undo | quit")
    try:
        while True:
            show(game)
            if game.is_game_over():
                s = game.status()
                print(f"\nGAME OVER: {s['result']} ({s['termination']})")
                return 0
            if game.human_to_move:
                human_turn(game)
            else:
                robot_turn(game, engine)
    finally:
        if engine is not None:
            engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
