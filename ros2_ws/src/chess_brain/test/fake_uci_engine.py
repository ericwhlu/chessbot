#!/usr/bin/env python3
"""A minimal UCI engine for tests, so the suite runs without Stockfish.

Plays a random legal move (seeded). Environment knobs:

* ``FAKE_ENGINE_SEED``     — RNG seed (default 0)
* ``FAKE_ENGINE_PROMOTE``  — ``q``/``r``/``b``/``n``: when a promotion is available,
                              always promote to this piece (tests the queen substitution)

Run via ``chess.engine.SimpleEngine.popen_uci([sys.executable, __file__])``.
"""

import os
import random
import sys

import chess


def emit(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def main() -> None:
    board = chess.Board()
    rng = random.Random(int(os.environ.get("FAKE_ENGINE_SEED", "0")))
    promote = os.environ.get("FAKE_ENGINE_PROMOTE", "").strip().lower() or None

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        cmd, *rest = line.split()

        if cmd == "uci":
            emit("id name FakeEngine")
            emit("id author chess_brain tests")
            emit("option name Skill Level type spin default 20 min 0 max 20")
            emit("uciok")
        elif cmd == "isready":
            emit("readyok")
        elif cmd == "position":
            moves = []
            if rest and rest[0] == "startpos":
                board = chess.Board()
                if len(rest) > 1 and rest[1] == "moves":
                    moves = rest[2:]
            elif rest and rest[0] == "fen":
                board = chess.Board(" ".join(rest[1:7]))
                if len(rest) > 7 and rest[7] == "moves":
                    moves = rest[8:]
            for uci in moves:
                board.push_uci(uci)
        elif cmd == "go":
            legal = list(board.legal_moves)
            if not legal:
                emit("bestmove (none)")
                continue
            if promote:
                wanted = chess.PIECE_SYMBOLS.index(promote)
                promos = [m for m in legal if m.promotion == wanted]
                if promos:
                    legal = promos
            emit("bestmove " + rng.choice(legal).uci())
        elif cmd == "quit":
            break
        # setoption / ucinewgame / stop / debug: accepted silently


if __name__ == "__main__":
    main()
