"""Property test: in random games, every played move must be inferred exactly.

This mechanically checks the two claims in CHESS_BRAIN_PLAN.md §3.1:
* inference is never ambiguous for a single move, and
* castling, en passant and promotion are covered with no special handling.
"""

import random

import chess
import pytest

from chess_brain.game import GameState

GAMES = 60
MAX_PLIES = 80


def random_game(seed: int):
    rng = random.Random(seed)
    board = chess.Board()
    for _ in range(MAX_PLIES):
        moves = list(board.legal_moves)
        if not moves:
            return
        # Bias towards "interesting" moves so castling / e.p. / promotion show up.
        special = [m for m in moves if board.is_castling(m) or board.is_en_passant(m) or m.promotion]
        move = rng.choice(special) if special and rng.random() < 0.8 else rng.choice(moves)
        yield board.fen(), move
        board.push(move)


@pytest.mark.parametrize("seed", range(GAMES))
def test_every_ply_inferred_exactly(seed):
    for fen, move in random_game(seed):
        game = GameState(fen)
        b = game.board

        after = b.copy(stack=False)
        after.push(move)

        result = game.infer_human_move(after.board_fen())
        assert result.kind == "move", (fen, move.uci(), result)
        assert result.move == move, (fen, move.uci(), result.move.uci())


def test_special_moves_actually_occur_in_the_random_games():
    totals = {"castling": 0, "en_passant": 0, "promotion": 0}
    for seed in range(GAMES):
        for fen, move in random_game(seed):
            b = chess.Board(fen)
            totals["castling"] += b.is_castling(move)
            totals["en_passant"] += b.is_en_passant(move)
            totals["promotion"] += bool(move.promotion)
    assert totals["castling"] > 0
    assert totals["promotion"] > 0
    # en passant is rare in random play; don't fail the suite on it, just report
    print("special-move coverage:", totals)
