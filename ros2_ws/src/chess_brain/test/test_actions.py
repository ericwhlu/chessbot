"""Tests for move → arm-action translation."""

import random

import chess
import pytest

from chess_brain.actions import DISCARD, ArmAction, UnsupportedMove, actions_payload, move_to_actions, spare_name


def A(pick, place):
    return ArmAction(pick, place)


def acts(fen, uci, **kw):
    board = chess.Board(fen)
    return move_to_actions(board, chess.Move.from_uci(uci), **kw)


# ------------------------------------------------------------------ table rows
def test_normal_move():
    assert acts(chess.STARTING_FEN, "e2e4") == [A("e2", "e4")]


def test_capture_removes_captured_piece_first():
    assert acts("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2", "e4d5") == [
        A("d5", DISCARD),
        A("e4", "d5"),
    ]


@pytest.mark.parametrize(
    "fen,uci,expected",
    [
        ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "e1g1", [A("e1", "g1"), A("h1", "f1")]),
        ("r3kbnr/pppqpppp/2n5/3p1b2/3P1B2/2N5/PPPQPPPP/R3KBNR w KQkq - 6 5", "e1c1", [A("e1", "c1"), A("a1", "d1")]),
        ("rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 5 4", "e8g8", [A("e8", "g8"), A("h8", "f8")]),
        ("r3kbnr/pppqpppp/2n5/3p1b2/3P1B2/2N5/PPPQPPPP/R3KBNR b KQkq - 6 5", "e8c8", [A("e8", "c8"), A("a8", "d8")]),
    ],
    ids=["white-K", "white-Q", "black-K", "black-Q"],
)
def test_castling_king_then_rook(fen, uci, expected):
    assert acts(fen, uci) == expected


def test_castling_king_takes_rook_encoding_still_places_king_on_g1():
    # python-chess accepts "e1h1" as legal castling on a standard board; the king
    # must still go to g1, never onto the rook's square.
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    assert acts(fen, "e1h1") == [A("e1", "g1"), A("h1", "f1")]


def test_en_passant_white():
    assert acts("rnbqkbnr/1pp1pppp/p7/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3", "e5d6") == [
        A("d5", DISCARD),
        A("e5", "d6"),
    ]


def test_en_passant_black():
    assert acts("rnbqkbnr/ppp1pppp/8/8/P2pP3/8/1PPP1PPP/RNBQKBNR b KQkq e3 0 3", "d4e3") == [
        A("e4", DISCARD),
        A("d4", "e3"),
    ]


def test_promotion_unsupported_by_default():
    with pytest.raises(UnsupportedMove):
        acts("8/P6k/8/8/8/8/8/K7 w - - 0 1", "a7a8q")


def test_promotion_queen_when_allowed():
    assert acts("8/P6k/8/8/8/8/8/K7 w - - 0 1", "a7a8q", allow_promotion=True) == [
        A("a7", DISCARD),
        A("SPARE_Q_W", "a8"),
    ]


def test_promotion_with_capture_black():
    assert acts("7k/8/8/8/8/8/1p6/R6K b - - 0 1", "b2a1q", allow_promotion=True) == [
        A("a1", DISCARD),
        A("b2", DISCARD),
        A("SPARE_Q_B", "a1"),
    ]


def test_promotion_with_capture_white():
    assert acts("r6k/1P6/8/8/8/8/8/1K6 w - - 0 1", "b7a8q", allow_promotion=True) == [
        A("a8", DISCARD),
        A("b7", DISCARD),
        A("SPARE_Q_W", "a8"),
    ]


def test_under_promotion_unsupported_even_when_allowed():
    with pytest.raises(UnsupportedMove):
        acts("8/P6k/8/8/8/8/8/K7 w - - 0 1", "a7a8n", allow_promotion=True)


def test_illegal_move_rejected():
    with pytest.raises(ValueError):
        acts(chess.STARTING_FEN, "e2e5")


def test_spare_names():
    assert spare_name(chess.QUEEN, chess.WHITE) == "SPARE_Q_W"
    assert spare_name(chess.QUEEN, chess.BLACK) == "SPARE_Q_B"


def test_payload_shape():
    mv = chess.Move.from_uci("e2e4")
    p = actions_payload(7, mv, [A("e2", "e4")])
    assert p == {"id": 7, "move": "e2e4", "actions": [{"pick": "e2", "place": "e4"}]}


# --------------------------------------------------- physical-board simulation
def simulate(board: chess.Board, actions):
    """Execute the actions on a dict model of the physical board and return it.

    Asserts the arm-team invariants: pick is occupied, place is empty.
    """
    phys = {chess.square_name(sq): p for sq, p in board.piece_map().items()}
    for a in actions:
        if a.pick.startswith("SPARE_"):
            _, sym, col = a.pick.split("_")
            piece = chess.Piece(chess.PIECE_SYMBOLS.index(sym.lower()), col == "W")
        else:
            assert a.pick in phys, f"pick from empty square {a.pick} in {actions}"
            piece = phys.pop(a.pick)
        if a.place != DISCARD:
            assert a.place not in phys, f"place onto occupied square {a.place} in {actions}"
            phys[a.place] = piece
    return phys


def expected_after(board: chess.Board, move: chess.Move):
    after = board.copy(stack=False)
    after.push(move)
    return {chess.square_name(sq): p for sq, p in after.piece_map().items()}


# A promotion-rich start so the simulation actually exercises that branch.
PROMO_FEN = "4k3/P1P3P1/8/8/8/8/p1p3p1/4K3 w - - 0 1"  # no pawn attacks either king


@pytest.mark.parametrize("seed", range(40))
def test_actions_reproduce_the_position_for_every_legal_move(seed):
    rng = random.Random(seed)
    board = chess.Board(PROMO_FEN if seed % 4 == 0 else chess.STARTING_FEN)
    for _ in range(60):
        legal = list(board.legal_moves)
        if not legal:
            break
        for move in legal:
            try:
                actions = move_to_actions(board, move, allow_promotion=True)
            except UnsupportedMove:
                assert move.promotion not in (None, chess.QUEEN)  # only under-promotions
                continue
            assert 1 <= len(actions) <= 3
            assert simulate(board, actions) == expected_after(board, move), (board.fen(), move.uci(), actions)
        special = [m for m in legal if board.is_castling(m) or board.is_en_passant(m) or m.promotion]
        board.push(rng.choice(special) if special and rng.random() < 0.8 else rng.choice(legal))
