"""Tests for the game bookkeeping of play_game.py (no MuJoCo needed)."""
import chess

from chessbot_sim.gameplay import STARTING_SQUARES, HumanPlayer, PieceBook, Report, discard_slots


def play(book, board, uci):
    move = chess.Move.from_uci(uci)
    changes = book.apply(board, move)
    board.push(move)
    return changes


def occupied(board):
    return frozenset(chess.square_name(s) for s in chess.SQUARES if board.piece_at(s))


def test_book_starts_with_32_bodies_on_their_squares():
    book = PieceBook()
    assert len(STARTING_SQUARES) == 32 and book.squares() == occupied(chess.Board())
    assert book.at["e2"] == "e2"


def test_quiet_move_and_capture():
    book, board = PieceBook(), chess.Board()
    assert play(book, board, "e2e4").moved == [("e2", "e4")]
    play(book, board, "d7d5")
    changes = play(book, board, "e4d5")
    assert changes.removed == ["d7"] and changes.moved == [("e2", "d5")]
    assert book.at["d5"] == "e2" and "e4" not in book.at
    assert book.squares() == occupied(board)


def test_en_passant_removes_the_pawn_beside():
    book, board = PieceBook(), chess.Board()
    for uci in ("e2e4", "a7a6", "e4e5", "d7d5"):
        play(book, board, uci)
    changes = play(book, board, "e5d6")
    assert changes.removed == ["d7"] and changes.moved == [("e2", "d6")]
    assert "d5" not in book.at and book.squares() == occupied(board)


def test_castling_moves_king_and_rook():
    book, board = PieceBook(), chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3", "g8f6", "f1c4", "f8c5"):
        play(book, board, uci)
    changes = play(book, board, "e1g1")
    assert changes.moved == [("e1", "g1"), ("h1", "f1")] and changes.removed == []
    play(book, board, "e8g8")
    assert book.at["g8"] == "e8" and book.at["f8"] == "h8"
    assert book.squares() == occupied(board)


def test_queenside_castling_and_promotion():
    book, board = PieceBook(), chess.Board()
    for uci in ("d2d4", "d7d5", "b1c3", "b8c6", "c1f4", "c8f5", "d1d2", "d8d7"):
        play(book, board, uci)
    assert play(book, board, "e1c1").moved == [("e1", "c1"), ("a1", "d1")]
    assert book.squares() == occupied(board)

    book2 = PieceBook()
    board2 = chess.Board()
    for uci in ("h2h4", "g7g5", "h4g5", "g8f6", "g5g6", "f6e4", "g6g7", "e4d6", "g7h8q"):
        play(book2, board2, uci)
    assert book2.at["h8"] == "h2"                           # the pawn's body stands for the new queen
    assert book2.squares() == occupied(board2)


def test_illegal_move_is_refused():
    book = PieceBook()
    try:
        book.apply(chess.Board(), chess.Move.from_uci("e2e5"))
    except ValueError:
        return
    raise AssertionError("an illegal move was accepted")


def test_book_follows_long_random_games():
    for seed in range(8):
        human, book, board = HumanPlayer(seed=seed), PieceBook(), chess.Board()
        for _ in range(120):
            move = human.choose(board)
            if move is None:
                break
            book.apply(board, move)
            board.push(move)
            assert book.squares() == occupied(board), board.fen()
            assert len(set(book.at.values())) == len(book.at)


def test_human_is_reproducible_and_plays_its_script_first():
    a, b = HumanPlayer(seed=3), HumanPlayer(seed=3)
    board = chess.Board()
    assert [a.choose(board).uci() for _ in range(5)] == [b.choose(board).uci() for _ in range(5)]
    scripted = HumanPlayer(seed=1, script=["e2e4", "zz", "e7e5", " D2D4 "])
    assert scripted.choose(board).uci() == "e2e4"
    assert scripted.choose(board).uci() == "d2d4"           # the junk and the illegal move are skipped
    assert scripted.choose(board) in board.legal_moves      # then it chooses by itself


def test_human_promotes_to_a_queen():
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    assert HumanPlayer(script=["a7a8n"]).choose(board).uci() == "a7a8q"
    for seed in range(20):
        assert HumanPlayer(seed=seed).choose(board).promotion in (None, chess.QUEEN)


def test_discard_slots_are_distinct_and_clear_of_the_board():
    slots = discard_slots()
    assert len(slots) == len(set(slots)) == 30
    for x, y in slots:
        assert not (-0.065 <= y <= 0.465)                   # outside the camera's rectified board image
        assert 0.08 <= x <= 0.32                            # between the corner tags (x = 0.03 and 0.37)
    assert discard_slots(3) == slots[:3]


def test_report_counts():
    report = Report()
    report.add(ply=1, who="human", move="e2e4", outcome="ok", detail="")
    report.add(ply=2, who="robot", move="e7e5", outcome="operator", detail="no arm path")
    report.add(ply=3, who="human", move="d4e5", outcome="ambiguous", detail="d4c5 or d4e5")
    assert not report.clean() and report.count("robot", "ok") == 0
    assert Report(plies=[ply for ply in report.plies if ply["who"] == "human"]).clean()   # ambiguity alone is fine
    text = "\n".join(report.lines())
    assert "robot moves:  1" in text and "finished by the operator: 1" in text
    assert "ambiguous, operator confirmed: 1" in text and "no arm path" in text
