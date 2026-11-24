import ast
import base64
import inspect
import chess
import numpy as np
import pytest
from chess_vision import vision
from chess_vision.game import Game,matrix_of
from chess_vision.geometry import sections,CUBE_WIDTH,CUBE_BOTTOM,CUBE_TOP
from chess_vision.render import render,Scene,png_bytes
from chess_vision.engine import detected_position,suggest,engine_path
from chess_vision.evaluate import benchmark


def test_rendered_image_benchmark():
    report=benchmark()
    assert report['baseline_passed'], [x for x in report['tests'] if x['required_baseline'] and x['correct']!=64]


def test_identical_gripping_cube_and_closed_base_caps():
    reference=sections('P')[1]
    for kind in 'PNBRQK':
        base,cube,top=sections(kind)
        assert all(np.array_equal(a,b) for a,b in zip(cube,reference))
        xyz=np.concatenate(cube)
        assert np.allclose(xyz.max(0)-xyz.min(0),[CUBE_WIDTH]*3)
        assert xyz[:,2].min()==CUBE_BOTTOM and xyz[:,2].max()==CUBE_TOP
        assert max(v[:,2].max() for v in base)==CUBE_BOTTOM
        assert min(v[:,2].min() for v in top)>=CUBE_TOP-1e-8


@pytest.mark.parametrize('fen,move,expected',[
    ('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1','e1g1',{'g1':'K','f1':'R','e1':None,'h1':None}),
    ('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1','e5d6',{'d6':'P','d5':None,'e5':None}),
    ('4k3/P7/8/8/8/8/8/4K3 w - - 0 1','a7a8n',{'a8':'N','a7':None})])
def test_special_moves(fen,move,expected):
    game=Game();game.board=chess.Board(fen)
    game.action({'action':'move','from':move[:2],'to':move[2:4],'promotion':move[4:]})
    for sq,symbol in expected.items():
        piece=game.board.piece_at(chess.parse_square(sq))
        assert (piece.symbol() if piece else None)==symbol


def test_illegal_move_and_edit_mode():
    game=Game()
    with pytest.raises(ValueError,match='Illegal'): game.action({'action':'move','from':'e2','to':'e5'})
    game.action({'action':'mode','mode':'edit'})
    game.action({'action':'move','from':'e2','to':'a5'})
    assert game.board.piece_at(chess.A5).symbol()=='P'
    game.action({'action':'place','square':'c3','piece':'q'})
    assert not game.board.castling_rights
    game.action({'action':'clear'})
    assert not game.board.piece_map()
    with pytest.raises(ValueError,match='valid'): game.action({'action':'mode','mode':'play'})


def test_detector_has_no_renderer_or_game_imports():
    tree=ast.parse(inspect.getsource(vision))
    imported=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    assert all(not any(word in (name or '') for word in ['game','render','geometry','chess']) for name in imported)
    assert list(inspect.signature(vision.Detector.detect).parameters)==['self','image','calibration']


def test_pixel_change_changes_prediction_and_engine_refuses_mismatch():
    # Hold calibration and controller fixed while changing ONLY input pixels.
    board=chess.Board();detector=vision.Detector()
    initial,cal=render(matrix_of(board))
    initial_result=detector.detect(initial,cal)
    board.push_uci('e2e4')
    moved,_=render(matrix_of(board))
    moved_result=detector.detect(moved,cal)
    assert initial_result['matrix'][6][4]=='P'
    assert moved_result['matrix'][6][4]=='.' and moved_result['matrix'][4][4]=='P'
    with pytest.raises(ValueError,match='differs'): detected_position(initial_result,board)
    assert detected_position(moved_result,board).fen()==board.fen()
    uncertain=dict(moved_result,warnings=['e4 uncertain'])
    with pytest.raises(ValueError,match='uncertain'): detected_position(uncertain,board)


def test_invalid_detected_position_refused():
    board=chess.Board(None)
    with pytest.raises(ValueError,match='invalid'):
        detected_position({'placement_fen':board.board_fen(),'warnings':[]},board)


def test_http_pixel_boundary_and_stale_snapshot():
    from chess_vision.app import create_app
    app=create_app();app.testing=True;client=app.test_client()
    frame=client.post('/api/frame',json={}).get_json()
    payload={k:frame[k] for k in ('image','calibration','frame_id')}
    assert client.post('/api/detect',json={**payload,'board':[]}).status_code==400
    response=client.post('/api/detect',json=payload).get_json()
    assert response['evaluation']['correct']==64
    client.post('/api/action',json={'action':'move','from':'e2','to':'e4'})
    response=client.post('/api/detect',json={**payload,'suggest':True}).get_json()
    assert response['stale'] and 'changed since capture' in response['engine_error']
    # Supply an empty rendered image with the original frame's controller truth.
    empty,_=render([['.']*8 for _ in range(8)])
    response=client.post('/api/detect',json={**payload,'image':'data:image/png;base64,'+base64.b64encode(png_bytes(empty)).decode()}).get_json()
    assert response['white_count']==0 and response['black_count']==0
    assert response['evaluation']['correct']==32


@pytest.mark.skipif(not engine_path(),reason='Optional Stockfish is not installed')
def test_real_stockfish_from_detected_pixels():
    board=chess.Board();board.push_uci('e2e4')
    image,cal=render(matrix_of(board))
    result=vision.Detector().detect(image,cal)
    answer=suggest(result,board)
    move=chess.Move.from_uci(answer['uci'])
    assert move in board.legal_moves
    assert answer['input_fen']==board.fen()
    assert answer['san']==board.san(move)
