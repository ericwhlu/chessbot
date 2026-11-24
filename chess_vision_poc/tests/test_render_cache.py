"""Rendering must preserve captured pixels and avoid mesh work on warm moves."""
import hashlib
import chess
import numpy as np
import pytest
from chess_vision import render as renderer
from chess_vision.game import matrix_of


# Captured from the original full-scene rasterizer before the optimization.
@pytest.mark.parametrize('settings,digest',[
    ({},'3caa2a810b525b445d451f221dc9ddcb4e26ce23235e08cbd5374431c35f5a74'),
    ({'tilt':30},'aa4ad1807e0c213ec682a93cb5f05e379f3f24dc55cb914fe3797d7a2d30be9f'),
    ({'tilt':60,'azimuth':137,'rotation':23},'ca9bd22e0bd104a151a39c607fb57cd6af61a3d3188d9a88f8043d35e87c0319'),
])
def test_original_pixels_preserved(settings,digest):
    pixels,_=renderer.render(matrix_of(chess.Board()),renderer.Scene(**settings))
    assert hashlib.sha256(pixels.tobytes()).hexdigest()==digest


def test_warm_move_reuses_layers_without_ghosts(monkeypatch):
    board=chess.Board()
    renderer.render(matrix_of(board))
    board.push_uci('e2e4')
    expected,_=renderer.render(matrix_of(board))
    def unexpected(*args,**kwargs):
        raise AssertionError('A warm move should not rasterize triangles')
    monkeypatch.setattr(renderer,'_triangle',unexpected)
    actual,_=renderer.render(matrix_of(board))
    assert np.array_equal(actual,expected)
    empty,_=renderer.render(matrix_of(chess.Board(None)))
    assert not np.array_equal(empty,actual)
    # Mutating a returned frame must not corrupt shared layers or another visitor.
    actual[:]=0
    fresh,_=renderer.render(matrix_of(board))
    assert np.array_equal(fresh,expected)


def test_baked_layers_match_runtime_layers():
    if not renderer.LAYER_FILE.exists():
        pytest.skip('Run python -m chess_vision.render to build default layers')
    for symbol,row,col in [('P',4,4),('n',0,0),('Q',7,7),('k',3,3)]:
        baked=renderer._default_layers()[(symbol,row,col)]
        generated=renderer._make_piece_layer(symbol,row,col,0,0,0,1,768)
        assert baked[:2]==generated[:2]
        assert np.array_equal(baked[2],generated[2])
        assert np.array_equal(baked[3],generated[3])
