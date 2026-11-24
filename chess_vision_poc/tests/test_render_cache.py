"""Rendering must preserve captured pixels and avoid mesh work on warm moves."""
import hashlib
import chess
import numpy as np
import pytest
from chess_vision import render as renderer
from chess_vision.game import matrix_of


# Reference frames for the Fusion base profiles with the runtime rasterizer.
@pytest.mark.parametrize('settings,digest',[
    ({},'a68207087c9be9a4fc57af20ed1dfa8272fcf9760768705a9e10f9dd9fe400c5'),
    ({'tilt':30},'0db9e9c7419e59ba952f96ba0b9ff09a13c740a42aa9866423ed74163c7e567d'),
    ({'tilt':60,'azimuth':137,'rotation':23},'b94edb4be327af0ad74b2d045e89c7bc3d419e3d85277554184539ff548cf3f4'),
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
