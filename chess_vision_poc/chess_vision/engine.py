"""Only a detected placement can enter the engine. Never substitute truth."""
import os
import shutil
from pathlib import Path
import chess
import chess.engine


def engine_path():
    requested=os.environ.get('STOCKFISH_PATH')
    if requested: return requested
    return shutil.which('stockfish') or next((p for p in ['/opt/homebrew/bin/stockfish','/usr/local/bin/stockfish','/usr/games/stockfish',
        str(Path(__file__).resolve().parents[1]/'bin/stockfish')] if Path(p).is_file()),None)


def detected_position(result,controller):
    if result['warnings']:
        raise ValueError('Detection contains uncertain predictions. Inspect the debug view before suggesting a move.')
    # Truth comparison happens AFTER inference and is only a veto.
    if result['placement_fen']!=controller.board_fen():
        raise ValueError('Detected placement differs from the controller. Engine request refused.')
    metadata=controller.fen().split()[1:]
    detected=chess.Board(result['placement_fen']+' '+' '.join(metadata))
    if not detected.is_valid(): raise ValueError(f'Detected position is invalid (status {detected.status()}).')
    if detected.is_game_over(): raise ValueError('This position is already game over.')
    return detected


def suggest(result,controller,path=None):
    detected=detected_position(result,controller)
    path=path or engine_path()
    if not path: raise ValueError('Stockfish is optional and is not installed. Set STOCKFISH_PATH or run brew install stockfish.')
    with chess.engine.SimpleEngine.popen_uci(path,timeout=10) as engine:
        engine.configure({'Threads':1,'Hash':32})
        answer=engine.play(detected,chess.engine.Limit(time=.25))
        if answer.move is None: raise ValueError('Stockfish returned no move')
        move=answer.move
        return {'uci':move.uci(),'san':detected.san(move),
                'from':chess.square_name(move.from_square),'to':chess.square_name(move.to_square),
                'input_fen':detected.fen()}
