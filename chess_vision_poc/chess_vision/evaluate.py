"""Rendered-image benchmark. Saves evidence, actual scores, and failure cases."""
import json
from dataclasses import asdict
from pathlib import Path
import time
import chess
import numpy as np
from .render import Scene,render,png_bytes
from .vision import Detector,annotate
from .game import matrix_of,evaluate

ARTIFACTS=Path(__file__).resolve().parents[1]/'artifacts'


def scenarios():
    board=chess.Board()
    yield 'initial',board.copy(),Scene(),True
    for move in ['e2e4','e7e5','g1f3','b8c6']:
        board.push_uci(move)
        yield f'after_{move}',board.copy(),Scene(),True
    capture=chess.Board()
    for move in ['e2e4','d7d5','e4d5']: capture.push_uci(move)
    yield 'capture',capture,Scene(),True
    yield 'empty',chess.Board(None),Scene(),True
    mixed=chess.Board(None)
    # Every type and color on both square colors, away from training locations.
    for i,symbol in enumerate('PNBRQKpnbrqk'):
        for parity in (0,1):
            square=chess.square((i%4)*2+parity,(i//4)*2)
            mixed.set_piece_at(square,chess.Piece.from_symbol(symbol))
    yield 'all_types_both_colors_and_tiles',mixed,Scene(),True
    rng=np.random.default_rng(2031)
    for n in range(3):
        arbitrary=chess.Board(None)
        for sq in rng.choice(64,32,replace=False):
            arbitrary.set_piece_at(int(sq),chess.Piece.from_symbol(str(rng.choice(list('PNBRQKpnbrqk')))))
        yield f'arbitrary_{n}',arbitrary,Scene(rotation=7+n*43),True
    for name,fen,move in [
        ('castling','r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1','e1g1'),
        ('en_passant','4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1','e5d6'),
        ('promotion','4k3/P7/8/8/8/8/8/4K3 w - - 0 1','a7a8n')]:
        special=chess.Board(fen);special.push_uci(move)
        yield name,special,Scene(),True
    for name,scene in [
        ('rotation_23',Scene(rotation=23)),
        ('rotation_137',Scene(rotation=137)),
        ('oblique_18',Scene(tilt=18,rotation=23)),
        ('oblique_30',Scene(tilt=30,rotation=37)),
        ('camera_azimuth',Scene(tilt=18,azimuth=12,rotation=83)),
        ('light_dim',Scene(light=.9)),
        ('light_bright',Scene(light=1.1)),
        ('palette_slate',Scene(palette='slate')),
        ('palette_clay',Scene(palette='clay')),
        ('noise_4',Scene(noise=4)),
        ('no_shadows',Scene(shadows=False)),
        ('combined',Scene(tilt=18,rotation=53,light=.95,palette='slate',noise=3)),
        ('stress_60_degrees',Scene(tilt=60,rotation=19)),
        ('stress_dim_075',Scene(light=.75)),
        ('stress_noise_12',Scene(noise=12))]:
        yield name,board.copy(),scene,False


def benchmark():
    ARTIFACTS.mkdir(exist_ok=True);detector=Detector();results=[]
    for name,board,scene,required in scenarios():
        start=time.perf_counter();truth=matrix_of(board)
        image,cal=render(truth,scene)
        # PNG round trip: evaluation consumes the same byte representation as UI.
        import cv2
        raw=png_bytes(image)
        decoded=cv2.cvtColor(cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        prediction=detector.detect(decoded,cal)
        score=evaluate(prediction,truth)
        item={'name':name,'required_baseline':required,'scene':asdict(scene),**score,
              'expected_fen':board.board_fen(),'detected_fen':prediction['placement_fen'],
              'warnings':prediction['warnings'],'elapsed_ms':round((time.perf_counter()-start)*1000)}
        results.append(item)
        (ARTIFACTS/f'{name}.png').write_bytes(raw)
        (ARTIFACTS/f'{name}_annotated.png').write_bytes(png_bytes(annotate(decoded,cal,prediction)))
        (ARTIFACTS/f'{name}.json').write_text(json.dumps({'calibration':cal,'prediction':prediction,'evaluation':score},indent=2)+'\n')
        print(f'{name:34} {score["correct"]:2}/64  warnings={len(prediction["warnings"])}',flush=True)
    report={'method':'synthetic silhouette templates from rendered pixels',
            'tests':results,'total_correct':sum(x['correct'] for x in results),'total_squares':64*len(results),
            'baseline_passed':all(x['correct']==64 for x in results if x['required_baseline'])}
    (ARTIFACTS/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Rendered-image benchmark','',
           'All predictions were computed from decoded PNG pixels and camera calibration before comparison with ground truth.',
           '', '| Case | Correct squares | Warnings |','| --- | ---: | ---: |']
    lines += [f'| {x["name"]} | {x["correct"]}/64 | {len(x["warnings"])} |' for x in results]
    lines += ['','## Actual failures','']
    for x in results:
        if x['errors']: lines.append(f'- **{x["name"]}**: '+', '.join(f'{e["square"]}: {e["expected"]} → {e["predicted"]}' for e in x['errors']))
    (ARTIFACTS/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return report


if __name__=='__main__':
    report=benchmark()
    raise SystemExit(0 if report['baseline_passed'] else 1)
