"""Chessmate HTTP application with visitor-isolated boards and snapshots."""
import argparse
import base64
from collections import OrderedDict
from dataclasses import fields, dataclass, field
import os
import secrets
import struct
from pathlib import Path
import threading
import time
import uuid
import cv2
import chess.engine
import numpy as np
from flask import Flask,jsonify,request,send_from_directory,session,g
from .game import Game,matrix_of,evaluate
from .render import Scene,render,png_bytes,warm_renderer
from .vision import Detector,annotate
from .engine import suggest,engine_path

ROOT=Path(__file__).resolve().parents[1]


@dataclass
class Visitor:
    game: Game = field(default_factory=Game)
    snapshots: OrderedDict = field(default_factory=OrderedDict)
    lock: object = field(default_factory=threading.RLock)
    last_seen: float = field(default_factory=time.monotonic)


def create_app():
    app=Flask(__name__,static_folder=str(ROOT/'static'),static_url_path='/static')
    app.config['MAX_CONTENT_LENGTH']=6*1024*1024
    app.config.update(SECRET_KEY=os.environ.get('CHESSMATE_SECRET_KEY') or secrets.token_hex(32),
                      SESSION_COOKIE_NAME='chessmate_session',SESSION_COOKIE_HTTPONLY=True,
                      SESSION_COOKIE_SAMESITE='Lax',
                      SESSION_COOKIE_SECURE=os.environ.get('CHESSMATE_HTTPS')=='1')
    detector=Detector(); warm_renderer()
    visitors=OrderedDict(); registry_lock=threading.RLock()
    work_slots=threading.BoundedSemaphore(2)

    @app.before_request
    def identify_visitor():
        if not request.path.startswith('/api/'):
            return
        if request.method=='POST' and not isinstance(request.get_json(),dict):
            raise ValueError('Expected a JSON object')
        if request.path in ('/api/frame','/api/action-frame','/api/detect'):
            if not work_slots.acquire(blocking=False):
                return jsonify(error='The demo is busy. Please try again in a moment.'),503
            g.has_work_slot=True
        now=time.monotonic()
        with registry_lock:
            for key in list(visitors):
                if now-visitors[key].last_seen>7200:
                    del visitors[key]
            sid=session.get('visitor_id')
            if not isinstance(sid,str) or sid not in visitors:
                sid=secrets.token_urlsafe(32)
                session['visitor_id']=sid
                visitors[sid]=Visitor()
                while len(visitors)>256:
                    visitors.popitem(last=False)
            g.visitor=visitors[sid]
            g.visitor.last_seen=now
            visitors.move_to_end(sid)

    @app.teardown_request
    def release_work_slot(error):
        if g.pop('has_work_slot',False):
            work_slots.release()

    @app.after_request
    def response_headers(response):
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='same-origin'
        if request.path.startswith('/api/'):
            response.headers['Cache-Control']='no-store'
        return response

    @app.errorhandler(ValueError)
    @app.errorhandler(KeyError)
    @app.errorhandler(TypeError)
    def bad_input(error):
        return jsonify(error=str(error)),400

    @app.get('/')
    def index(): return send_from_directory(ROOT/'static','index.html')

    @app.get('/healthz')
    def health(): return jsonify(status='ok',app='Chessmate')

    @app.get('/api/state')
    def state():
        game,lock=g.visitor.game,g.visitor.lock
        with lock: return jsonify(**game.state(),stockfish_available=bool(engine_path()))

    @app.post('/api/action')
    def action():
        game,lock=g.visitor.game,g.visitor.lock
        with lock: return jsonify(game.action(request.get_json()))

    def parse_scene(data):
        if not isinstance(data,dict): raise ValueError('Expected scene settings')
        allowed={f.name for f in fields(Scene)}
        if set(data)-allowed: raise ValueError('Unknown scene setting')
        return Scene(**data).validate()

    @app.post('/api/frame')
    def frame():
        return capture(parse_scene(request.get_json()))

    @app.post('/api/action-frame')
    def action_frame():
        data=request.get_json()
        scene=parse_scene(data['scene'])
        if not isinstance(data['action'],dict): raise ValueError('Expected an action object')
        return capture(scene,data['action'])

    def capture(scene,action=None):
        started=time.perf_counter()
        game,lock,snapshots=g.visitor.game,g.visitor.lock,g.visitor.snapshots
        with lock:
            if action is not None: game.action(action)
            truth=matrix_of(game.board); board=game.board.copy(); state=game.state()
        pixels,cal=render(truth,scene)
        token=uuid.uuid4().hex
        with lock:
            snapshots[token]={'truth':truth,'board':board,'revision':state['revision'],'calibration':cal}
            while len(snapshots)>16: snapshots.popitem(last=False)
        image='data:image/png;base64,'+base64.b64encode(png_bytes(pixels)).decode()
        return jsonify(image=image,calibration=cal,frame_id=token,state=state,
                       render_ms=round((time.perf_counter()-started)*1000,1))

    @app.post('/api/detect')
    def detect():
        game,lock,snapshots=g.visitor.game,g.visitor.lock,g.visitor.snapshots
        started=time.perf_counter(); data=request.get_json()
        if set(data)-{'image','calibration','frame_id','suggest'}:
            raise ValueError('Detection accepts pixels, calibration, frame ID and suggestion flag only')
        with lock:
            snapshot=snapshots.get(data['frame_id'])
            if snapshot is None: raise ValueError('Frame expired; capture a new frame')
        if data['calibration']!=snapshot['calibration']:
            raise ValueError('Calibration does not match captured frame')
        if not data['image'].startswith('data:image/png;base64,'): raise ValueError('Expected a PNG capture')
        try: raw=base64.b64decode(data['image'].split(',',1)[1],validate=True)
        except Exception as e: raise ValueError('Invalid image encoding') from e
        if (len(raw)<33 or raw[:8]!=b'\x89PNG\r\n\x1a\n' or raw[12:16]!=b'IHDR'
                or list(struct.unpack('>II',raw[16:24]))!=data['calibration']['image_size']):
            raise ValueError('PNG dimensions do not match calibration')
        bgr=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2]!=tuple(reversed(data['calibration']['image_size'])):
            raise ValueError('Image dimensions do not match calibration')
        pixels=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        # The only inference call: no board, labels, object IDs or controller.
        result=detector.detect(pixels,data['calibration'])
        result['evaluation']=evaluate(result,snapshot['truth'])
        result['debug_image']='data:image/png;base64,'+base64.b64encode(png_bytes(annotate(pixels,data['calibration'],result))).decode()
        result['elapsed_ms']=round((time.perf_counter()-started)*1000)
        result['revision']=snapshot['revision']
        with lock: result['stale']=snapshot['revision']!=game.revision
        if data.get('suggest'):
            with lock:
                if result['stale']: result['engine_error']='Board changed since capture. Capture again.'
                else:
                    try: result['suggestion']=suggest(result,snapshot['board'])
                    except (ValueError,OSError,RuntimeError,chess.engine.EngineError,TimeoutError) as e: result['engine_error']=str(e)
        return jsonify(result)

    return app


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--open',action='store_true',help='Open the local application in your default browser')
    args=parser.parse_args()
    if args.open:
        import webbrowser
        timer=threading.Timer(1.0,lambda:webbrowser.open(f'http://127.0.0.1:{args.port}'))
        timer.daemon=True;timer.start()
    create_app().run(host='127.0.0.1',port=args.port,debug=False,threaded=True)


if __name__=='__main__': main()
