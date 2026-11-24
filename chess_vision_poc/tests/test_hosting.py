import base64
import struct
import pytest
from chess_vision.app import create_app


def test_visitors_have_independent_boards_and_private_frames():
    app=create_app();app.testing=True
    alice=app.test_client();bob=app.test_client()
    assert alice.post('/api/action',json={'action':'move','from':'e2','to':'e4'}).status_code==200
    assert alice.get('/api/state').get_json()['matrix'][4][4]=='P'
    assert bob.get('/api/state').get_json()['matrix'][4][4]=='.'
    assert bob.get('/api/state').get_json()['matrix'][6][4]=='P'
    frame=alice.post('/api/frame',json={}).get_json()
    payload={k:frame[k] for k in ('image','calibration','frame_id')}
    assert bob.post('/api/detect',json=payload).status_code==400
    result=alice.post('/api/detect',json=payload).get_json()
    assert result['evaluation']['correct']==64
    assert result['matrix'][4][4]=='P'
    # API clients without a page key keep their own cookie-backed board.
    assert alice.get('/api/state').get_json()['turn']=='black'
    assert bob.get('/api/state').get_json()['turn']=='white'


def test_new_page_gets_opening_board_while_original_page_keeps_its_game():
    client = create_app().test_client()
    first = {'X-Chessmate-Board': 'a' * 32}
    second = {'X-Chessmate-Board': 'b' * 32}
    opening = client.post('/api/frame', json={}, headers=first).get_json()['state']
    move = {'action': {'action': 'move', 'from': 'e2', 'to': 'e4'}, 'scene': {}}
    moved = client.post('/api/action-frame', json=move, headers=first).get_json()
    assert moved['state']['turn'] == 'black'
    assert client.post('/api/frame', json={}, headers=second).get_json()['state'] == opening
    assert client.get('/api/state', headers=first).get_json()['matrix'][4][4] == 'P'
    payload = {k: moved[k] for k in ('image', 'calibration', 'frame_id')}
    assert client.post('/api/detect', json=payload, headers=second).status_code == 400


@pytest.mark.parametrize('payload',[[],{'rotation':float('nan')},{'size':768.5},{'seed':-1},{'shadows':'true'}])
def test_rejects_malformed_public_scene_requests(payload):
    client=create_app().test_client()
    assert client.post('/api/frame',json=payload).status_code==400


def test_image_dimensions_checked_before_decode():
    client=create_app().test_client()
    frame=client.post('/api/frame',json={}).get_json()
    payload={k:frame[k] for k in ('image','calibration','frame_id')}
    raw=bytearray(base64.b64decode(frame['image'].split(',')[1]))
    raw[16:24]=struct.pack('>II',100000,100000)
    payload['image']='data:image/png;base64,'+base64.b64encode(raw).decode()
    response=client.post('/api/detect',json=payload)
    assert response.status_code==400
    assert 'dimensions' in response.get_json()['error']


def test_brand_health_and_secure_cookie(monkeypatch):
    monkeypatch.setenv('CHESSMATE_HTTPS','1')
    client=create_app().test_client()
    assert client.get('/healthz').get_json()=={'status':'ok','app':'Chessmate'}
    html=client.get('/').get_data(as_text=True)
    assert 'Chessmate · Duke Robotics' in html and 'CHESS VISION' not in html
    response=client.get('/api/state',base_url='https://example.test')
    cookie=response.headers['Set-Cookie']
    assert 'Secure' in cookie and 'HttpOnly' in cookie and 'SameSite=Lax' in cookie
    assert response.headers['Cache-Control']=='no-store'


def test_move_and_capture_in_one_request():
    client=create_app().test_client()
    action={'action':'move','from':'e2','to':'e4'}
    rejected=client.post('/api/action-frame',json={'action':action,'scene':{'tilt':999}})
    assert rejected.status_code==400
    assert client.get('/api/state').get_json()['matrix'][6][4]=='P'
    response=client.post('/api/action-frame',json={'action':action,'scene':{}})
    assert response.status_code==200
    frame=response.get_json()
    assert frame['state']['matrix'][4][4]=='P'
    assert frame['state']['revision']==1
    result=client.post('/api/detect',json={k:frame[k] for k in ('image','calibration','frame_id')}).get_json()
    assert result['evaluation']['correct']==64
    assert result['matrix'][6][4]=='.' and result['matrix'][4][4]=='P'
