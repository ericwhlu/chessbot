"""Exercise a deployed Chessmate API using two independent visitor cookies."""
import argparse
from http.cookiejar import CookieJar
import json
from pathlib import Path
import time
from urllib.request import build_opener,HTTPCookieProcessor,Request
from urllib.error import HTTPError


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('url')
    args=parser.parse_args()
    base=args.url.rstrip('/')
    def visitor():
        opener=build_opener(HTTPCookieProcessor(CookieJar()))
        def call(path,payload=None):
            body=json.dumps(payload).encode() if payload is not None else None
            response=opener.open(Request(base+path,data=body,headers={'Content-Type':'application/json'}),timeout=120)
            return json.load(response)
        return call
    alice,bob=visitor(),visitor()
    checks=[]
    started=time.monotonic()
    assert alice('/healthz')=={'app':'Chessmate','status':'ok'}
    checks.append('public health endpoint')
    initial=alice('/api/frame',{})
    initial_input={k:initial[k] for k in ('image','calibration','frame_id')}
    predicted=alice('/api/detect',initial_input)
    assert predicted['evaluation']['correct']==64
    assert predicted['white_count']==16 and predicted['black_count']==16
    checks.append('starting position pixels: 64/64')
    alice('/api/action',{'action':'move','from':'e2','to':'e4'})
    assert bob('/api/state')['matrix'][6][4]=='P'
    assert bob('/api/state')['matrix'][4][4]=='.'
    checks.append('independent visitor boards')
    moved=alice('/api/frame',{})
    result=alice('/api/detect',{**{k:moved[k] for k in ('image','calibration','frame_id')},'suggest':True})
    assert result['evaluation']['correct']==64
    assert result['matrix'][6][4]=='.' and result['matrix'][4][4]=='P'
    assert 'suggestion' in result,result.get('engine_error')
    assert result['suggestion']['input_fen'].split()[0]==result['placement_fen']
    checks.append('e2e4 pixels: 64/64 and real Stockfish response')
    try:
        bob('/api/detect',initial_input)
        raise AssertionError('Cross-visitor frame accepted')
    except HTTPError as error:
        assert error.code==400
    checks.append('cross-visitor frame refused')
    bob('/api/action',{'action':'clear'})
    empty=bob('/api/frame',{})
    empty_result=bob('/api/detect',{k:empty[k] for k in ('image','calibration','frame_id')})
    assert empty_result['evaluation']['correct']==64
    assert empty_result['placement_fen']=='8/8/8/8/8/8/8/8'
    checks.append('empty board pixels: 64/64')
    report={'url':base,'passed':True,'checks':checks,
            'elapsed_seconds':round(time.monotonic()-started,2),
            'engine_result':result['suggestion']}
    output=Path(__file__).resolve().parents[1]/'artifacts/hosted_report.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
