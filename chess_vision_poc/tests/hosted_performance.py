"""Measure first capture and click-equivalent moves on a fresh hosted visitor."""
import argparse
from http.cookiejar import CookieJar
import json
from pathlib import Path
import time
from urllib.request import build_opener,HTTPCookieProcessor,Request


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('url')
    args=parser.parse_args()
    base=args.url.rstrip('/')
    opener=build_opener(HTTPCookieProcessor(CookieJar()))
    timings=[]

    def call(path,payload=None):
        start=time.perf_counter()
        body=None if payload is None else json.dumps(payload).encode()
        with opener.open(Request(base+path,data=body,headers={'Content-Type':'application/json'}),timeout=120) as response:
            data=json.load(response)
        row={'path':path,'seconds':round(time.perf_counter()-start,3)}
        if 'render_ms' in data: row['server_render_ms']=data['render_ms']
        timings.append(row)
        print(json.dumps(row),flush=True)
        return data

    call('/healthz')
    frame=call('/api/frame',{})
    assert 'render_ms' in frame,'Optimized deployment is not live'
    for move in ['e2e4','e7e5','g1f3','b8c6']:
        frame=call('/api/action-frame',{'action':{'action':'move','from':move[:2],'to':move[2:]},'scene':{}})
        assert frame['state']['matrix'][8-int(move[1])]['abcdefgh'.index(move[0])]=='.'
        result=call('/api/detect',{k:frame[k] for k in ('image','calibration','frame_id')})
        assert result['evaluation']['correct']==64
    output=Path(__file__).resolve().parents[1]/'artifacts/performance_after.json'
    output.write_text(json.dumps({'url':base,'passed':True,'timings':timings},indent=2)+'\n')


if __name__=='__main__': main()
