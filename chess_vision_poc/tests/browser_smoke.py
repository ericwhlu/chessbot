"""Real browser interactions. Run against the local server (not through pytest)."""
import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]


def main():
    with sync_playwright() as p:
        executable=os.environ.get('CHROME_PATH','/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
        browser=p.chromium.launch(executable_path=executable if Path(executable).exists() else None,headless=True)
        page=browser.new_page(viewport={'width':1512,'height':1200},device_scale_factor=1)
        errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(os.environ.get('CHESSMATE_URL','http://127.0.0.1:8765'))
        assert page.title()=='Chessmate · Duke Robotics'

        def idle(): page.wait_for_function("typeof busy !== 'undefined' && !busy && !!frame")
        def click_id(name):
            page.locator('#'+name).click();idle()
        def square(name):
            position=page.evaluate("s => { const [x,y]=point('abcdefgh'.indexOf(s[0])+.5,8-Number(s[1])+.5); const r=canvas.getBoundingClientRect(); return {x:r.left+x*r.width/768,y:r.top+y*r.height/768}; }",name)
            page.mouse.click(position['x'],position['y']);idle()

        idle();click_id('reset');click_id('playMode');click_id('detect')
        assert page.locator('#accuracy').inner_text()=='64 / 64'
        assert page.locator('#whiteCount').inner_text()=='16'
        page.screenshot(path=str(ROOT/'artifacts/browser_initial.png'),full_page=True)
        page.locator('#auto').check();idle()
        for move in ['e2e4','e7e5','g1f3','b8c6']:
            square(move[:2]);square(move[2:]);idle()
            assert page.locator('#accuracy').inner_text()=='64 / 64'
        assert '4P3' in page.locator('#fen').inner_text()
        click_id('suggest')
        engine=page.locator('#engineResult').inner_text()
        assert 'From' in engine,engine
        assert page.evaluate('recommendation !== null')
        page.screenshot(path=str(ROOT/'artifacts/browser_move_suggestion.png'),full_page=True)
        page.locator('#tilt').fill('30');page.locator('#tilt').dispatch_event('change');idle()
        assert page.locator('#accuracy').inner_text()=='64 / 64'
        page.locator('[data-tab="debug"]').click()
        page.screenshot(path=str(ROOT/'artifacts/browser_oblique_debug.png'),full_page=True)
        click_id('clear')
        assert page.locator('#whiteCount').inner_text()=='0'
        assert page.locator('#blackCount').inner_text()=='0'
        assert page.locator('#accuracy').inner_text()=='64 / 64'
        page.locator('#editTool').select_option('place')
        page.locator('#pieceType').select_option('Q');square('d4')
        assert page.locator('#whiteCount').inner_text()=='1'
        page.locator('#pieceColor').select_option('black');page.locator('#pieceType').select_option('N');square('f6')
        assert page.locator('#blackCount').inner_text()=='1'
        page.locator('#editTool').select_option('move');square('d4');square('a3')
        page.locator('#editTool').select_option('remove');square('f6')
        assert page.locator('#blackCount').inner_text()=='0'
        assert page.locator('#accuracy').inner_text()=='64 / 64'
        # Exercise FEN loading and UI underpromotion.
        page.locator('#fenInput').fill('4k3/P7/8/8/8/8/8/4K3 w - - 0 1');click_id('loadFen')
        click_id('playMode');page.locator('#promotion').select_option('n');square('a7');square('a8')
        assert page.evaluate("state.matrix[0][0]")=='N'
        assert page.locator('#accuracy').inner_text()=='64 / 64'
        page.locator('#tilt').fill('0');page.locator('#tilt').dispatch_event('change');idle()
        page.locator('#promotion').select_option('q');click_id('reset')
        page.locator('[data-tab="position"]').click()
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        page.screenshot(path=str(ROOT/'artifacts/browser_mobile.png'),full_page=True)
        assert not errors,errors
        report={'passed':True,'browser':'Chrome headless, real canvas PNG capture',
                'checks':['initial recognition','four legal click moves','Auto Detect','real Stockfish suggestion',
                          '30 degree camera','empty board','edit add both colors','edit free move','edit remove',
                          'FEN load','underpromotion','mobile layout without horizontal overflow','no JavaScript errors'],
                'engine_result':engine,'javascript_errors':errors}
        (ROOT/'artifacts/browser_report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
        browser.close()


if __name__=='__main__': main()
