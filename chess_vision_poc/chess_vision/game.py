"""Chess rules, edit operations, and post-inference evaluation."""
import chess


def matrix_of(board):
    return [[board.piece_at(chess.square(col,7-row)).symbol()
             if board.piece_at(chess.square(col,7-row)) else '.' for col in range(8)] for row in range(8)]


def evaluate(result,truth):
    errors=[]
    for row in range(8):
        for col in range(8):
            if result['matrix'][row][col]!=truth[row][col]:
                errors.append({'square':f'{"abcdefgh"[col]}{8-row}',
                               'expected':truth[row][col],'predicted':result['matrix'][row][col]})
    correct=64-len(errors)
    return {'correct':correct,'total':64,'accuracy':correct/64,'errors':errors}


class Game:
    def __init__(self):
        self.board=chess.Board()
        self.mode='play'
        self.revision=0

    def state(self):
        valid=self.board.is_valid()
        return {'matrix':matrix_of(self.board),'fen':self.board.fen(),'mode':self.mode,
                'turn':'white' if self.board.turn else 'black','valid':valid,
                'status':str(self.board.status()),'revision':self.revision,
                'outcome':str(self.board.outcome()) if valid and self.board.is_game_over() else None,
                'legal_moves':[m.uci() for m in self.board.legal_moves] if valid and self.mode=='play' else []}

    def action(self,data):
        action=data['action']
        if action=='reset': self.board.reset()
        elif action=='clear':
            self.board.clear(); self.mode='edit'
        elif action=='mode':
            if data['mode'] not in ('play','edit'): raise ValueError('Unknown mode')
            if data['mode']=='play' and not self.board.is_valid():
                raise ValueError('This arrangement is not a valid chess position; finish editing first.')
            self.mode=data['mode']
        elif action=='turn':
            if self.mode!='edit': raise ValueError('Set turn in edit mode only')
            if data['turn'] not in ('white','black'): raise ValueError('Invalid turn')
            self.board.turn=data['turn']=='white'
        elif action=='move':
            source=chess.parse_square(data['from']); dest=chess.parse_square(data['to'])
            if self.mode=='play':
                if not self.board.is_valid(): raise ValueError('Invalid chess position')
                if self.board.is_game_over(): raise ValueError('The game is over. Reset or edit to continue.')
                move=chess.Move.from_uci(data['from']+data['to']+data.get('promotion',''))
                if move not in self.board.legal_moves: raise ValueError('Illegal move')
                self.board.push(move)
            else:
                piece=self.board.piece_at(source)
                if not piece: raise ValueError('No piece on source square')
                self.board.remove_piece_at(source); self.board.set_piece_at(dest,piece)
                self._clear_history()
        elif action=='place':
            if self.mode!='edit': raise ValueError('Switch to Edit to add or remove pieces')
            square=chess.parse_square(data['square']); symbol=data['piece']
            if symbol=='.': self.board.remove_piece_at(square)
            elif symbol in 'PNBRQKpnbrqk': self.board.set_piece_at(square,chess.Piece.from_symbol(symbol))
            else: raise ValueError('Unknown piece')
            self._clear_history()
        elif action=='fen':
            if self.mode!='edit': raise ValueError('Switch to Edit to load a FEN')
            self.board=chess.Board(data['fen'])
        else: raise ValueError('Unknown action')
        self.revision+=1
        return self.state()

    def _clear_history(self):
        self.board.clear_stack(); self.board.castling_rights=0; self.board.ep_square=None
        self.board.halfmove_clock=0
