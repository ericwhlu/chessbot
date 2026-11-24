"""Inference boundary. Imports neither renderer nor game/controller.

Input: RGB pixels and four board-plane corner coordinates, nothing else.
Model: masks learned from independently rendered synthetic examples.
"""
from pathlib import Path
import cv2
import numpy as np

MODEL = Path(__file__).resolve().parents[1] / 'models/templates.npz'
CELL = 48


def rectify(image, calibration):
    corners = np.asarray(calibration['corners'],np.float32)
    if corners.shape != (4,2) or not np.isfinite(corners).all():
        raise ValueError('Expected four finite board-plane corners')
    if abs(cv2.contourArea(corners)) < 100:
        raise ValueError('Degenerate calibration')
    end = CELL*8
    h = cv2.getPerspectiveTransform(corners,np.float32([[0,0],[end,0],[end,end],[0,end]]))
    return cv2.warpPerspective(image,h,(end,end))


def masks(crop):
    # Deliberately explicit synthetic material prior. These are RGB thresholds,
    # not IDs or markers. Natural photographs require new segmentation/training.
    gray = cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
    white = (gray>179).astype(np.uint8)
    black = (gray<78).astype(np.uint8)
    # Exclude the one-pixel square rim: interpolation at the outer board edge
    # can blend in the dark scene background and mimic an occupied square.
    for mask in (white,black):
        mask[[0,-1],:]=0
        mask[:,[0,-1]]=0
    return white,black


def feature(mask):
    return cv2.resize(mask.astype(np.float32),(32,32),interpolation=cv2.INTER_AREA).flatten()


def placement_fen(matrix):
    ranks=[]
    for row in matrix:
        text=''; empty=0
        for p in row:
            if p=='.': empty+=1
            else:
                if empty: text+=str(empty); empty=0
                text+=p
        if empty: text+=str(empty)
        ranks.append(text)
    return '/'.join(ranks)


class Detector:
    def __init__(self,path=MODEL):
        if not Path(path).exists():
            raise FileNotFoundError('Build the synthetic template model: python -m chess_vision.train')
        with np.load(path,allow_pickle=False) as data:
            self.templates = data['features'].astype(np.float32)
            self.labels = data['labels']
        self.mass = self.templates.sum(1)
        self.kinds = np.array(list('PNBRQK'))

    def detect(self,image,calibration):
        warped=rectify(image,calibration)
        matrix=[]; predictions=[]; warnings=[]
        for row in range(8):
            rank=[]
            for col in range(8):
                crop=warped[row*CELL:(row+1)*CELL,col*CELL:(col+1)*CELL]
                white,black=masks(crop)
                white_count,black_count=int(white.sum()),int(black.sum())
                occupied=max(white_count,black_count)
                square=f'{"abcdefgh"[col]}{8-row}'
                symbol='.'; similarity=1.; margin=1.; confidence=1.
                if occupied>70:
                    f=feature(white if white_count>black_count else black)
                    intersection=np.minimum(self.templates,f).sum(1)
                    scores=intersection/(self.mass+f.sum()-intersection+1e-6)
                    class_scores=np.array([scores[self.labels==k].max() for k in self.kinds])
                    order=np.argsort(class_scores)
                    symbol=str(self.kinds[order[-1]])
                    if black_count>white_count: symbol=symbol.lower()
                    similarity=float(class_scores[order[-1]])
                    margin=float(class_scores[order[-1]]-class_scores[order[-2]])
                    confidence=float(np.clip(similarity*min(1,margin/.10),0,1))
                    if similarity<.76 or margin<.018:
                        warnings.append(f'{square}: weak template match or ambiguous silhouette')
                rank.append(symbol)
                predictions.append({'square':square,'piece':symbol,'confidence':round(confidence,4),
                                    'similarity':round(similarity,4),'margin':round(margin,4)})
            matrix.append(rank)
        return {'matrix':matrix,'placement_fen':placement_fen(matrix),
                'white_count':sum(p.isupper() for r in matrix for p in r),
                'black_count':sum(p.islower() for r in matrix for p in r),
                'predictions':predictions,'warnings':warnings,
                'confidence_note':'Heuristic template similarity and class separation; not calibrated probabilities.'}


def annotate(image,calibration,result):
    out=rectify(image,calibration)
    out=cv2.resize(out,(768,768),interpolation=cv2.INTER_NEAREST)
    for i,p in enumerate(result['predictions']):
        row,col=divmod(i,8); x,y=col*96,row*96
        color=(122,237,188) if p['confidence']>=.5 else (255,151,97)
        cv2.rectangle(out,(x,y),(x+95,y+95),(60,85,90),1)
        if p['piece']!='.':
            cv2.rectangle(out,(x+2,y+2),(x+85,y+21),(20,30,40),-1)
            cv2.putText(out,f"{p['piece']} {p['confidence']:.2f}",(x+5,y+16),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1,cv2.LINE_AA)
    return out
