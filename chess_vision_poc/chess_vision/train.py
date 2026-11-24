"""Reproducible synthetic template fitting. Ground truth labels only here.

Training scenes deliberately differ from evaluation chess positions. Every
feature is extracted from raster pixels, never from geometric outlines.
"""
import json
from dataclasses import asdict
import numpy as np
from .render import render,Scene
from .vision import MODEL,CELL,rectify,masks,feature


def train():
    features=[]; labels=[]
    # Each board carries all six types on both square colors and piece colors.
    matrix=[['.']*8 for _ in range(8)]
    for row in (1,3,5):
        for col,kind in enumerate('PNBRQK'):
            matrix[row][col+1]=kind if row!=3 else kind.lower()
    for tilt in (0,12,24,36):
        for rotation in range(0,360,15):
            image,cal=render(matrix,Scene(tilt=tilt,rotation=rotation,size=512))
            warped=rectify(image,cal)
            for row in (1,3,5):
                for col in range(1,7):
                    crop=warped[row*CELL:(row+1)*CELL,col*CELL:(col+1)*CELL]
                    white,black=masks(crop)
                    features.append(feature(white if matrix[row][col].isupper() else black))
                    labels.append(matrix[row][col].upper())
        print(f'Fitted camera tilt {tilt} degrees',flush=True)
    MODEL.parent.mkdir(exist_ok=True)
    np.savez_compressed(MODEL,features=np.array(features,np.float32),labels=np.array(labels))
    (MODEL.parent/'training.json').write_text(json.dumps({'method':'nearest synthetic silhouette template / soft IoU',
        'base_geometry':'Fusion STEP footprints, 50 mm squares (v1)', 'templates':len(labels),'tilts':[0,12,24,36],'rotations':list(range(0,360,15)),
        'seed':'deterministic','material_segmentation':'gray > 179 white, gray < 78 black',
        'note':'Includes complete projected geometry; confidence is not a probability.'},indent=2)+'\n')
    print(f'Saved {len(labels)} templates to {MODEL}',flush=True)


if __name__=='__main__':
    train()
