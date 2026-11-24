"""Deterministic CPU 3D rasterizer: orthographic camera, depth buffer, shading.

The browser displays these actual RGB frames. This avoids a GPU/browser driver
dependency and lets the exact same renderer produce reproducible test images.
"""
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from .geometry import piece_faces, base_outline


@dataclass
class Scene:
    tilt: float = 0
    azimuth: float = 0
    rotation: float = 0
    light: float = 1.0
    palette: str = 'sage'
    shadows: bool = True
    noise: float = 0
    seed: int = 42
    size: int = 768

    def validate(self):
        numbers=(self.tilt,self.azimuth,self.rotation,self.light,self.noise)
        if (any(type(x) not in (int,float) or not np.isfinite(x) for x in numbers)
                or type(self.size) is not int or type(self.seed) is not int
                or not 0<=self.seed<=2**32-1 or type(self.shadows) is not bool
                or not isinstance(self.palette,str) or not -360<=self.rotation<=360):
            raise ValueError('Invalid scene settings')
        if not (0 <= self.tilt <= 60 and -180 <= self.azimuth <= 180
                and .75 <= self.light <= 1.2 and 0 <= self.noise <= 12
                and 256 <= self.size <= 1024 and self.palette in PALETTES):
            raise ValueError('Unsupported camera, lighting, size, noise, or palette')
        return self


PALETTES = {'sage':[(160,158,132),(100,128,120)],
            'slate':[(154,160,169),(102,117,141)],
            'clay':[(167,151,131),(139,108,99)]}


def camera(scene):
    a,t = np.deg2rad([scene.azimuth,scene.tilt])
    # Screen right, screen down, depth towards camera.
    right = np.array([np.cos(a),np.sin(a),0])
    down = np.array([-np.sin(a)*np.cos(t),np.cos(a)*np.cos(t),-np.sin(t)])
    toward = np.cross(right,down)
    scale = scene.size / 10.9
    return np.array([right,down,toward]),scale


def project(points, scene):
    basis,scale = camera(scene)
    p = (np.asarray(points)-[4,4,0]) @ basis.T
    p[:,:2] = p[:,:2]*scale + scene.size/2
    return p


def calibration(scene):
    return {'corners':project([[0,0,0],[8,0,0],[8,8,0],[0,8,0]],scene)[:,:2].round(4).tolist(),
            'image_size':[scene.size,scene.size],
            'description':'Board-plane corners: a8 outer, h8 outer, h1 outer, a1 outer. No piece metadata.'}


def _triangle(image, depth, triangle, color, origin=(0,0)):
    ox,oy = origin
    x0,y0 = np.maximum(np.floor(triangle[:,:2].min(0)).astype(int),origin)
    x1,y1 = np.minimum(np.ceil(triangle[:,:2].max(0)).astype(int),[ox+image.shape[1]-1,oy+image.shape[0]-1])
    if x1<x0 or y1<y0:
        return
    a,b,c = triangle
    den = (b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
    if abs(den)<1e-7:
        return
    yy,xx = np.mgrid[y0:y1+1,x0:x1+1]
    w0 = ((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den
    w1 = ((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den
    w2 = 1-w0-w1
    z = w0*a[2]+w1*b[2]+w2*c[2]
    d = depth[y0-oy:y1-oy+1,x0-ox:x1-ox+1]
    mask = (w0>=-1e-5)&(w1>=-1e-5)&(w2>=-1e-5)&(z>=d-1e-6)
    d[mask] = z[mask]
    image[y0-oy:y1-oy+1,x0-ox:x1-ox+1][mask] = color


@lru_cache(maxsize=8)
def _board(tilt, azimuth, size, palette):
    scene = Scene(tilt=tilt,azimuth=azimuth,size=size,palette=palette)
    image = np.full((scene.size,scene.size,3),[24,34,43],np.uint8)
    depth = np.full((scene.size,scene.size),-np.inf,np.float32)
    for row in range(8):
        for col in range(8):
            face = project([[col,row,0],[col+1,row,0],[col+1,row+1,0],[col,row+1,0]],scene)
            color = PALETTES[scene.palette][(row+col)%2]
            for i in (1,2):
                _triangle(image,depth,face[[0,i,i+1]],color)
    image.setflags(write=False); depth.setflags(write=False)
    return image,depth


@lru_cache(maxsize=96)
def _shaded_faces(symbol, rotation, light_level):
    light = np.array([-.4,-.6,1.0]); light /= np.linalg.norm(light)
    material = np.array([244,237,217] if symbol.isupper() else [44,56,72])
    result=[]
    for face in piece_faces(symbol,0,0,rotation):
        normal = np.cross(face[1]-face[0],face[2]-face[0])
        norm = np.linalg.norm(normal)
        if norm<1e-9: continue
        normal /= norm
        shade = .79+.21*max(0,float(normal@light))
        color = np.clip(material*shade*light_level,0,255).astype(np.uint8)
        result.append((face,color))
    return result


def _make_piece_layer(symbol,row,col,tilt,azimuth,rotation,light,size):
    scene=Scene(tilt=tilt,azimuth=azimuth,rotation=rotation,light=light,size=size)
    faces=[(project(face+[col+.5,row+.5,0],scene),color)
           for face,color in _shaded_faces(symbol,rotation,light)]
    points=np.concatenate([face[:,:2] for face,_ in faces])
    x0,y0=np.maximum(np.floor(points.min(0)).astype(int),0)
    x1,y1=np.minimum(np.ceil(points.max(0)).astype(int),size-1)
    image=np.zeros((y1-y0+1,x1-x0+1,3),np.uint8)
    depth=np.full(image.shape[:2],-np.inf,np.float32)
    for face,color in faces:
        for i in range(1,len(face)-1):
            _triangle(image,depth,face[[0,i,i+1]],color,(x0,y0))
    image.setflags(write=False); depth.setflags(write=False)
    return int(x0),int(y0),image,depth


# The default overhead layers are generated during the image build, not on a
# visitor's first request. They contain RGB/depth only; detection still sees PNGs.
LAYER_FILE=Path(__file__).resolve().parents[1]/'models/render_layers_fusion_v1.npz'


@lru_cache(maxsize=1)
def _default_layers():
    if not LAYER_FILE.exists(): return {}
    with np.load(LAYER_FILE,allow_pickle=False) as data:
        layers={}
        for symbol in 'PNBRQKpnbrqk':
            for row in range(8):
                for col in range(8):
                    key=f'{symbol}{row}{col}'
                    image,depth=data[key+'_rgb'],data[key+'_z']
                    image.setflags(write=False);depth.setflags(write=False)
                    x,y=data[key+'_xy']
                    layers[(symbol,row,col)]=(int(x),int(y),image,depth)
        return layers


@lru_cache(maxsize=256)
def _piece_layer(symbol,row,col,tilt,azimuth,rotation,light,size):
    if (tilt,azimuth,rotation,light,size)==(0,0,0,1,768):
        default=_default_layers().get((symbol,row,col))
        if default is not None: return default
    return _make_piece_layer(symbol,row,col,tilt,azimuth,rotation,light,size)


def render(matrix, scene=None):
    scene = (scene or Scene()).validate()
    board,board_depth=_board(scene.tilt,scene.azimuth,scene.size,scene.palette)
    image,depth=board.copy(),board_depth.copy()
    # Soft directional ground shadows from the projected solid base footprint.
    if scene.shadows:
        shadow = np.zeros(image.shape[:2],np.uint8)
        for row in range(8):
            for col in range(8):
                symbol = matrix[row][col]
                if symbol=='.': continue
                p = base_outline(symbol.upper())
                t = np.deg2rad(scene.rotation)
                p = p @ np.array([[np.cos(t),np.sin(t)],[-np.sin(t),np.cos(t)]])
                p = np.column_stack((p+[col+.59,row+.57],np.zeros(len(p))))
                cv2.fillPoly(shadow,[project(p,scene)[:,:2].astype(np.int32)],255)
        shadow = cv2.GaussianBlur(shadow,(9,9),2)/255
        image = (image.astype(float)*(1-shadow[:,:,None]*.13)).astype(np.uint8)
    for row in range(8):
        for col in range(8):
            symbol = matrix[row][col]
            if symbol=='.': continue
            x,y,rgb,z=_piece_layer(symbol,row,col,scene.tilt,scene.azimuth,
                                  scene.rotation,scene.light,scene.size)
            h,w=z.shape
            target_depth=depth[y:y+h,x:x+w]
            mask=np.isfinite(z)&(z>=target_depth-1e-6)
            target_depth[mask]=z[mask]
            image[y:y+h,x:x+w][mask]=rgb[mask]
    if scene.noise:
        image = np.clip(image.astype(float)+np.random.default_rng(scene.seed).normal(0,scene.noise,image.shape),0,255).astype(np.uint8)
    return image,calibration(scene)


def png_bytes(image):
    return cv2.imencode('.png',cv2.cvtColor(image,cv2.COLOR_RGB2BGR))[1].tobytes()


def warm_renderer():
    _default_layers()
    _board(0,0,768,'sage')


def build_layers():
    """Bake every default-view piece/square, including edits and promotions."""
    arrays={}
    for symbol in 'PNBRQKpnbrqk':
        for row in range(8):
            for col in range(8):
                x,y,rgb,z=_make_piece_layer(symbol,row,col,0,0,0,1,768)
                key=f'{symbol}{row}{col}'
                arrays[key+'_xy']=np.array([x,y],np.int32)
                arrays[key+'_rgb']=rgb; arrays[key+'_z']=z
    np.savez_compressed(LAYER_FILE,**arrays)
    print(f'Built {len(arrays)//3} rendered piece layers ({LAYER_FILE.stat().st_size} bytes)')


if __name__=='__main__': build_layers()
