"""Solid meshes in square-width units. Z is up; all grippers are identical.

Faces are closed polygonal surfaces, including filled base tops and bottoms.
No labels, fiducials, or piece-specific changes to the gripping cube.
"""
from functools import lru_cache
import numpy as np

BASE_HEIGHT = 5 / 50
CUBE_WIDTH = 0.30
CUBE_BOTTOM = BASE_HEIGHT
CUBE_TOP = BASE_HEIGHT + CUBE_WIDTH


def polygon(n, radius, phase=0):
    angles = np.arange(n) * 2 * np.pi / n + phase
    return np.column_stack((np.cos(angles), np.sin(angles))) * radius


def prism(points, bottom, top, scale_top=1.0):
    p = np.asarray(points, dtype=float)
    low = np.column_stack((p, np.full(len(p), bottom)))
    high = np.column_stack((p * scale_top, np.full(len(p), top)))
    # Triangulate horizontal caps as fans: bases are star-convex about origin.
    faces = []
    for i in range(len(p)):
        j = (i + 1) % len(p)
        faces.extend([np.array([[0, 0, top], high[i], high[j]]),
                      np.array([[0, 0, bottom], low[j], low[i]]),
                      np.array([low[i], low[j], high[j], high[i]])])
    return faces


def box(width, depth, bottom, top, x=0, y=0):
    faces = prism([[-width/2, -depth/2], [width/2, -depth/2],
                   [width/2, depth/2], [-width/2, depth/2]], bottom, top)
    return [f + [x, y, 0] for f in faces]


def sphere(radius, center, rings=8, segments=16):
    faces = []
    for i in range(rings):
        a, b = -np.pi/2 + np.pi*i/rings, -np.pi/2 + np.pi*(i+1)/rings
        for j in range(segments):
            c, d = 2*np.pi*j/segments, 2*np.pi*(j+1)/segments
            faces.append(np.array([[np.cos(t)*np.cos(s), np.cos(t)*np.sin(s), np.sin(t)]
                                  for t, s in [(a,c),(a,d),(b,d),(b,c)]]) * radius + center)
    return faces


def base_outline(kind):
    """Fusion STEP outer contours, centered on the grip, in 50 mm squares."""
    if kind == 'P':
        return polygon(48, 15 / 50)
    if kind == 'R':
        return np.array([[-15,-15],[15,-15],[15,15],[-15,15]]) / 50
    if kind == 'N':
        return polygon(3, 25 / 50, np.pi / 2)
    if kind == 'B':
        return polygon(5, 15 / (50 * np.cos(np.pi / 5)), np.pi / 2)
    if kind == 'Q':
        points = polygon(10, 1, np.pi / 2)
        radii = np.tile([25, 10 / np.cos(np.pi / 5)], 5)
        return points * radii[:, None] / 50
    if kind == 'K':
        return np.array([[-10,-20],[10,-20],[10,-10],[20,-10],[20,10],
                         [10,10],[10,20],[-10,20],[-10,10],[-20,10],
                         [-20,-10],[-10,-10]]) / 50
    raise ValueError(kind)


@lru_cache(maxsize=6)
def sections(kind):
    """Return three distinct solid sections; the cube is independent of kind."""
    base = prism(base_outline(kind), 0, BASE_HEIGHT)
    cube = box(CUBE_WIDTH, CUBE_WIDTH, CUBE_BOTTOM, CUBE_TOP)
    z = CUBE_TOP
    if kind == 'P':
        top = prism(polygon(24,.16), z, z+.19, .50) + sphere(.105, [0,0,z+.265])
    elif kind == 'R':
        top = box(.22,.22,z,z+.22) + box(.32,.32,z+.19,z+.25)
        for x,y in [(-.115,-.115),(.115,-.115),(-.115,.115),(.115,.115)]:
            top += box(.09,.09,z+.25,z+.34,x,y)
    elif kind == 'N':
        # Asymmetric horse profile extruded along X, with muzzle and ear.
        profile = np.array([[-.12,z], [.12,z], [.08,z+.15], [.23,z+.23],
                            [.19,z+.33], [.02,z+.38], [-.10,z+.29]])
        left = np.column_stack((np.full(len(profile),-.09),profile))
        right = np.column_stack((np.full(len(profile),.09),profile))
        top = [left[::-1],right]
        top += [np.array([left[i],left[(i+1)%len(left)],right[(i+1)%len(left)],right[i]]) for i in range(len(left))]
        top += box(.09,.07,z+.33,z+.44,0,-.03)
    elif kind == 'B':
        top = prism(polygon(24,.15),z,z+.25,.32)
        top += prism(polygon(8,.105),z+.24,z+.37,.05)
    elif kind == 'Q':
        top = prism(polygon(24,.13),z,z+.22,.65)
        top += prism(polygon(24,.11),z+.22,z+.29,1.65)
        for x,y in polygon(5,.15,-np.pi/2):
            top += sphere(.047,[x,y,z+.31],4,8)
    else:
        top = prism(polygon(24,.145),z,z+.26,.65)
        top += prism(polygon(16,.16),z+.24,z+.29)
        top += box(.07,.07,z+.29,z+.47) + box(.22,.07,z+.38,z+.44)
    return base,cube,top


def piece_faces(kind, x, y, rotation=0):
    t = np.deg2rad(rotation)
    r = np.array([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1]])
    return [face @ r.T + [x,y,0] for section in sections(kind.upper()) for face in section]
