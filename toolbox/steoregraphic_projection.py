import numpy as np


def steoregraphic_proj(v, keep_direction = False):
    # keep_direction: North pole steoregraphic projection will change the direction of the triangles. To correct the direction, set this flag.
    x = v[:, 0] / (1-v[:, 2])
    y = v[:, 1] / (1-v[:, 2])
    if keep_direction:
        y = -y # To preserve the vertices direction of the faces
    return np.stack((x, y)).T

def steoregraphic_proj_inv(org_v, keep_direction = False):
    # keep_direction: North pole steoregraphic projection will change the direction of the triangles. To correct the direction, set this flag.
    assert org_v.shape[1] == 2
    v = org_v.copy()
    if keep_direction:
        v[:, 1] = -v[:, 1]
    x = (2*v[:,0]) / (1 + v[:,0]**2 + v[:,1]**2)
    y = (2*v[:,1]) / (1 + v[:,0]**2 + v[:,1]**2)
    z = (-1 + v[:,0]**2 + v[:,1]**2) / (1 + v[:,0]**2 + v[:,1]**2)
    return np.stack((x, y, z)).T

def steoregraphic_proj_south(v):
    x = v[:, 0] / (1+v[:, 2])
    y = v[:, 1] / (1+v[:, 2])
    return np.stack((x, y)).T

def steoregraphic_proj_south_inv(org_v):
    assert org_v.shape[1] == 2
    v = org_v.copy()
    v[:, 1] = -v[:, 1]
    x = (2*v[:,0]) / (1 + v[:,0]**2 + v[:,1]**2)
    y = (-2*v[:,1]) / (1 + v[:,0]**2 + v[:,1]**2)
    z = -(-1 + v[:,0]**2 + v[:,1]**2) / (1 + v[:,0]**2 + v[:,1]**2)
    return np.stack((x, y, z)).T
