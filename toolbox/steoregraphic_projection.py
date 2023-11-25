import numpy as np

def steoregraphic_proj_inv(org_v):
    assert org_v.shape[1] == 2
    v = org_v.copy()
    #v[:, 1] = -v[:, 1]
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
