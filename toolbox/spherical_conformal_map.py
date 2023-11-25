import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve

from .steoregraphic_projection import steoregraphic_proj_inv, steoregraphic_proj_south, steoregraphic_proj_south_inv
from .spherical_tutte_map import Spherical_tutte_map
from .compute_Beltrami_coef import beltrami_coefficient
from .lbs import LBS
from .mesh_utils import cotangent_laplacian

   
def Spherical_conformal_map(vert, face):
    
    # check whether the input mesh is genus-0
    if vert.shape[0] - 3 * face.shape[0] / 2 + face.shape[0] != 2:
        print("Error: The mesh is not a genus-0 closed surface.")
        assert(False)

    # Find the most regular triangle as the "big triangle"
    f1, f2, f3 = vert[face[:, 0], :], vert[face[:, 1], :], vert[face[:, 2], :]
    e1 = np.sqrt(np.sum((f2 - f3)**2, axis = 1))
    e2 = np.sqrt(np.sum((f3 - f1)**2, axis = 1))
    e3 = np.sqrt(np.sum((f1 - f2)**2, axis = 1))
    regularity = np.abs(e1/(e1+e2+e3)-1/3) + np.abs(e2/(e1+e2+e3)-1/3) + np.abs(e3/(e1+e2+e3)-1/3) 
    bigtri = np.argmin(regularity)
    ## In case the spherical parameterization result is not evenly distributed, try to change bigtri to the id of some other triangles with good quality

    # North pole step: Compute spherical map by solving laplace equation on a big triangle
    Nv = vert.shape[0]
    M = cotangent_laplacian(vert, face)

    p1, p2, p3 = face[bigtri, :]
    fixed = np.array([p1, p2, p3])

    mrow, mcol = M[fixed, :].nonzero()
    mrow = fixed[mrow]
    mval = np.asarray(M[mrow, mcol]).reshape(-1)
    M_fixed = sps.csr_matrix((mval, (mrow, mcol)), shape = (Nv, Nv))
    tmp = sps.csr_matrix((np.ones(3), (fixed, fixed)), shape = (Nv, Nv))
    M = M - M_fixed + tmp
    #M.data[fixed] = tmp.data
    #M.rows[fixed] = tmp.rows

    # set the boundary condition for big triangle
    x1, y1, x2, y2 = 0, 0, 1, 0 # arbitrarily set the two points
    a = vert[p2, :] - vert[p1, :]
    b = vert[p3, :] - vert[p1, :]
    sin1 = np.linalg.norm(np.cross(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    ori_h = np.linalg.norm(b)*sin1
    ratio = np.linalg.norm((x1-x2, y1-y2)) / np.linalg.norm(a)
    y3 = ori_h * ratio
    x3 = np.sqrt(np.linalg.norm(b)**2 * ratio**2 - y3**2)

    # Solve the Laplace equation to obtain a harmonic map
    c, d = np.zeros((Nv)), np.zeros((Nv))
    c[p1], d[p1]= x1, y1
    c[p2], d[p2]= x2, y2
    c[p3], d[p3]= x3, y3
    c, d = c.reshape((-1, 1)), d.reshape((-1, 1))

    zr = spsolve(M, c)
    zi = spsolve(M, d)

    zr = zr - np.mean(zr)
    zi = zi - np.mean(zi)
    z = np.hstack((zr.reshape((-1, 1)), zi.reshape((-1, 1))))

    # inverse stereographic projection
    S = steoregraphic_proj_inv(z)

    # Find optimal big triangle size
    w = steoregraphic_proj_south(S)

    # find the index of the southernmost triangle
    z_norm = np.linalg.norm(z, axis = 1)
    sum_norm = z_norm[face[:, 0]] + z_norm[face[:, 1]] + z_norm[face[:, 2]]
    index = np.argsort(sum_norm)
    inner = index[0]

    if inner == bigtri:
        inner = index[1]

    # Compute the size of the northern most and the southern most triangles
    NorthTriSide = (np.linalg.norm(z[face[bigtri, 0]] - z[face[bigtri, 1]]) + np.linalg.norm(z[face[bigtri, 1]] - z[face[bigtri, 2]]) + np.linalg.norm(z[face[bigtri, 2]] - z[face[bigtri, 0]])) / 3
    SouthTriSide = (np.linalg.norm(w[face[inner, 0]] - w[face[inner, 1]]) + np.linalg.norm(w[face[inner, 1]] - w[face[inner, 2]]) + np.linalg.norm(w[face[inner, 2]] - w[face[inner, 0]])) / 3

    # rescale to get the best distribution
    z = z * np.sqrt(NorthTriSide * SouthTriSide) / NorthTriSide

    # inverse stereographic projection
    S = steoregraphic_proj_inv(z)

    if np.sum(np.isnan(S)) != 0:
    # if harmonic map fails due to very bad triangulations, use tutte map
        S = Spherical_tutte_map(face, bigtri)

    # South pole step
    I = np.argsort(S[:, 2])

    fixnum = np.max((np.round(vert.shape[0] / 10).astype(np.int64), 3))
    fixed = I[:np.min((vert.shape[0], fixnum))]

    P = steoregraphic_proj_south(S)

    # compute the Beltrami coefficient
    mu = beltrami_coefficient(P, face, vert)

    # compose the map with another quasi-conformal map to cancel the distortion
    mapping = LBS(P, face, mu, fixed, P[fixed, :]) 

    if np.sum(np.isnan(mapping)) != 0:
        # if the result has NaN entries, then most probably the number of
        # boundary constraints is not large enough
    
        # increase the number of boundary constraints and run again
        fixnum = fixnum * 5  # again, this number can be changed
        fixed = I[:np.min((vert.shape[0], fixnum))]
        mapping = LBS(P, face, mu, fixed, P[fixed, :]) 
    
        if np.sum(np.isnan(mapping)) != 0:
            mapping = P  # use the old result

    mapping = steoregraphic_proj_south_inv(mapping)
    return mapping


