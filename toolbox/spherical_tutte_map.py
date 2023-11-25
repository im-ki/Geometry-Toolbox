import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve

from .steoregraphic_projection import steoregraphic_proj_inv, steoregraphic_proj_south

def Spherical_tutte_map(face, bigtri = None):

    if bigtri is None:
        bigtri = 0

    Nv = np.max(face)+1
    Nf = face.shape[0]

    I = face.reshape(-1)
    J = face[:, (1, 2, 0)].reshape(-1)
    V = np.ones(Nf*3) / 2
    II = np.concatenate((I, J))
    JJ = np.concatenate((J, I))
    VV = np.concatenate((V, V))
    W = sps.csr_matrix((VV, (II, JJ)), shape = (Nv, Nv))
    diag = - W.diagonal() - np.sum(W, axis=1).reshape(-1)
    diag = np.asarray(diag).reshape(-1)
    diag = sps.csr_matrix((diag, (np.arange(Nv), np.arange(Nv))), shape = (Nv, Nv))

    M = W + diag

    boundary = face[bigtri, :]

    mrow, mcol = M[boundary, :].nonzero()
    mrow = boundary[mrow]
    mval = np.asarray(M[mrow, mcol]).reshape(-1)
    M_boundary = sps.csr_matrix((mval, (mrow, mcol)), shape = (Nv, Nv))
    tmp = sps.csr_matrix((np.ones(boundary.shape[0]), (boundary, boundary)), shape = (Nv, Nv))
    M = M - M_boundary + tmp

    target = np.exp(1j * (2*np.pi*(np.array((0, 1, 2))/3)))
    bx, by = np.zeros((Nv, 1)), np.zeros((Nv, 1))
    bx[boundary, 0] = target.real
    by[boundary, 0] = target.imag

    zr = spsolve(M, bx)
    zi = spsolve(M, by)

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

    return S
