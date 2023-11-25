import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve

def generalized_laplacian2D(face, vertex, mu):#, h, w):
    """
    Inputs:
        face : m x 3 index of triangulation connectivity
        vertex : n x 2 vertices coordinates(x, y)
        mu : m x 1 Beltrami coefficients
        h, w: int
    Outputs:
        A : 2-dimensional generalized laplacian operator (h*w, h*w)
        abc : vectors containing the coefficients alpha, beta and gamma (m, 3)
        area : float, area of every triangles in the mesh
    """
    af = (1 - 2 * np.real(mu) + np.abs(mu)**2) / (1 - np.abs(mu)**2)
    bf = -2 * np.imag(mu) / (1 - np.abs(mu)**2)
    gf = (1 + 2 * np.real(mu) + np.abs(mu)**2) / (1 - np.abs(mu)**2)
    abc = np.hstack((af, bf, gf))

    f0, f1, f2 = face[:, 0, np.newaxis], face[:, 1, np.newaxis], face[:, 2, np.newaxis]

    uxv0 = vertex[f1,1] - vertex[f2,1]
    uyv0 = vertex[f2,0] - vertex[f1,0]
    uxv1 = vertex[f2,1] - vertex[f0,1]
    uyv1 = vertex[f0,0] - vertex[f2,0]
    uxv2 = vertex[f0,1] - vertex[f1,1]
    uyv2 = vertex[f1,0] - vertex[f0,0]

    #area = (1/(h-1)) * (1/(w-1)) / 2

    #l = [np.sqrt(np.sum(uxv0**2 + uyv0**2,2)), np.sqrt(sum(uxv1**2 + uyv1**2,2)), np.sqrt(sum(uxv2**2 + uyv2**2,2))];
    # test

    l = np.hstack([np.sqrt(uxv0**2 + uyv0**2), np.sqrt(uxv1**2 + uyv1**2), np.sqrt(uxv2**2 + uyv2**2)])
    assert l.shape[1] == 3
    s = np.sum(l, axis = 1) / 2
    area = np.sqrt(s * (s-l[:,0]) * (s-l[:,1]) * (s-l[:,2])).reshape((-1, 1))
    assert area.shape[0] == af.shape[0]

    v00 = (af * uxv0 * uxv0 + 2 * bf * uxv0 * uyv0 + gf * uyv0 * uyv0) / area;
    v11 = (af * uxv1 * uxv1 + 2 * bf * uxv1 * uyv1 + gf * uyv1 * uyv1) / area;
    v22 = (af * uxv2 * uxv2 + 2 * bf * uxv2 * uyv2 + gf * uyv2 * uyv2) / area;

    v01 = (af * uxv1 * uxv0 + bf * uxv1 * uyv0 + bf * uxv0 * uyv1 + gf * uyv1 * uyv0) / area;
    v12 = (af * uxv2 * uxv1 + bf * uxv2 * uyv1 + bf * uxv1 * uyv2 + gf * uyv2 * uyv1) / area;
    v20 = (af * uxv0 * uxv2 + bf * uxv0 * uyv2 + bf * uxv2 * uyv0 + gf * uyv0 * uyv2) / area;

    I = np.vstack((f0,f1,f2,f0,f1,f1,f2,f2,f0)).reshape(-1)
    J = np.vstack((f0,f1,f2,f1,f0,f2,f1,f0,f2)).reshape(-1)
    nRow = vertex.shape[0]
    V = np.vstack((v00,v11,v22,v01,v01,v12,v12,v20,v20)).reshape(-1) / 2
    A = sps.csr_matrix((-V, (I, J)), shape = (nRow, nRow))

    return A#, abc, area

def LBS(v, f, mu, lm_idx, landmarks):
    """
    Inputs:
        v : (Nv, 2)
        f : (Nf, 3)
        mu : (Nf,) complex
        lm_idx : (N_lm,)
    Outputs:
        mapping: (2, h, w)
        Ax, Ay: (h*w, h*w)
        bx, by: (h*w, 1)
    """
    # a = time.time()
    Ax = generalized_laplacian2D(f, v, mu)
    Ay = Ax.copy()
    bx = np.zeros((v.shape[0], 1))
    by = bx.copy()

    Nv = v.shape[0]

    #landmarkx = np.vstack((Edge4, Edge2))
    targetx = landmarks[:, 0].reshape((-1, 1))#   np.vstack((np.zeros_like(Edge4), np.ones_like(Edge2)))
    targety = landmarks[:, 1].reshape((-1, 1))    #np.vstack((np.zeros_like(Edge1), np.ones_like(Edge3)))

    lmx = lm_idx #landmarkx.reshape(-1)
    lmy = lm_idx #landmarkx.reshape(-1)

    bx[lmx] = targetx
    mrow, mcol = Ax[lmx, :].nonzero()
    mrow = lmx[mrow]
    mval = np.array(Ax[mrow, mcol]).reshape(-1)
    Ax_lm = sps.csr_matrix((mval, (mrow, mcol)), shape = (Nv, Nv))
    tmp = sps.csr_matrix((np.ones(lmx.shape[0]), (lmx, lmx)), shape = (Nv, Nv))
    Ax = Ax - Ax_lm + tmp
    mapx = spsolve(Ax, bx).reshape(-1)

    by[lmy] = targety
    mrow, mcol = Ay[lmy, :].nonzero()
    mrow = lmy[mrow]
    mval = np.array(Ay[mrow, mcol]).reshape(-1)
    Ay_lm = sps.csr_matrix((mval, (mrow, mcol)), shape = (Nv, Nv))
    tmp = sps.csr_matrix((np.ones(lmy.shape[0]), (lmy, lmy)), shape = (Nv, Nv))
    Ay = Ay - Ay_lm + tmp
    mapy = spsolve(Ay.tocsc(), by).reshape(-1)

    mapping = np.array((mapx, mapy)).T
    return mapping#, Ax, Ay, bx, by
