import numpy as np
import scipy.sparse as sps

def cotangent_laplacian(vert, face):
    Nv = vert.shape[0]

    i1, i2, i3 = face[:, 0], face[:, 1], face[:, 2]
    f1, f2, f3 = vert[i1, :], vert[i2, :], vert[i3, :]

    e1 = np.sqrt(np.sum((f2 - f3)**2, axis = 1))
    e2 = np.sqrt(np.sum((f3 - f1)**2, axis = 1))
    e3 = np.sqrt(np.sum((f1 - f2)**2, axis = 1))

    s = (e1 + e2 + e3) * 0.5
    area = np.sqrt(s * (s-e1) * (s-e2) * (s-e3))

    cot12 = (e1**2 + e2**2 - e3**2) / area / 2
    cot23 = (e2**2 + e3**2 - e1**2) / area / 2
    cot31 = (e1**2 + e3**2 - e2**2) / area / 2
    diag1 = -cot12-cot31
    diag2 = -cot12-cot23
    diag3 = -cot31-cot23

    II = np.concatenate((i1, i2, i2, i3, i3, i1, i1, i2, i3))
    JJ = np.concatenate((i2, i1, i3, i2, i1, i3, i1, i2, i3))
    V = np.concatenate((cot12, cot12, cot23, cot23, cot31, cot31, diag1, diag2, diag3))
    A = sps.csr_matrix((V, (II, JJ)), shape = (Nv, Nv))

    return A
