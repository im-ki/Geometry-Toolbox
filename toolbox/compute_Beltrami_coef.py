import numpy as np
import scipy.sparse as sps

def beltrami_coefficient(v, f, mapping):
    # f : nf * 3 matrix
    # v : nv * 2 matrix
    # mapping : nv * 3 matrix
    # (u, v) -> (X, Y, Z)


    nf = f.shape[0]
    nv = v.shape[0]
    Mi = np.stack((np.arange(nf), np.arange(nf), np.arange(nf))).T.reshape(-1)
    Mj = f.reshape(-1)

    e1 = v[f[:, 2], :] - v[f[:, 1], :]
    e2 = v[f[:, 0], :] - v[f[:, 2], :]
    e3 = v[f[:, 1], :] - v[f[:, 0], :]

    area = (-e2[:,0]*e1[:,1] + e1[:,0]*e2[:,1]) / 2
    area = np.stack((area, area, area))

    Mx = np.stack((e1[:, 1], e2[:, 1], e3[:, 1])) / area / 2
    Mx = Mx.T.reshape(-1)
    My = np.stack((e1[:, 0], e2[:, 0], e3[:, 0])) / area / 2
    My = -My.T.reshape(-1)

    Dx = sps.csr_matrix((Mx, (Mi, Mj)), shape=(nf, nv)) # d/du
    Dy = sps.csr_matrix((My, (Mi, Mj)), shape=(nf, nv)) # d/dv

    dXdu = Dx.dot(mapping[:, 0])
    dXdv = Dy.dot(mapping[:, 0])
    dYdu = Dx.dot(mapping[:, 1])
    dYdv = Dy.dot(mapping[:, 1])
    dZdu = Dx.dot(mapping[:, 2])
    dZdv = Dy.dot(mapping[:, 2])

    E = dXdu**2 + dYdu**2 + dZdu**2 # The square of the norm of d(mapping)/du 
    G = dXdv**2 + dYdv**2 + dZdv**2 # The square of the norm of d(mapping)/dv
    F = dXdu*dXdv + dYdu*dYdv + dZdu*dZdv # The dot product of the d(mapping)/du and d(mapping)/dv
    mu = (E - G + 2 * 1j * F) / (E + G + 2*np.sqrt(E*G - F**2)) # np.sqrt(E*G - F**2) represent the cross product of d(mapping)/du and d(mapping)/dv, representing the change of area

    return mu.reshape((-1, 1))
