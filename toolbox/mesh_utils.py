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

def triangulation2adjacency(f, v = None):
    assert(f.shape[1] == 3)
    if v is not None:
        assert(v.shape[1] == 3 or v.shape[1] == 2)
        Nv = v.shape[0]
    else:
        Nv = np.max(f) + 1

    rows = np.concatenate([f[:, 0], f[:, 0], f[:, 1], f[:, 1], f[:, 2], f[:, 2]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 2], f[:, 0], f[:, 1]])
    data = np.ones_like(rows)

    A = sps.csr_matrix((data, (rows, cols)), shape=(Nv, Nv))
    A = (A > 0).astype(float)

    return A

def calculate_vertex_face_incidence_matrix(face):
    import scipy.sparse as sps
    Nv = np.max(face) + 1
    Nf = face.shape[0]

    row_indices = face.flatten()
    col_indices = np.repeat(np.arange(Nf), 3)
    data = np.ones_like(row_indices)

    M = sps.csr_matrix((data, (row_indices, col_indices)), shape=(Nv, Nf))

    return M 

def compute_normal(vertex, face):
    assert(face.shape[1] == 3)
    assert(vertex.shape[1] == 3)
    nface = face.shape[0]
    nvert = vertex.shape[0]
    normal = np.zeros((nvert, 3))

    # unit normals to the faces
    e1, e2 = vertex[face[:, 1], :] - vertex[face[:, 0], :], vertex[face[:, 2], :] - vertex[face[:, 0], :]
    normalf = np.cross(e1, e2)
    d = np.linalg.norm(normalf, axis=1)
    d[d < np.finfo(float).eps] = 1
    normalf = normalf / d[:, np.newaxis]

    # unit normal to the vertex
    #normal = np.zeros((nvert, 3))
    #for i in range(nface):
    #    f = face[i, :]
    #    for j in range(3):
    #        normal[f[j], :] = normal[f[j], :] + normalf[i, :]

    M = calculate_vertex_face_incidence_matrix(face)
    normal = M.dot(normalf) 
    #print(np.sum(np.abs(normal - normal2)))

    # normalize
    d = np.linalg.norm(normal, axis=1)
    d[d < np.finfo(float).eps] = 1
    normal = normal / d[:, np.newaxis]

    # enforce that the normal are outward
    v = vertex - np.mean(vertex, axis = 0).reshape((1, 3))

    s = np.sum(v * normal, axis=1)
    if np.sum(s > 0) < np.sum(s < 0):
        # flip
        normal = -normal
        normalf = -normalf

    return normal, normalf


def perform_mesh_smoothing(face, vertex, f, type = 'combinatorial'):
    from scipy.sparse import spdiags
    if f.ndim == 1:
        f = f.reshape((-1, 1))

    N_average = 1

    Nv = np.max(face) + 1

    # compute normalized averaging matrix
    if type == 'combinatorial':
        # add diagonal
        W = triangulation2adjacency(face) + spdiags(np.ones(Nv), 0, Nv, Nv)
        D = spdiags(np.reciprocal(np.sum(W, axis=1)).reshape(-1), 0, Nv, Nv)
        W = D.dot(W)
    #else:
    #    options = {'normalize': 1}
    #    W = compute_mesh_weight(vertex, face, type, options)

    # do averaging to smooth the field
    for k in range(N_average):
        f = W.dot(f)

    return f

def compute_curvature(vert, face):
    """
    Umin is the direction of minimum curvature
    Umax is the direction of maximum curvature
    Cmin is the minimum curvature
    Cmax is the maximum curvature
    Cmean=(Cmin+Cmax)/2
    Cgauss=Cmin*Cmax
    Normal is the normal to the surface

    This is a reimplementation of the curvature computation algorithm described in
        David Cohen-Steiner and Jean-Marie Morvan.
        Restricted Delaunay triangulations and normal cycle.
        In Proc. 19th Annual ACM Symposium on Computational Geometry,
        pages 237-246, 2003.
    and also in
        Pierre Alliez, David Cohen-Steiner, Olivier Devillers, Bruno Le�vy, and Mathieu Desbrun.
        Anisotropic Polygonal Remeshing.
        ACM Transactions on Graphics, 2003.
        Note: SIGGRAPH '2003 Conference Proceedings
    """
    assert(vert.shape[1] == 3 and face.shape[1] == 3)
    orient = 1
    Nf, Nv = face.shape[0], vert.shape[0]
    face = face.astype(np.int64)

    bit = len(str(Nv))
    box = 10**bit

    I = np.concatenate((face[:, 0], face[:, 1], face[:, 2]))
    J = np.concatenate((face[:, 1], face[:, 2], face[:, 0]))
    S = np.concatenate((np.arange(1, Nf+1), np.arange(1, Nf+1), np.arange(1, Nf+1)))

    # associate each edge to a pair of faces
    mask = (I < J)
    mask_inv = np.logical_not(mask)
    I1, J1, S1 = I[mask], J[mask], S[mask]
    I2, J2, S2 = J[mask_inv], I[mask_inv], S[mask_inv] # Here we reverse the order of I and J to ensure that I2 < J2.

    set1, set2 = I1*box+J1, I2*box+J2
    IJ, x_ind, y_ind = np.intersect1d(set1, set2, return_indices=True)
    S1, S2 = S1[x_ind], S2[y_ind]

    J = IJ % box
    I = (IJ - J) // box
    S = np.vstack((S1, S2)).T - 1
    Ne = I.shape[0]

    # Normalize edge
    edge = vert[J, :] - vert[I, :]
    d = np.linalg.norm(edge, axis = 1)
    edge = edge / d.reshape((-1, 1))
    # avoid too large numerics
    d = d / np.mean(d)

    # normals to vertices and faces
    normalv, normal = compute_normal(vert, face)
    # inner product of normals
    dp = np.sum(normal[S[:, 0], :] * normal[S[:, 1], :], axis = 1)
    # angle un-signed
    beta = np.arccos(np.clip(dp, -1, 1))
    # compute the sign
    cp = np.cross(normal[S[:, 0], :], normal[S[:, 1], :])
    si = orient * np.sign(np.sum(cp * edge, axis = 1))
    # angle signed
    beta = beta * si

    # tensors
    T = np.zeros((Ne, 3, 3))
    # outer product, since it is a simmetric rank-1 matrix, we only need to compute part of them
    for x in range(3):
        for y in range(x+1):
            T[:, x, y] = edge[:, x] * edge[:, y]
            T[:, y, x] = T[:, x, y]
    T = T * np.reshape(d*beta, (-1, 1, 1)) # apply weights

    # do pooling on vertices, w record the number of edges incident to the vertices
    Tv = np.zeros((Nv, 3, 3))
    w = np.zeros(Nv)
    for i in range(Ne):
        Tv[I[i], :, :] = Tv[I[i], :, :] + T[i, :, :]
        Tv[J[i], :, :] = Tv[J[i], :, :] + T[i, :, :]
        w[I[i]] = w[I[i]] + 1
        w[J[i]] = w[J[i]] + 1
    w[w < np.finfo(float).eps] = 1
    Tv = Tv / w.reshape((-1, 1, 1))

    # do averaging to smooth the field
    for x in range(3):
        for y in range(3):
            Tv[:, x, y] = perform_mesh_smoothing(face, vert, Tv[:, x, y]).reshape(-1)

    # extract eigenvectors and eigenvalues
    U = np.zeros((Nv, 3, 3))
    D = np.zeros((Nv, 3))
    val, vec = np.linalg.eig(Tv)
    ind = np.argsort(np.abs(val), axis = 1)
    D = np.take_along_axis(val, ind, axis = 1)
    ind = np.expand_dims(ind, axis = 1)
    U = np.take_along_axis(vec, ind, axis = 2)

    Normal = U[:, :, 0]
    # According to the paper "Anisotropic Polygonal Remeshing", the associated directions of Cmin and Cmax are switched.
    Umin = U[:, :, 2]
    Umax = U[:, :, 1]
    Cmin = D[:, 1]
    Cmax = D[:, 2]
    Cmean = (Cmin+Cmax) / 2
    Cgauss = (Cmin * Cmax)

    # Try to re-orient the normals
    s = np.sign(np.sum(Normal * normalv, axis = 1))
    Normal = Normal * s.reshape((-1, 1))
    return Umin, Umax, Cmin, Cmax, Cmean, Cgauss, Normal


