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

if __name__ == '__main__':
    from scipy.io import loadmat
    data = loadmat('/mnt/c/Users/qgchen/Documents/workspace/PhD/3D_Beltrami/3DQCLR/test_3d_tet/step1/outer_mesh.mat')
    face = data['face'] - 1
    vert = data['vert']
    print(face.shape, vert.shape)
    
    normalv, normalf = compute_normal(vert, face)

    f = np.arange(vert.shape[0])
    new_f = perform_mesh_smoothing(face, vert, f)
    print(f[:10])
    print(new_f[:10, :])
