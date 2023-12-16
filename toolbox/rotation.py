import numpy as np

def r_x(vec):
    v1 = vec.copy().transpose()
    v1[0, 0] = 0
    norm_v1 = np.sqrt(np.sum(v1**2))

    cos = v1[0, 2] / norm_v1
    theta = np.arccos(cos)

    if vec[1] < 0:
        theta = -theta

    return np.array(((1, 0, 0),
                     (0, np.cos(theta), -np.sin(theta)),
                     (0, np.sin(theta), np.cos(theta))))

### Matrix that rotate around the y axis
def r_y(vec):
    v1 = vec.copy().transpose()
    v1[0, 1] = 0

    norm_v1 = np.sqrt(np.sum(v1**2))

    cos = v1[0, 2] / norm_v1
    theta = np.arccos(cos)

    if vec[0] > 0:
        theta = -theta

    return np.array(((np.cos(theta), 0, np.sin(theta)),
                     (0, 1, 0),
                     (-np.sin(theta), 0, np.cos(theta))))

### Rotation matrix
def rot_mat_to_top(vec):
    if vec.ndim == 1:
        vec = vec.reshape((-1, 1))
    mat_x = r_x(vec)
    inter_vec = mat_x.dot(vec)
    mat_y = r_y(inter_vec)
    return mat_y.dot(mat_x)

def rot_mat_to_bottom(vec):
    if vec.ndim == 1:
        vec = vec.reshape((-1, 1))
    up_down = rot_mat_to_top(np.array((0, 0, -1)).reshape((3, 1)))
    mat = rot_mat_to_top(vec)
    mat = up_down.dot(mat)
    return mat
