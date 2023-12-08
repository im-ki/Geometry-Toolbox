import numpy as np
from scipy.io import loadmat
from toolbox.spherical_conformal_map import Spherical_conformal_map
from toolbox.mesh_utils import compute_curvature
from toolbox.vis import draw_surface


data = loadmat('../../step1/outer_mesh.mat')

f = data['face'] - 1
v = data['vert']

Umin, Umax, Cmin, Cmax, Cmean, Cgauss, Normal = compute_curvature(v, f)

C_sign = np.sign(Cmean)
C_abs = np.abs(Cmean)
C_sqrt = np.sqrt(C_abs)
C_sqrt = np.sqrt(C_sqrt)
colors = C_sqrt * C_sign
draw_surface(f, v, colors)

mapping = Spherical_conformal_map(v, f)
print(mapping[:4, :])

