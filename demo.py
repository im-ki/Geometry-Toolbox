import numpy as np
from scipy.io import loadmat
from toolbox.spherical_conformal_map import Spherical_conformal_map


data = loadmat('../step1/outer_mesh.mat')

f = data['face'] - 1
v = data['vert']

mapping = Spherical_conformal_map(v, f)
print(mapping[:4, :])

