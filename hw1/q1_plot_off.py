from utils.io import read_off
from pathlib import Path

root = Path(Path(__file__).resolve().parents[1])

import matplotlib
matplotlib.use('TkAgg')

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

item = 7
data_dir = os.path.join(root, 'data', 'off_files')
file = glob.glob(os.path.join(data_dir, '*.off'))[item]
print("off file: ", file)

data = read_off(file)
vs = data[0]

x_coor = np.array([v[0] for v in vs])
y_coor = np.array([v[1] for v in vs])
z_coor = np.array([v[2] for v in vs])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_coor, y_coor, z_coor)
plt.show()
