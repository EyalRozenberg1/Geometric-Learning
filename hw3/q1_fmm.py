import os
import glob
from pathlib import Path
root = Path(Path(__file__).resolve().parents[1])
from numpy.linalg import norm
import numpy as np
import eikonalfm
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

# load maze.png
file = os.path.join(root, 'data', 'HW3_Resources', 'maze.png')
image = Image.open(file)
# convert image to numpy array - setting walls value to 100k and path to 1
c = np.ceil(asarray(image).mean(2)/255)
image_float = c
c[c == 0] = 100e3
# c = c[300:450, 750:900]
# Calculate the FMM distances inside the maze from source point
source_point = (383, 814)
dx = (1.0, 1.0)
order = 2
T_fmm = eikonalfm.fast_marching(c, source_point, dx, order)
plt.imshow(T_fmm, cmap='jet')
# plt.imshow(image)
plt.title(f'distances from source-point: {source_point}')
plt.show()

target_point = (1, 1)
#  finite differences approximation
Tx, Ty = np.gradient(T_fmm)
grad = np.concatenate((Tx[:, :, None], Ty[:, :, None]), 2)
grad[grad != 0] = grad[grad != 0] / np.tile(norm(grad, axis=2, keepdims=True),(1,1,2))[grad!=0]
Tx = grad[:, :, 0]
Ty = grad[:, :, 1]

X = np.arange(0, T_fmm.shape[1])
Y = np.arange(0, T_fmm.shape[0])

path = np.array([target_point])
iter = 0
rate = 1
while ((path[-1] - np.array(source_point))**2).sum() > 1e-4 and iter < 100:
    last_point = path[-1]
    x = last_point[0]
    y = last_point[1]

    dx = interp2d(X, Y, Tx)(x, y)[0]
    dy = interp2d(X, Y, Ty)(x, y)[0]

    path = np.append(path, [last_point - rate * np.array([dx, dy])], axis=0)
    iter += 1

print(path)