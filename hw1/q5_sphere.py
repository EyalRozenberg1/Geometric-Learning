import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pyvista as pv

u, v = np.mgrid[-16 * np.pi:16 * np.pi:2000j, -16 * np.pi:16*np.pi:2000j]

x = 2*u/(u**2+v**2+1)
y = 2*v/(u**2+v**2+1)
z = (u**2+v**2-1)/(u**2+v**2+1)

grid = pv.StructuredGrid(x, y, z)
grid.plot()
