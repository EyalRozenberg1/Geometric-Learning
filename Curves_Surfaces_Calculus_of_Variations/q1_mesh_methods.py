import matplotlib
matplotlib.use('TkAgg')
from utils.mesh import Mesh
import os
import glob
from pathlib import Path
root = Path(Path(__file__).resolve().parents[1])

item = 7
data_dir = os.path.join(root, 'data', 'off_files')
file = glob.glob(os.path.join(data_dir, '*.off'))[item]

mesh = Mesh(file)
demo = 'pointcloud'

if demo == 'wireframe':
    mesh.render_wireframe()

if demo == 'pointcloud':
    scalar_func = 'coo'
    if scalar_func == 'degree':
        mesh.render_pointcloud(scalar_func=scalar_func)
    if scalar_func == 'coo':
        mesh.render_pointcloud(scalar_func=scalar_func)

if demo == 'surface':
    scalar_func = 'coo'
    if scalar_func == 'coo':
        mesh.render_surface(scalar_func='coo')
    if scalar_func == 'inds':
        mesh.render_surface(scalar_func='inds')

if demo == 'normals':
    type = 'vertices'
    if type == 'vertices':
        # mesh.render_vertices_normals(normalize=True, mag=0.05)
        # mesh.render_vertices_normals(normalize=False, mag=5e4)
        mesh.render_vertices_normals(normalize=False, mag=5e4, add_norms=True)
    if type == 'faces':
        mesh.render_get_faces_normals(normalize=True, mag=0.05)
        mesh.render_get_faces_normals(normalize=False, mag=1e5)
        # mesh.render_get_faces_normals(normalize=False, mag=1e5, add_norms=True)


if demo == 'curvature':
    mesh.render_surface(scalar_func='curvature')

if demo == 'face_area':
    mesh.render_surface(scalar_func='face_area')
if demo == 'vertex_area':
    mesh.render_surface(scalar_func='vertex_area')
if demo == 'centroid':
    mesh.render_distance_from_centroid()


