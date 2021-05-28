from pathlib import Path
root = Path(Path(__file__).resolve().parents[1])
from utils.mesh3 import Mesh

import matplotlib
matplotlib.use('TkAgg')
import os
import glob

file_type = 'ply'  # file type in ply, off

items = [14]
for item in items:
    if file_type == 'off':
        data_dir = os.path.join(root, 'data', 'off_files')
        file = glob.glob(os.path.join(data_dir, '*.off'))[item]
    else:
        data_dir = os.path.join(root, 'data', 'MPI-FAUST', 'training', 'registrations')
        file = glob.glob(os.path.join(data_dir, '*.ply'))[item]
        print("ply file: ", file)

    # define mesh type
    mesh = Mesh(file, file_type=file_type, cls='half_cotangent')  # half_cotangent ,uniform
    Q = 3  # select question

    if Q==3:
        mesh.render_mesh(k=5, scalar_func='eig')
    elif Q==4:
        mesh.render_mesh(scalar_func='mean_curvature')
    elif Q==5:
        a = True
        if a:
            mesh.render_mesh(scalar_func='scalar_funcs')
        else:
            mesh.render_mesh(scalar_func='area_normalized_laplacian')
    elif Q==6:
        mesh.render_mesh(scalar_func='hks')  # first part in question 6
