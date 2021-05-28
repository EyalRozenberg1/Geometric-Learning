from pathlib import Path
root = Path(Path(__file__).resolve().parents[1])
from utils.mesh3 import Mesh
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')
import os
import glob
import numpy as np

result_dir = os.path.join(root, 'figs', 'Ex3', 'Q6', 'mds')

def spectral_decomp(data,n_dim):
    """
    Spectral decomposition
    :param data: data
    :param n_dim: SVD for n_dim
    :return: low dimensionality data with SVD decomposition
    """

    G_eigenValues, G_eigenVectors = np.linalg.eigh(data)
    idx = G_eigenValues.argsort()
    G_eigenValues = G_eigenValues[idx]
    G_eigenVectors = G_eigenVectors[:, idx]
    G_eigenValues = G_eigenValues[-n_dim:]
    G_eigenVectors = G_eigenVectors[:, -n_dim:]
    return G_eigenVectors @ np.diag(G_eigenValues ** 0.5)

def mds(data,n_dim):
    """
    Multidimentional scaling - reducing the dimentionality to n_dim
    :param data: data which we would like to reduce the dimentionality
    :param n_dim: what is the dimentsion we would like to embed
    :return: Reduce dimention with MDS algo.
    """

    row, n_feature = data.shape

    H = np.identity(row) - (1 / row) * np.ones_like(data)
    K = -0.5 * (H @ (data**2) @ H)
    return spectral_decomp(K, n_dim)

def plot_mds(mds, names, path):
    num_descriptors = len(mds)
    _, axs = plt.subplots(2, num_descriptors)
    for j, desc_mds in enumerate(mds):
        num_objects = desc_mds.shape[0]
        posses_vector = np.array([(i % 10)**2 for i in range(num_objects)])
        object_vector = np.array([(i // 10)**2 for i in range(num_objects)])
        for i in range(2):
            if i == 0:
                color_vec = object_vector
                title = f"{names[j]} subjects"
            else:
                color_vec = posses_vector
                title = f"{names[j]} posses"
            axs[i, j].scatter(desc_mds[:, 0], desc_mds[:, 1], c=color_vec, s=10, cmap='rainbow')
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].set_title(title, fontsize=10)
    plt.savefig(path)

file_type = 'ply'

items = range(100)
gps = np.zeros((100, 6890, 999))  # i,d,k: i-number of shapes, d-number of vertices, k-number of eigenvalues
dna = np.zeros((100, 1000))
mcurve = np.zeros((100, 6890))
hks5 = np.zeros((100, 10, 6890))
hks50 = np.zeros((100, 10, 6890))
hks200 = np.zeros((100, 10, 6890))
hks1000 = np.zeros((100, 10, 6890))
shks5 = np.zeros((100, 10, 6890))
shks50 = np.zeros((100, 10, 6890))
shks200 = np.zeros((100, 10, 6890))
shks1000 = np.zeros((100, 10, 6890))
# hks5 = np.zeros((100, 6890))
# hks50 = np.zeros((100, 6890))
# hks200 = np.zeros((100, 6890))
# hks1000 = np.zeros((100, 6890))


for item in items:
    print(f'item number {item}')
    data_dir = os.path.join(root, 'data', 'MPI-FAUST', 'training', 'registrations')
    file = glob.glob(os.path.join(data_dir, '*.ply'))[item]
    print("ply file: ", file)

    mesh = Mesh(file, file_type='ply', cls='uniform')  # half_cotangent ,uniform
    gps[item] = mesh.gps_desc()
    dna[item] = mesh.dna_desc()
    hks5[item], hks50[item], hks200[item], hks1000[item] = mesh.hks_desc(start=1e-3, stop=5, n=10)  #  half_cotangent: t=10, uniform: t=5,
    shks5[item], shks50[item], shks200[item], shks1000[item] = mesh.shks_desc(start=1e-3, stop=5, n=10)  #  half_cotangent: t=10, uniform: t=5,
    # hks5[item], hks50[item], hks200[item], hks1000[item] = mesh.hks_desc2(t=10)  #  half_cotangent: t=10, uniform: t=5,
    mcurve[item] = mesh.mcurv_desc

np.save(os.path.join(result_dir, "gps.npy"), gps)
np.save(os.path.join(result_dir, "dna.npy"), dna)
np.save(os.path.join(result_dir, "hks5.npy"), hks5)
np.save(os.path.join(result_dir, "hks50.npy"), hks50)
np.save(os.path.join(result_dir, "hks200.npy"), hks200)
np.save(os.path.join(result_dir, "hks1000.npy"), hks1000)
np.save(os.path.join(result_dir, "shks5.npy"), shks5)
np.save(os.path.join(result_dir, "shks50.npy"), shks50)
np.save(os.path.join(result_dir, "shks200.npy"), shks200)
np.save(os.path.join(result_dir, "shks1000.npy"), shks1000)
np.save(os.path.join(result_dir, "mcurve.npy"), mcurve)

D_gps5 = cdist(gps[:, :, :4].reshape(100, -1), gps[:, :, :4].reshape(100, -1))
D_gps50 = cdist(gps[:, :, :49].reshape(100, -1), gps[:, :, :49].reshape(100, -1))
D_gps200 = cdist(gps[:, :, :199].reshape(100, -1), gps[:, :, :199].reshape(100, -1))
D_gps1000 = cdist(gps.reshape(100, -1), gps.reshape(100, -1))

D_dna5 = cdist(dna[:, :5], dna[:, :5])
D_dna50 = cdist(dna[:, :50], dna[:, :50])
D_dna200 = cdist(dna[:, :200], dna[:, :200])
D_dna1000 = cdist(dna, dna)

D_hks5 = cdist(hks5.reshape(100, -1), hks5.reshape(100, -1))
D_hks50 = cdist(hks50.reshape(100, -1), hks50.reshape(100, -1))
D_hks200 = cdist(hks200.reshape(100, -1), hks200.reshape(100, -1))
D_hks1000 = cdist(hks1000.reshape(100, -1), hks1000.reshape(100, -1))

D_shks5 = cdist(shks5.reshape(100, -1), shks5.reshape(100, -1))
D_shks50 = cdist(shks50.reshape(100, -1), shks50.reshape(100, -1))
D_shks200 = cdist(shks200.reshape(100, -1), shks200.reshape(100, -1))
D_shks1000 = cdist(shks1000.reshape(100, -1), shks1000.reshape(100, -1))

D_mcurve = cdist(mcurve, mcurve)


gps_mds5 = mds(D_gps5, 2)
gps_mds50 = mds(D_gps50, 2)
gps_mds200 = mds(D_gps200, 2)
gps_mds1000 = mds(D_gps1000, 2)

dna_mds5 = mds(D_dna5, 2)
dna_mds50 = mds(D_dna50, 2)
dna_mds200 = mds(D_dna200, 2)
dna_mds1000 = mds(D_dna1000, 2)

hks_mds5 = mds(D_hks5, 2)
hks_mds50 = mds(D_hks50, 2)
hks_mds200 = mds(D_hks200, 2)
hks_mds1000 = mds(D_hks1000, 2)

shks_mds5 = mds(D_shks5, 2)
shks_mds50 = mds(D_shks50, 2)
shks_mds200 = mds(D_shks200, 2)
shks_mds1000 = mds(D_shks1000, 2)

mcurve_mds = mds(D_mcurve, 2)

plot_mds([dna_mds5, gps_mds5, hks_mds5, shks_mds5, mcurve_mds], [r'$D_{dna}$', r'$D_{gps}$', r'$D_{hks}$', r'$D_{shks}$', r'$D_{H}$'], os.path.join(result_dir, f"k5_mds.png"))
plot_mds([dna_mds50, gps_mds50, hks_mds50, shks_mds50, mcurve_mds], [r'$D_{dna}$', r'$D_{gps}$', r'$D_{hks}$', r'$D_{shks}$', r'$D_{H}$'], os.path.join(result_dir, f"k50_mds.png"))
plot_mds([dna_mds200, gps_mds200, hks_mds200, shks_mds200, mcurve_mds], [r'$D_{dna}$', r'$D_{gps}$', r'$D_{hks}$', r'$D_{shks}$', r'$D_{H}$'], os.path.join(result_dir, f"k200_mds.png"))
plot_mds([dna_mds1000, gps_mds1000, hks_mds1000, shks_mds1000, mcurve_mds], [r'$D_{dna}$', r'$D_{gps}$', r'$D_{hks}$', r'$D_{shks}$', r'$D_{H}$'], os.path.join(result_dir, f"k1000_mds.png"))

