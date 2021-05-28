import matplotlib.pyplot as plt
import numpy as np
import imageio, glob, os
from scipy.integrate import odeint
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import sklearn.neighbors as neigh_search
from scipy.sparse import csr_matrix, diags, identity
import time
from scipy.spatial.distance import pdist, squareform

# import torch
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def GifIt(path, name):
    """
    :param path: directory with images
    :param name: Generated Gif name
    :return: Gif generated from images in directory
    """
    image_list = []
    for filename in sorted(glob.glob(path + '\\*'), key=os.path.getmtime):
        im = Image.open(filename)
        image_list.append(im)
    images = []
    for filename in image_list:
        images.append(imageio.imread(filename.filename))
    imageio.mimsave(name, images, fps=2)
    return


def dynamic_sys(xy, t, A: float = 0.25, alpha: float = 0.25, omega: float = 2*np.pi):
    m = int(np.size(xy) / 2)
    x = xy[:m]
    y = xy[m:]
    f = alpha * np.sin(omega * t) * x ** 2 + (1 - 2 * alpha * np.sin(omega * t)) * x
    df_dx = 2 * x * alpha * np.sin(omega * t) + 1 - 2 * alpha * np.sin(omega * t)
    xdot = -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * y)
    ydot = np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * y) * df_dx
    dx_dy = np.concatenate((xdot.reshape(-1), ydot.reshape(-1)))
    return dx_dy


def DoubleGryeFlow_data(T: int = 201, m=200, n=100, save=False):
    xx, yy = np.meshgrid(np.linspace(0, 2, m), np.linspace(0, 1, n))
    t = np.linspace(0, 20, T)
    m = np.size(xx)
    xy = np.concatenate((xx.reshape(-1), yy.reshape(-1)))

    xy_int = odeint(dynamic_sys, xy, t)
    X = xy_int[:, : m]
    Y = xy_int[:, m:]
    XY = np.zeros((2, m, T))

    for i in range(T):
        XY[:, :, i] = np.concatenate((X[None, i, :], Y[None, i, :]))
        if save:
            plt.figure(figsize=(10, 5))
            plt.scatter(XY[0, :, i], XY[1, :, i], s=3, c=np.arange(20e3), cmap='hot')
            plt.savefig(f'data\q1\All\Time{i}')
            plt.close()
    if save:
        GifIt('data/q1/All', 'data\q1\Double_gyre_flow.gif')
    return XY.T  # array shaped as T x mn, 2


def Q_eps_sparse(data, eps: float=0.4, r: float=2):
    radius = np.sqrt(r * eps)
    T, n, _ = data.shape
    Q = csr_matrix((n, n))
    for t in range(T):
        neigh = NearestNeighbors(radius=radius, algorithm='ball_tree').fit(data[t])
        K = neigh.radius_neighbors_graph(data[t], mode='distance')  # CSR sparse matrix
        K = -K.power(2)/eps
        K = K.expm1()
        K.data += 1

        Pepsi = diags(csr_matrix(1/K.sum(1), shape=(n, 1)).toarray().squeeze(1), 0) * K
        Depsi = diags(csr_matrix(1/Pepsi.sum(0), shape=(1, n)).toarray().squeeze(0), 0)

        Q += Depsi * Pepsi.T * Pepsi
    return Q/T


def Q_eps_np(data, eps: float=0.4, r: float=2):
    radius = np.sqrt(r * eps)
    T, n, _ = data.shape
    Q = csr_matrix((n, n))
    for t in range(T):
        dist_mat = squareform(pdist(data[t]))
        K = np.exp(-dist_mat ** 2 / eps)
        K = csr_matrix(K*(dist_mat < radius))


        Pepsi = diags(csr_matrix(1/K.sum(1), shape=(n, 1)).toarray().squeeze(1), 0) * K
        Depsi = diags(csr_matrix(1/Pepsi.sum(0), shape=(1, n)).toarray().squeeze(0), 0)

        Q += Depsi * Pepsi.T * Pepsi
    return Q/T


def Q_eps_hybrid_np_sparse(data, eps: float = 0.4, r: float = 2, corrupt='', opt: int = 1):
    """
    :param data: the analysed dataset
    :param eps: epsilon as in the article
    :param r: r is defoult 2 as in the article
    :param corrupt: a flag that indicates if corrupted dataset is analysed. 'corrupt' for True
    :param opt: Two optional methods were implemented: 1) erasing elements directly on the kernel.
                                                       0) incraesig the value of selected points to infinity
    :return: Qeps
    """
    radius = np.sqrt(r * eps)
    T, n, _ = data.shape
    Q = np.zeros((n, n))
    for t in range(T):
        data_ = data[t].copy()
        if corrupt == 'corrupt' and opt == 0:
            samp = np.random.permutation(np.arange(n))[:int(0.8 * n)]
            data_[samp, 0] *= 1e9
            data_[samp, 1] *= 1e9

        K = neigh_search.radius_neighbors_graph(data_, radius, mode='distance')

        if corrupt == 'corrupt' and opt == 1:
            corrup_samples = np.random.permutation(np.arange(n))[:int(0.8 * 500)]
            mask = np.ones((n, n))
            mask[corrup_samples, :] = 0
            mask[:, corrup_samples] = 0
            mask = csr_matrix(mask)
            K = K.multiply(mask)

        K.data = np.exp(-(K.data ** 2) / eps)
        K = K + identity(K.shape[0], format='csr')

        # Pepsi = (1. / np.sum(K, axis=1)) * K
        # Q += (np.dot(Pepsi, Pepsi.T) / (np.sum(Pepsi, axis=1))).T
        Pepsi = diags(csr_matrix(1 / K.sum(1), shape=(n, 1)).toarray().squeeze(1), 0) * K
        Depsi = diags(csr_matrix(1 / Pepsi.sum(0), shape=(1, n)).toarray().squeeze(0), 0)

        Q += Depsi * Pepsi.T * Pepsi
    return Q/T  # Eq. 19

def Q_eps_torch(data, eps: float=0.4, r: float=2):
    radius = np.sqrt(r * eps)
    T = data.shape[0]
    dist_mat = (data.norm(dim=2, keepdim=True) ** 2 + (data.norm(dim=2, keepdim=True) ** 2).transpose(1, 2) - 2 * (
                data @ data.transpose(1, 2))) ** 0.5

    # dist_mat[dist_mat != dist_mat] = 0.0
    dist_mat.diagonal(offset=0, dim1=1, dim2=2).fill_(0.0)
    K = torch.exp(-dist_mat**2/eps)


    K.diagonal(offset=0, dim1=1, dim2=2).fill_(1.0)
    # K[dist_mat > radius] = 0.0
    K *= (dist_mat < radius)
    del dist_mat

    Pepsi = torch.diag_embed(1 / K.sum(2), dim1=1, dim2=2) @ K
    Depsi = torch.diag_embed(1 / Pepsi.sum(1), dim1=1, dim2=2)

    Q = Depsi @ Pepsi.transpose(1, 2) @ Pepsi
    Q = Q.sum(0)
    return Q/T

def Q_eps_hybrid_np_torch(data, eps: float=0.4, r: float=2):
    radius = np.sqrt(r * eps)
    T, n, _ = data.shape
    Q = csr_matrix((n, n))
    for t in range(T):
        K = (data[t].norm(dim=1, keepdim=True) ** 2 + (data[t].norm(dim=1, keepdim=True) ** 2).T - 2 * (
                    data[t] @ data[t].T)) ** 0.5

        K.diagonal(offset=0).fill_(0.0)
        K[K != K] = 0
        K *= (K < radius)

        K = csr_matrix(K.cpu().numpy())
        K = -K.power(2) / eps
        K = K.expm1()
        K.data += 1

        Pepsi = diags(csr_matrix(1 / K.sum(1), shape=(n, 1)).toarray().squeeze(1), 0) * K
        Depsi = diags(csr_matrix(1 / Pepsi.sum(0), shape=(1, n)).toarray().squeeze(0), 0)

        Q += Depsi * Pepsi.T * Pepsi
    return Q/T

if __name__ == '__main__':
    start_time = time.time()
    T, m, n = 2, 200, 100
    XY = DoubleGryeFlow_data(T=T, m=m, n=n)

    print("--- Generate DoubleGryeFlow_data  %s seconds with T=%s, m=%s, n=%s ---" % (time.time() - start_time, T, m, n))
    # plt.scatter(XY[83, :, 0], XY[83, :, 1], cmap='hot', c=np.arange(20e3)), plt.show()
    print(XY.shape)

    # XY = np.random.randn(10, 2000, 2)
    start_time = time.time()
    Qeps_ = Q_eps_hybrid_np_sparse(XY, eps=0.004)
    print("--- Q_eps sparse %s seconds ---" % (time.time() - start_time))


    # start_time = time.time()
    # Q_np = Q_eps_np(XY, eps=0.4)
    # print("--- Q_eps numpy %s seconds ---" % (time.time() - start_time))
    #
    # print("--- ||np_sparse-np|| %s  ---" % (((Q_np_sparse - Q_np)**2).sum()))


