"""
This is an implementation of the paper:
“Understanding the geometry of transport:
        Diffusion maps for Lagrangian trajectory data unravel coherent sets”,
Chaos, 2017. By Banisch and Koltai.
https://aip.scitation.org/doi/10.1063/1.4971788
"""

import os, warnings
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigs
from TheDynamicLaplacian.utils import DoubleGryeFlow_data, GifIt
from TheDynamicLaplacian.utils import Q_eps_hybrid_np_sparse as Qeps
from numpy import genfromtxt


# save_gif = True
T, m, n = 201, 200, 100
def get_X(corrupt='', perturbed='', bickley_type=''):
    if perturbed == 'perturbed':
        if bickley_type == 'regular':
            x = genfromtxt('..\data\BickleyJet\\bickley_x.csv', delimiter=',').transpose(1, 0)
            y = genfromtxt('..\data\BickleyJet\\bickley_y.csv', delimiter=',').transpose(1, 0)
        else:
            x = genfromtxt('..\data\BickleyJet\\bickley_wild_x.csv', delimiter=',').transpose(1, 0)
            y = genfromtxt('..\data\BickleyJet\\bickley_wild_y.csv', delimiter=',').transpose(1, 0)
        X = np.zeros((*x.shape, 2))
        X[..., 0] = x
        X[..., 1] = y
        # if save_gif:
        #     for i in range(X.shape[0]):
        #         plt.figure(figsize=(10, 5))
        #         plt.scatter(X[i, :, 0], X[i, :, 1], s=3, c=np.arange(X.shape[1]), cmap='hot')
        #         plt.savefig(f'data\q1\\bickley\Time{i}')
        #         plt.close()
        #     GifIt('data\q1\\bickley', 'data\q1\\bickley.gif')
    else:
        if os.path.exists('data/X.npy'):
            X = np.load('data/X.npy')
        else:
            X = DoubleGryeFlow_data(T=T, m=m, n=n)
            np.save('data/X.npy', X)

        if corrupt == 'corrupt':
            np.random.seed(132)
            mn = 500
            idx_sample = np.sort(np.random.permutation(np.arange(X.shape[1]))[:mn])
            X = X[:, idx_sample, :]
    return X


def Qeps_Leps(eps: float=0.0002, corrupt='', perturbed='', bickley_type='', X = None, opt=1):
    mn = X.shape[1]  # m * n
    # global X
    # if perturbed == 'perturbed':
    #     x = genfromtxt('..\data\BickleyJet\\bickley_wild_x.csv', delimiter=',').transpose(1, 0)
    #     y = genfromtxt('..\data\BickleyJet\\bickley_wild_y.csv', delimiter=',').transpose(1, 0)
    #     X = np.zeros((*x.shape, 2))
    #     X[..., 0] = x
    #     X[..., 1] = y
    #     mn = x.shape[1]
    #
    # if corrupt == 'corrupt':
    #     np.random.seed(1989)
    #     mn = 500
    #     idx_sample = np.sort(np.random.permutation(np.arange(X.shape[1]))[:mn])
    #     X = X[:, idx_sample, :]

    if os.path.exists(f'data\Qeps_{eps}{corrupt}{perturbed}{bickley_type}.npy'):
        Qeps_ = np.load(f'data\Qeps_{eps}{corrupt}{perturbed}{bickley_type}.npy')
    else:
        Qeps_ = Qeps(X, eps=eps, corrupt=corrupt, opt=opt)
        if corrupt == '':
            np.save(f'data\Qeps_{eps}{corrupt}{perturbed}{bickley_type}.npy', Qeps_)

    if os.path.exists(f'data\Leps_{eps}{corrupt}{perturbed}{bickley_type}.npy'):
        Leps_ = np.load(f'data\Leps_{eps}{corrupt}{perturbed}{bickley_type}.npy')
    else:
        Leps_ = (Qeps_ - np.identity(mn)) / eps
        if corrupt == '':
            np.save(f'data\Leps_{eps}{corrupt}{perturbed}{bickley_type}.npy', Leps_)
    return Qeps_, Leps_


def get_eval_evecs(k: int = 10, eps: float=0.0002, corrupt='', perturbed='', bickley_type='', X=None, opt=1):
    Qeps_, Leps_ = Qeps_Leps(eps=eps, corrupt=corrupt, perturbed=perturbed, bickley_type=bickley_type, X=X, opt=opt)
    w_Q, v_Q = eigs(Qeps_, k=k, which='LR')
    idx = np.argsort(abs(w_Q))[::-1]
    w_Q = w_Q[idx]
    v_Q = v_Q[:, idx]
    w_Q, v_Q = w_Q.real, v_Q.real

    w_L, v_L = eigs(Leps_, k=k, which='LR')
    idx = np.argsort(abs(w_L))
    w_L = w_L[idx]
    v_L = v_L[:, idx]
    w_L, v_L = w_L.real, v_L.real

    if corrupt == '':
        np.save(f'data\w_Q{eps}{corrupt}{perturbed}{bickley_type}.npy', w_Q)
        np.save(f'data\\v_Q{eps}{corrupt}{perturbed}{bickley_type}.npy', v_Q)

        np.save(f'data\w_L{eps}{corrupt}{perturbed}{bickley_type}.npy', w_L)
        np.save(f'data\\v_L{eps}{corrupt}{perturbed}{bickley_type}.npy', v_L)

    return w_Q, v_Q, w_L, v_L


def Gen_fig4(k: int = 10, scale: float = 0.5, regenerate=False, savefig=False):
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    eps_vec = [0.0002, 0.0005, 0.001, 0.002, 0.004]
    c       = ['b', 'g', 'y', 'b', 'r']
    marker  = ['^', 's', '*', 's', 'h']
    X = get_X()
    for i, eps in enumerate(eps_vec):
        if not regenerate and os.path.exists(f'data\w_Q{eps}.npy') and os.path.exists(
                f'data\\v_Q{eps}.npy') and os.path.exists(f'data\w_L{eps}.npy') and os.path.exists(
            f'data\\v_L{eps}.npy'):

            if k != 10 and not regenerate:
                warnings.warn(f'k is ignored: k is only effective if regenerate=True or *{eps}.np files are missing')

            w_Q = np.load(f'data\w_Q{eps}.npy')
            w_L = np.load(f'data\w_L{eps}.npy')
        else:
            w_Q, _, w_L, _ = get_eval_evecs(k=25, eps=eps, X=X)

        ax1.scatter(np.arange(1, len(w_Q) + 1), w_Q, s=20, c=c[i], marker=marker[i], label=str(eps))
        ax2.scatter(np.arange(1, len(w_L) + 1), scale*w_L, s=20, c=c[i], marker=marker[i], label=str(eps))

    ax1.legend()
    ax1.set_title('eigenvalues of Q_{eps}')
    ax1.set_xlabel('n')
    ax1.set_ylabel('eigenvalues')
    ax1.set_xlim(left=0, right=k + 0.1)
    ax1.set_ylim(bottom=0.65, top=1.01)
    if savefig:
        fig1.savefig(f'data\q1\\fig4_Qeps')

    ax2.legend()
    ax2.set_title('Scaled eigenvalues of the space-time diffusion matrix')
    ax2.set_xlabel('n')
    ax2.set_ylabel('eigenvalues')
    ax2.set_xlim(left=0, right=k + 0.1)
    ax2.set_ylim(bottom=-70, top=1)
    if savefig:
        fig2.savefig(f'data\q1\\fig4')
    plt.show()
    return


def Gen_figs5(k: int = 3, regenerate=False, savefig = False):
    eps = [0.0002]
    X = get_X()
    if not regenerate and os.path.exists(f'data\w_Q{eps[0]}.npy') and os.path.exists(
            f'data\\v_Q{eps[0]}.npy') and os.path.exists(f'data\w_L{eps[0]}.npy') and os.path.exists(
        f'data\\v_L{eps[0]}.npy'):

            w_Q = np.load(f'data\w_Q{eps[0]}.npy')
            v_Q = np.load(f'data\\v_Q{eps[0]}.npy')
            w_L = np.load(f'data\w_L{eps[0]}.npy')
            v_L = np.load(f'data\\v_L{eps[0]}.npy')
    else:
        w_Q, v_Q, w_L, v_L = get_eval_evecs(k=k, eps=eps[0], X=X)

    embedding   = 1 / w_L[1:k, None] * v_L[:, 1:k].T
    kmeans      = KMeans(n_clusters=k).fit(embedding.T)
    clusters = kmeans.predict(embedding.T)

    plt.figure('', figsize=(7, 5))
    ax = plt.axes()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.scatter(*X[1].T, c=clusters, s=20)
    plt.title('Result of 3-clustering the double gyre trajectory at initial time (t = 0)')
    if savefig:
        plt.savefig('data\q1\\fig5a_L')
    plt.show()

    plt.figure('', figsize=(7, 5))
    ax = plt.axes()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.scatter(*X[196].T, c=clusters, s=20)
    plt.title('Result of 3-clustering the double gyre trajectory at t = 19.5')
    if savefig:
        plt.savefig('data\q1\\fig5b_L')
    plt.show()

    embedding   = 1 / w_Q[1:k, None] * v_Q[:, 1:k].T
    kmeans      = KMeans(n_clusters=k).fit(embedding.T)
    clusters = kmeans.predict(embedding.T)

    plt.figure('', figsize=(7, 5))
    ax = plt.axes()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.scatter(*X[1].T, c=clusters, s=20)
    plt.title('Result of 3-clustering the double gyre trajectory at initial time (t = 0)')
    if savefig:
        plt.savefig('data\q1\\fig5a_Q')
    plt.show()

    plt.figure('', figsize=(7, 5))
    ax = plt.axes()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.scatter(*X[196].T, c=clusters, s=20)
    plt.title('Result of 3-clustering the double gyre trajectory at t = 19.5')
    if savefig:
        plt.savefig('data\q1\\fig5b_Q')
    plt.show()

    return

def Gen_figs6(k: int = 2, regenerate=False, savefig = False):
    eps = [0.0002, 0.004]
    fig_num_u = ['a', 'b']
    fig_num_d = ['c', 'd']
    X = get_X()
    for i in range(len(eps)):
        if not regenerate and os.path.exists(f'data\w_Q{eps[i]}.npy') and os.path.exists(
                f'data\\v_Q{eps[i]}.npy') and os.path.exists(f'data\w_L{eps[i]}.npy') and os.path.exists(
            f'data\\v_L{eps[i]}.npy'):

                w_Q = np.load(f'data\w_Q{eps[i]}.npy')
                v_Q = np.load(f'data\\v_Q{eps[i]}.npy')
                w_L = np.load(f'data\w_L{eps[i]}.npy')
                v_L = np.load(f'data\\v_L{eps[i]}.npy')
        else:
            w_Q, v_Q, w_L, v_L = get_eval_evecs(k=k, eps=eps[i], X=X)

        embedding   = 1 / w_L[1:k, None] * v_L[:, 1:k].T
        kmeans      = KMeans(n_clusters=k).fit(embedding.T)
        clusters = kmeans.predict(embedding.T)

        plt.figure('', figsize=(7, 5))
        ax = plt.axes()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.scatter(*X[1].T, c=clusters)
        plt.title(f'Result of 2-clustering the d. g. trajectory at initial time, eps={eps[i]}')
        if savefig:
            plt.savefig(f'data\q1\\fig6{fig_num_d[i]}_L')
        plt.show()

        plt.figure('', figsize=(7, 5))
        ax = plt.axes()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.scatter(*X[1].T, c=v_L[:, 1:k].T)
        plt.title(f'Second eigenfunction of Q_eps, for eps={eps[i]}')
        if savefig:
            plt.savefig(f'data\q1\\fig6{fig_num_u[i]}_L')
        plt.show()

        embedding   = 1 / w_Q[1:k, None] * v_Q[:, 1:k].T
        kmeans      = KMeans(n_clusters=k).fit(embedding.T)
        clusters = kmeans.predict(embedding.T)

        plt.figure('', figsize=(7, 5))
        ax = plt.axes()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.scatter(*X[1].T, c=clusters)
        plt.title(f'Result of 2-clustering the d. g. trajectory at initial time, eps={eps[i]}')
        if savefig:
            plt.savefig(f'data\q1\\fig6{fig_num_d[i]}_Q')
        plt.show()

        plt.figure('', figsize=(7, 5))
        ax = plt.axes()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.scatter(*X[1].T, c=v_Q[:, 1:k].T)
        plt.title(f'Second eigenfunction of Q_eps, for eps={eps[i]}')
        if savefig:
            plt.savefig(f'data\q1\\fig6{fig_num_u[i]}_Q')
        plt.show()

    return


def Gen_figs7(k: int = 4, regenerate=False, savefig=False):
    eps = [0.0005, 0.004]
    fig_num = ['a', 'b']
    X = get_X()
    for i in range(len(eps)):
        if not regenerate and os.path.exists(f'data\w_Q{eps[i]}.npy') and os.path.exists(
                f'data\\v_Q{eps[i]}.npy') and os.path.exists(f'data\w_L{eps[i]}.npy') and os.path.exists(
            f'data\\v_L{eps[i]}.npy'):

                w_Q = np.load(f'data\w_Q{eps[i]}.npy')
                v_Q = np.load(f'data\\v_Q{eps[i]}.npy')
                w_L = np.load(f'data\w_L{eps[i]}.npy')
                v_L = np.load(f'data\\v_L{eps[i]}.npy')
        else:
            w_Q, v_Q, w_L, v_L = get_eval_evecs(k=k, eps=eps[i], X=X)

        embedding   = 1 / w_L[1:k, None] * v_L[:, 1:k].T
        kmeans      = KMeans(n_clusters=k).fit(embedding.T)
        clusters = kmeans.predict(embedding.T)

        plt.figure('', figsize=(7, 5))
        ax = plt.axes()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.scatter(*X[1].T, c=clusters)
        plt.title(f'Result of 4-clustering the d. g. trajectory at initial time, eps={eps[i]}')
        if savefig:
            plt.savefig(f'data\q1\\fig7{fig_num[i]}_L')
        plt.show()


        if eps[i]==0.0005:
            embedding = 1 / w_Q[1:k+2, None] * v_Q[:, 1:k+2].T
        else:
            embedding = 1 / w_Q[1:k, None] * v_Q[:, 1:k].T
        kmeans      = KMeans(n_clusters=k).fit(embedding.T)
        clusters = kmeans.predict(embedding.T)

        plt.figure('', figsize=(7, 5))
        ax = plt.axes()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.scatter(*X[1].T, c=clusters)
        plt.title(f'Result of 4-clustering the d. g. trajectory at initial time, eps={eps[i]}')
        if savefig:
            plt.savefig(f'data\q1\\fig7{fig_num[i]}_Q')
        plt.show()

    return


def Gen_figs8(k: np.array = 25, regenerate=False, savefig=False, corrupt='', opt=1):
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()

    eps = [0.004]
    X = get_X()
    if not regenerate and os.path.exists(f'data\w_Q{eps[0]}.npy') and os.path.exists(
            f'data\\v_Q{eps[0]}.npy') and os.path.exists(f'data\w_L{eps[0]}.npy') and os.path.exists(
        f'data\\v_L{eps[0]}.npy'):

        w_Q = np.load(f'data\w_Q{eps[0]}.npy')
        v_Q = np.load(f'data\\v_Q{eps[0]}.npy')
        w_L = np.load(f'data\w_L{eps[0]}.npy')
        v_L = np.load(f'data\\v_L{eps[0]}.npy')
    else:
        w_Q, v_Q, w_L, v_L = get_eval_evecs(k=k, eps=eps[0], X=X)

    embedding = 1 / w_L[1:2, None] * v_L[:, 1:2].T
    kmeans = KMeans(n_clusters=2).fit(embedding.T)
    clusters = kmeans.predict(embedding.T)
    ax1.scatter(*X[1].T, c=clusters, s=20)

    embedding = 1 / w_L[1:3, None] * v_L[:, 1:3].T
    kmeans = KMeans(n_clusters=3).fit(embedding.T)
    clusters = kmeans.predict(embedding.T)
    ax2.scatter(*X[1].T, c=clusters, s=20)

    embedding = 1 / w_Q[1:2, None] * v_Q[:, 1:2].T
    kmeans = KMeans(n_clusters=2).fit(embedding.T)
    clusters = kmeans.predict(embedding.T)
    ax3.scatter(*X[1].T, c=clusters, s=20)

    embedding = 1 / w_Q[1:3, None] * v_Q[:, 1:3].T
    kmeans = KMeans(n_clusters=3).fit(embedding.T)
    clusters = kmeans.predict(embedding.T)
    ax4.scatter(*X[1].T, c=clusters, s=20)


    eps = [0.01]
    i = 0
    X = get_X(corrupt=corrupt)
    w_Q, v_Q, w_L, v_L = get_eval_evecs(k=k, eps=eps[i], corrupt=corrupt, X=X, opt=opt)

    embedding   = 1 / w_L[1:2, None] * v_L[:, 1:2].T
    kmeans      = KMeans(n_clusters=2).fit(embedding.T)
    clusters = kmeans.predict(embedding.T)

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.scatter(*X[1].T, c=clusters, cmap='jet', s=10)
    ax1.set_title(f'Result of 2-clustering the d. g. trajectory at initial time, eps={eps[i]}')
    if savefig:
        fig1.savefig(f'data\q1\\fig8a_L{corrupt}')

    embedding = 1 / w_L[1:3, None] * v_L[:, 1:3].T
    kmeans = KMeans(n_clusters=3).fit(embedding.T)
    clusters = kmeans.predict(embedding.T)

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.scatter(*X[1].T, c=clusters, cmap='jet', s=10)
    ax2.set_title(f'Result of 3-clustering the d. g. trajectory at initial time, eps={eps[i]}')
    if savefig:
        fig2.savefig(f'data\q1\\fig8b_L{corrupt}')


    embedding = 1 / w_Q[1:2, None] * v_Q[:, 1:2].T
    kmeans      = KMeans(n_clusters=2).fit(embedding.T)
    clusters = kmeans.predict(embedding.T)

    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.scatter(*X[1].T, c=clusters, cmap='jet', s=10)
    ax3.set_title(f'Result of 2-clustering the d. g. trajectory at initial time, eps={eps[i]}')
    if savefig:
        fig3.savefig(f'data\q1\\fig8a_Q{corrupt}')

    embedding = 1 / w_Q[1:3, None] * v_Q[:, 1:3].T
    kmeans = KMeans(n_clusters=3).fit(embedding.T)
    clusters = kmeans.predict(embedding.T)

    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.scatter(*X[1].T, c=clusters, cmap='jet', s=10)
    ax4.set_title(f'Result of 3-clustering the d. g. trajectory at initial time, eps={eps[i]}')
    if savefig:
        fig4.savefig(f'data\q1\\fig8b_Q{corrupt}')

    plt.show()
    return

def Gen_figs9a(k: int = 20, scale: float = 0.5, regenerate=False, savefig=False, bickley_type=''):
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    eps_vec = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    c       = ['r', 'b', 'y', 'g', 'c', 'k']
    marker  = ['o', 'x', '*', 's', '+', '^']
    assert bickley_type == 'regular' or bickley_type == '', "bickley_type must be either 'regular' or '' (for wild)"

    X = get_X(perturbed='perturbed', bickley_type=bickley_type)
    for i, eps in enumerate(eps_vec):
        if not regenerate and os.path.exists(f'data\w_Q{eps}perturbed{bickley_type}.npy') and os.path.exists(
                f'data\\v_Q{eps}perturbed{bickley_type}.npy') and os.path.exists(
            f'data\w_L{eps}perturbed{bickley_type}.npy') and os.path.exists(
            f'data\\v_L{eps}perturbed{bickley_type}.npy'):

            # if k != 10 and not regenerate:
            #     warnings.warn(f'k is ignored: k is only effective if regenerate=True or *{eps}.np files are missing')

            w_Q = np.load(f'data\w_Q{eps}perturbed{bickley_type}.npy')
            w_L = np.load(f'data\w_L{eps}perturbed{bickley_type}.npy')
        else:
            w_Q, _, w_L, _ = get_eval_evecs(k=25, eps=eps, perturbed='perturbed', bickley_type=bickley_type, X=X)

        ax1.scatter(np.arange(1, len(w_Q) + 1), w_Q, s=20, c=c[i], marker=marker[i], label=str(eps))
        ax2.scatter(np.arange(1, len(w_L) + 1), scale*w_L, s=20, c=c[i], marker=marker[i], label=str(eps))

    ax1.legend()
    ax1.set_title('Bickley jet eigenvalues')
    ax1.set_xlabel('n')
    ax1.set_ylabel('eigenvalues')
    ax1.set_xlim(left=0, right=k + 0.1)
    ax1.set_ylim(bottom=0.25, top=1.01)
    if savefig:
        fig1.savefig(f'data\q1\\fig9a_Qeps{bickley_type}')

    ax2.legend()
    ax2.set_title('Bickley jet eigenvalues')
    ax2.set_xlabel('n')
    ax2.set_ylabel('eigenvalues')
    ax2.set_xlim(left=0, right=k + 0.1)
    ax2.set_ylim(bottom=-7, top=0.1)
    if savefig:
        fig2.savefig(f'data\q1\\fig9a_Leps{bickley_type}')
    plt.show()
    return


def Gen_figs9b_10(k: int = 9, regenerate=False, savefig=False, bickley_type=''):
    eps = [0.02]
    assert bickley_type == 'regular' or bickley_type == '', "bickley_type must be either 'regular' or '' (for wild)"
    X = get_X(perturbed='perturbed', bickley_type=bickley_type)
    if not regenerate and os.path.exists(f'data\w_Q{eps[0]}perturbed{bickley_type}.npy') and os.path.exists(
            f'data\\v_Q{eps[0]}perturbed{bickley_type}.npy') and os.path.exists(
        f'data\w_L{eps[0]}perturbed{bickley_type}.npy') and os.path.exists(
        f'data\\v_L{eps[0]}perturbed{bickley_type}.npy'):

            w_Q = np.load(f'data\w_Q{eps[0]}perturbed{bickley_type}.npy')
            v_Q = np.load(f'data\\v_Q{eps[0]}perturbed{bickley_type}.npy')
            w_L = np.load(f'data\w_L{eps[0]}perturbed{bickley_type}.npy')
            v_L = np.load(f'data\\v_L{eps[0]}perturbed{bickley_type}.npy')
    else:
        w_Q, v_Q, w_L, v_L = get_eval_evecs(k=k, eps=eps[0], perturbed='perturbed', X=X)

    embedding   = 1 / w_L[1:k, None] * v_L[:, 1:k].T
    kmeans      = KMeans(n_clusters=k).fit(embedding.T)
    clusters = kmeans.predict(embedding.T)

    plt.figure('', figsize=(7, 5))
    ax = plt.axes(projection='3d')
    ax.set_xlabel('1')
    ax.set_ylabel('3')
    ax.set_zlabel('4')
    ax.scatter(embedding[0], embedding[2], embedding[3], c=clusters, cmap='jet')
    plt.title('Result of 9-clustering of Bickley jet, using the eigenfunctions 1, 3, and 4')
    if savefig:
        plt.savefig(f'data\q1\\fig9b_L{bickley_type}')
    plt.show()


    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('Bickley jet, clusters at t=5, 20, 35 for 9-clustering')
    ax1.scatter(*X[5].T, c=clusters, s=1, cmap='jet')
    ax2.scatter(*X[20].T, c=clusters, s=1, cmap='jet')
    ax3.scatter(*X[35].T, c=clusters, s=1, cmap='jet')
    ax3.set_xlabel('x [mM]')
    ax3.set_ylabel('y [mM]')
    ax1.set_xlim(0, 20)
    ax1.set_ylim(-3.1, 3.1)
    ax2.set_xlim(0, 20)
    ax2.set_ylim(-3.1, 3.1)
    ax3.set_xlim(0, 20)
    ax3.set_ylim(-3.1, 3.1)

    if savefig:
        plt.savefig(f'data\q1\\fig10a_L{bickley_type}')
    plt.show()

    embedding   = 1 / w_Q[1:k, None] * v_Q[:, 1:k].T
    kmeans      = KMeans(n_clusters=k).fit(embedding.T)
    clusters = kmeans.predict(embedding.T)

    plt.figure('', figsize=(7, 5))
    ax = plt.axes(projection='3d')
    ax.set_xlabel('1')
    ax.set_ylabel('3')
    ax.set_zlabel('4')
    ax.scatter(embedding[0], embedding[2], embedding[3], c=clusters, cmap='jet')
    plt.title('Result of 9-clustering of Bickley jet, using the eigenfunctions 1, 3, and 4')
    if savefig:
        plt.savefig(f'data\q1\\fig9b_Q{bickley_type}')
    plt.show()


    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('Bickley jet, clusters at t=5, 20, 35 for 9-clustering')
    ax1.scatter(*X[5].T, c=clusters, s=1, cmap='jet')
    ax2.scatter(*X[20].T, c=clusters, s=1, cmap='jet')
    ax3.scatter(*X[35].T, c=clusters, s=1, cmap='jet')
    ax3.set_xlabel('x [mM]')
    ax3.set_ylabel('y [mM]')
    ax1.set_xlim(0, 20)
    ax1.set_ylim(-3.1, 3.1)
    ax2.set_xlim(0, 20)
    ax2.set_ylim(-3.1, 3.1)
    ax3.set_xlim(0, 20)
    ax3.set_ylim(-3.1, 3.1)

    if savefig:
        plt.savefig(f'data\q1\\fig10a_Q{bickley_type}')
    plt.show()


    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('eigenfunctions 1, 2, and 3 at t=20')
    ax1.scatter(*X[20].T, c=v_L[:, 1], s=1, cmap='jet')
    ax2.scatter(*X[20].T, c=v_L[:, 2], s=1, cmap='jet')
    ax3.scatter(*X[20].T, c=v_L[:, 3], s=1, cmap='jet')
    ax3.set_xlabel('x [mM]')
    ax3.set_ylabel('y [mM]')
    ax1.set_xlim(0, 20)
    ax1.set_ylim(-3.1, 3.1)
    ax2.set_xlim(0, 20)
    ax2.set_ylim(-3.1, 3.1)
    ax3.set_xlim(0, 20)
    ax3.set_ylim(-3.1, 3.1)

    if savefig:
        plt.savefig(f'data\q1\\fig10b_L{bickley_type}')
    plt.show()


    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('eigenfunctions 1, 2, and 3 at t=20')
    ax1.scatter(*X[20].T, c=v_Q[:, 1], s=1, cmap='jet')
    ax2.scatter(*X[20].T, c=v_Q[:, 2], s=1, cmap='jet')
    ax3.scatter(*X[20].T, c=v_Q[:, 3], s=1, cmap='jet')
    ax3.set_xlabel('x [mM]')
    ax3.set_ylabel('y [mM]')
    ax1.set_xlim(0, 20)
    ax1.set_ylim(-3.1, 3.1)
    ax2.set_xlim(0, 20)
    ax2.set_ylim(-3.1, 3.1)
    ax3.set_xlim(0, 20)
    ax3.set_ylim(-3.1, 3.1)
    if savefig:
        plt.savefig(f'data\q1\\fig10b_Q{bickley_type}')
    plt.show()
    return


Gen_fig4(scale=0.5, regenerate=False, savefig=False)
Gen_figs5(regenerate=False, savefig=False)
Gen_figs6(regenerate=False, savefig=False)
Gen_figs7(regenerate=False, savefig=False)
Gen_figs8(regenerate=False, savefig=False, corrupt='corrupt', opt=1)
Gen_figs9a(regenerate=False, savefig=False, bickley_type='')
Gen_figs9b_10(regenerate=False, savefig=False, bickley_type='')
