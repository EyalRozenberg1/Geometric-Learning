from skimage import data
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
root = Path(Path(__file__).resolve().parents[1])

import matplotlib
matplotlib.use('TkAgg')
import os
import scipy.sparse as sparse
from scipy.sparse.linalg import expm
from scipy.sparse.linalg import cg
results_dir = os.path.join(root, 'figs', 'Ex4', 'Q1')

Q="Q.1c"

grayscale = rgb2gray(data.astronaut())
X0 = np.array(grayscale, dtype=np.float)
Y = X0 + 0.1 * np.random.randn(*(X0.shape))
y = Y.reshape(-1)

if Q=="Q.1b":
    plt.imshow(Y, cmap=plt.cm.gray)
    plt.savefig(os.path.join(results_dir, 'Y'))

    plt.imshow(X0, cmap=plt.cm.gray)
    plt.savefig(os.path.join(results_dir, 'X0'))

if Q=="Q.1c":
    def ConvMtx(H, n, m):
        Idxs = np.reshape(np.arange(n*m), (m, n)).T
        Idxs = np.pad(Idxs, np.floor(np.array(H.shape)/2).astype(np.int), mode='symmetric')

        Cols = np.zeros((m * n * H.size, 1))
        Rows = np.zeros((m * n * H.size, 1))
        Vals = np.zeros((m * n * H.size, 1))

        i = -1
        p = -1
        for c in np.arange(n):
            for r in np.arange(m):
                p += 1
                mKerInds = Idxs[r:r + H.shape[0], c: c + H.shape[1]]
                U = np.unique(mKerInds.reshape(-1))
                for k in np.arange(U.size):
                    i += 1
                    Cols[i] = U[k]
                    Rows[i] = p
                    Vals[i] = H[mKerInds == U[k]].sum()
        T = sparse.csr_matrix((Vals[:i+1,0], (Rows[:i+1,0], Cols[:i+1,0])),  shape=[m*n, m*n], dtype=np.float)
        return T

    H = -np.ones((3, 3)) / 8
    H[1, 1] = 1
    Operator = ConvMtx(H, *(Y.shape))
    gamma = 10
    maxiter = 1000
    tol = 1e-11
    eye_add = sparse.eye(*(Operator.shape))
    rec_img, info = cg(eye_add + gamma*Operator, y, tol=tol, maxiter=maxiter)
    rec_img = rec_img - rec_img.min()
    rec_img = rec_img/rec_img.max()

    rec_img = rec_img.reshape(*(Y.shape))
    plt.imshow(rec_img, cmap=plt.cm.gray)
    plt.savefig(os.path.join(results_dir, f'Y_rec1_g{gamma}_tol{tol}_info{info}'))

if Q == "Q.1d":
        def WeightMat(y, n, m, sigma):
            p = n * m
            y = y.reshape((n, m))
            zero_r = np.zeros((1, m))
            m_zero_r = np.zeros((1, m - 1))

            diff_next_row = y[:-1] - y[1:]
            diag_p1 = np.concatenate((zero_r, diff_next_row))
            diag_m1 = np.concatenate((diff_next_row, zero_r))

            diff_prev_row_next_col = y[1:,:-1] - y[:-1, 1:]
            diag_p2 = np.concatenate((diff_prev_row_next_col, m_zero_r))
            diag_m2 = np.concatenate((m_zero_r, diff_prev_row_next_col))

            diag3 = y[:,:-1] - y[:,1:]

            diff_next_row_prev_col = y[:-1,:-1] - y[1:,1:]
            diag_p4 = np.concatenate((m_zero_r, diff_next_row_prev_col))
            diag_m4 = np.concatenate((diff_next_row_prev_col, m_zero_r))

            pad = np.zeros((m, 1))
            bin_m4 = np.concatenate((diag_m4.reshape(-1, 1), pad))
            bin_m3 = np.concatenate((diag3.reshape(-1, 1), pad))
            bin_m2 = np.concatenate((diag_m2.reshape(-1, 1), pad))
            bin_m1 = diag_m1.reshape(-1, 1)
            bin_p1 = diag_p1.reshape(-1, 1)
            bin_p2 = np.concatenate((pad, diag_p2.reshape(-1, 1)))
            bin_p3 = np.concatenate((pad, diag3.reshape(-1, 1)))
            bin_p4 = np.concatenate((pad, diag_p4.reshape(-1, 1)))
            bin = np.concatenate([bin_m4, bin_m3, bin_m2, bin_m1, bin_p1, bin_p2, bin_p3, bin_p4], 1)
            d = np.array([-m - 1, -m, -m + 1, -1, 1, m - 1, m, m + 1])
            diff = sparse.spdiags(bin.T, d, p, p)

            W = expm(-diff.power(2) / (2 * sigma**2))
            return W


        def build_lg(y, n, m, sigma):
            W = WeightMat(y, n, m, sigma)
            diag = sparse.coo_matrix(W.sum(0), shape=(1, m*n))
            L = sparse.diags(diag.toarray().squeeze(0), 0) - W
            return L

        sigma = 0.1
        gamma = 5
        maxiter = 1000
        tol = 1e-11
        L = build_lg(y, *(Y.shape), sigma)
        eye_add = sparse.eye(*(L.shape))
        total_op = eye_add + gamma * L
        rec_img, info = cg(total_op, y)
        rec_img = rec_img - rec_img.min()
        rec_img = rec_img / rec_img.max()

        rec_img = rec_img.reshape(*(Y.shape))
        plt.imshow(rec_img, cmap=plt.cm.gray)
        plt.savefig(os.path.join(results_dir, f'Y_rec2_g{gamma}_tol{tol}_info{info}'))