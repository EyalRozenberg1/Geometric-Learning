import numpy as np
from numpy.linalg import eig
from numpy.linalg import norm
from numpy.random import randn

eps = 1e-5
n = 30

def PowerMethod(B, eps=1e-5):
    """
    PowerMethod(B): returns the largest (in absolute value) eigenvalue and
    the corresponding eigenvector. With a reasonable stopping criterion to the iterative procedure
    :param B: the analyzed matrix
    :param eps: stopping threshold
    :return: eigenvalue, eigenvector
    """
    n = B.shape[0]
    v1 = randn(n)
    v1 = v1 / norm(v1)
    lambda1 = v1 @ B @ v1
    while norm(B @ v1 - lambda1 * v1) >= eps:
        v1 = B @ v1
        v1 = v1 / norm(v1)
        lambda1 = v1 @ B @ v1
    return v1, lambda1


def PowerMethod2(B, lambda1=None, v1=None, eps=1e-5):
    """
    PowerMethod(B): returns the second largest (in absolute value) eigenvalue and
    the corresponding eigenvector. With a reasonable stopping criterion to the iterative procedure
    :param B: the analyzed matrix
    :param eps: stopping threshold
    :return: eigenvalue, eigenvector
    """
    if lambda1 is None or v1 is None:
        v1, lambda1 = PowerMethod(B, eps)

    # Wielandt Deflation
    B = B - lambda1 * v1[:, None] @ v1[:, None].T
    u2, lambda2 = PowerMethod(B, eps)
    v2 = (lambda2 - lambda1) * u2 + lambda1 * (v1 @ u2) * v1
    v2 = v2 / norm(v2)

    return v2, lambda2

def GetMat(n, eps):

    # generate random symmetric matrix
    V = randn(n, n)
    # while np.linalg.matrix_rank(V) < n:
    #     V = randn(n, n)

    H = np.diag(randn(n))
    # V_ = np.linalg.inv(V)
    A = V@H@V.T
    d, v = eig(A)
    idx = np.argsort(abs(d))
    d = d[idx[-3:]]
    while norm(d[2]-d[1]) < eps or norm(d[1]-d[0]) < eps:
        A = randn((n, n))
        d, v = eig(A)
        idx = np.argsort(abs(d))
        d = d[idx[-3:]]

    v = v[:, idx[-2:]]
    d = d[1:]
    return A, v, d


#  validate power method algorithms
B, v, d = GetMat(n, eps)
lambda1_gt = d[1]
lambda2_gt = d[0]

v1_gt = v[:, 1]
v2_gt = v[:, 0]

v1, lambda1 = PowerMethod(B, eps*1e-2)
v2, lambda2 = PowerMethod2(B, lambda1=lambda1, v1=v1, eps=eps*1e-2)

assert norm(v1 - v1_gt) < eps or norm(v1 + v1_gt) < eps, "1st eigenvector did not converge"
assert norm(lambda1-lambda1_gt) < eps, "1st eigenvalue did not converge"

assert norm(v2 - v2_gt) < eps or norm(v2 + v2_gt) < eps, "2nd eigenvector did not converge"
assert norm(lambda2-lambda2_gt) < eps, "2nd eigenvalue did not converge"

print("results converge to first two eigenvectors and eigenvalues")

