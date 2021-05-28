from abc import ABC
import numpy as np
import scipy.sparse as sparse
from scipy.linalg import toeplitz
from scipy.linalg import eigh
import networkx as nx
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank as rank
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import block_diag
from numpy.linalg import norm


class Graph(ABC):
    def __init__(self, data: np.array, sigma=0.2, normalized=False):
        """
        :param n: number of vertices
        """

        self.n = data.shape[0]
        self.data = data
        self.sigma = sigma
        self.W = None
        self.L = None
        self.NL = normalized
        self.D = None
        self.L_eigs = None
        self.L_eig_vecs = None
        self.embedding = None

    def weight_matrix(self) -> np.array:
        """
        :return: weight matrix
        """
        if self.W is None:
            euc_dist = squareform(pdist(self.data, 'euclidean'))
            W = np.exp(-euc_dist**2 / (2 * self.sigma**2))
            self.W = W - np.diag(np.diag(W))
        return self.W

    def degree_matrix(self) -> np.array:
        """
        :return: degree matrix
        """
        if self.W is None:
            self.weight_matrix()

        if self.D is None:
            diag = self.W.sum(0)
            self.D = np.diag(diag)
        return self.D

    def laplacian_matrix(self) -> np.array:
        """
        :return: Laplacian matrix
        """
        if self.W is None:
            self.weight_matrix()

        if self.D is None:
            self.degree_matrix()

        if self.L is None:
            self.L = self.D - self.W
            if self.NL:
                D_inv = np.diag(np.diag(self.D)**(-0.5))
                self.L = D_inv @ self.L @ D_inv

        return self.L

    def eigs(self, K=5) -> (np.array, np.array):
        """
        Compute eigenvalues and eigenvectors of the Laplacian matrix
        :return: first K+1 eigenvalues, eigenvectors (np.array, np.array)
        """
        if self.L is None:
            self.L = self.laplacian_matrix()

        if self.L_eigs is None:
            self.L_eigs, self.L_eig_vecs = eigh(self.L)
            idx = np.argsort(self.L_eigs)

            self.L_eigs = self.L_eigs[idx]
            self.L_eig_vecs = self.L_eig_vecs[:, idx]
            self.L_eigs = self.L_eigs[: K + 1]
            self.L_eig_vecs = self.L_eig_vecs[:, : K + 1]

        return self.L_eigs, self.L_eig_vecs

    def calc_embedding(self, K=5) -> np.array:
        """
        Constructing the embeddings of the vertices
        :return: embeddings for the first K+1 eigenvalues, eigenvectors, np.array
        """
        if self.L_eigs is None:
            self.L_eigs, self.L_eig_vecs = self.eigs(K)

        if self.embedding is None:
            if self.NL:
                self.embedding = 1 / self.L_eigs[:-1, None] * self.L_eig_vecs[:, :-1].T
            else:
                self.embedding = 1 / self.L_eigs[1:, None] * self.L_eig_vecs[:, 1:].T

        return self.embedding


