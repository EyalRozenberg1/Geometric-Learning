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
    def __init__(self, type: str = None, n: int = 201, nx: int = 201, ny: int = 31, pertube=False):
        """
        :param type: graph type, Pn- path graph, Rn- ring graph, RnXRn-ring graph product
        :param n: number of vertices
        """
        assert type in ['Pn', 'Rn', 'RnXRn'], "graph type should be 'Pn', 'Rn' or RnXRn"

        self.graph_type = type
        self.n = n

        if type == 'RnXRn':
            self.nx = nx
            self.ny = ny
            self.n = nx*ny
            self.pertube = pertube
            self.Rnx = Graph('Rn', self.nx)
            self.Rny = Graph('Rn', self.ny)
            self.nodes_xyz = None
            self.V = None
            self.E = None

        self.M = None
        self.L = None
        self.D = None
        self.L_eigs = None
        self.L_eig_vecs = None
        self.eigs_anal = None
        self.eig_vecs_anal = None

    def adjacency_matrix(self) -> sparse.coo_matrix:
        """
        :return: adjacency matrix (scipy.sparse.coo_matrix)
        """
        n = self.n

        if self.M is None:
            if self.graph_type == "Pn":  # path graph
                row = np.zeros(n, np.int)
                row[1] = True
                self.M = sparse.coo_matrix(toeplitz(row), shape=[n, n], dtype=np.int)

            elif self.graph_type == "Rn":  # ring graph
                row = np.zeros(n, np.int)
                row[1] = True
                row[-1] = True
                self.M = sparse.coo_matrix(toeplitz(row), shape=[n, n], dtype=np.int)
            elif self.graph_type == "RnXRn":
                if self.pertube:
                    if self.nodes_xyz is None:
                        self.nodes()
                    dist_mat = squareform(pdist(self.nodes_xyz, 'euclidean'))
                    M = np.zeros(dist_mat.shape)
                    # M[dist_mat < 0.18] = True
                    max_val = dist_mat.max()
                    for m in np.arange(M.shape[0]):
                        k_idx = np.argpartition(dist_mat[m, :], 8)
                        dist_mat[m, k_idx[:8]] = 0
                        dist_mat[k_idx[:8], m] = 0

                        dist_mat[m, k_idx[8:]] = max_val
                        dist_mat[k_idx[8:], m] = max_val

                        M[m, k_idx[:8]] = True
                        M[k_idx[:8], m] = True

                        M[m, k_idx[8:]] = False
                        M[k_idx[8:], m] = False

                    M = M - np.diag(np.diag(M))
                    self.M = sparse.coo_matrix(M, shape=[n, n], dtype=np.int)
                else:
                    Mx = self.Rnx.adjacency_matrix()
                    My = self.Rny.adjacency_matrix()
                    Ix = np.eye(self.nx, self.nx)
                    Iy = np.eye(self.ny, self.ny)

                    M = np.kron(Mx.toarray(), Iy) + np.kron(Ix, My.toarray())
                    self.M = sparse.coo_matrix(M, shape=[n, n], dtype=np.int)

        return self.M

    def degree_matrix(self) -> sparse.coo_matrix:
        """
        :return: degree matrix (scipy.sparse.coo_matrix)
        """
        if self.M is None:
            self.adjacency_matrix()

        if self.D is None:
            n = self.n

            diag = sparse.coo_matrix(self.M.sum(0), shape=(1, n), dtype=np.int)
            self.D = sparse.diags(diag.toarray().squeeze(0), 0)
        return self.D

    def laplacian_matrix(self) -> sparse.coo_matrix:
        """
        :return: Laplacian matrix (scipy.sparse.coo_matrix)
        """
        if self.M is None:
            self.adjacency_matrix()

        if self.D is None:
            self.degree_matrix()

        if self.L is None:

            self.L = self.D - self.M

        return self.L

    def eigs(self) -> (np.array, np.array):
        """
        Compute eigenvalues and eigenvectors of the Laplacian matrix
        :return: eigenvalues, eigenvectors (np.array, np.array)
        """
        if self.L is None:
            self.laplacian_matrix()

        if self.L_eigs is None:
            self.L_eigs, self.L_eig_vecs = eigh(self.L.toarray())
            idx = np.argsort(self.L_eigs)

            self.L_eigs     = self.L_eigs[idx]
            self.L_eig_vecs = self.L_eig_vecs[:, idx]

        if self.eigs_anal is None:
            if self.graph_type == 'Pn':  # path graph
                k = np.arange(0, self.n)
                a = np.array([np.arange(1, self.n + 1)]).transpose()
                self.eigs_anal = 2 * (1 - np.cos(np.pi * k / self.n))
                k = np.array([k])
                self.eig_vecs_anal = np.cos(np.pi * np.matmul(a, k) / self.n - 0.5 * np.pi * k / self.n)
            elif self.graph_type == 'Rn':  # ring graph
                k = np.array([np.arange(0, self.n//2 + 1)])
                a = np.array([np.arange(0, self.n)]).transpose()
                self.eigs_anal = 2 * (1 - np.cos(2 * np.pi * k[:, 1:] / self.n))
                self.eigs_anal = np.concatenate((np.zeros(1), np.repeat(self.eigs_anal, 2)))
                self.eig_vecs_anal = np.zeros((self.n, self.n))
                self.eig_vecs_anal[:, np.arange(0, self.n, 2)] = np.cos(2 * np.pi * np.matmul(a, k) / self.n)
                self.eig_vecs_anal[:, np.arange(1, self.n, 2)] = np.sin(2 * np.pi * np.matmul(a, k[:, 1:]) / self.n)

            elif self.graph_type == 'RnXRn':  # ring cross product
                self.Rnx.eigs()
                self.Rny.eigs()
                self.eigs_anal = np.repeat(self.Rnx.eigs_anal, self.ny) + np.tile(self.Rny.eigs_anal, self.nx)
                idx = np.argsort(self.eigs_anal)
                self.eigs_anal = self.eigs_anal[idx]
                self.eig_vecs_anal = np.kron(self.Rnx.eig_vecs_anal, self.Rny.eig_vecs_anal)[:, idx]
        return self.L_eigs, self.L_eig_vecs

    def eigs_compare_analytic(self) -> None:
        """
        Compute with analytic eigenvalues and eigenvectors of the Laplacian matrix
        :return: None
        """

        if self.L_eigs is None:
            self.eigs()

        self.plot_eigenvalues(analytic=True)

        print(f'Comparing eigenspace for {self.graph_type}')
        mse_eigs = self.mse(self.L_eigs, self.eigs_anal)
        print(f'norm(Sig-Sig_analytic)/len(Sig)) = {mse_eigs}')
        self.compare_eigenspaces(self.L_eig_vecs, self.eig_vecs_anal, self.eigs_anal)
        print(f'Increasing order of calculated eigenvectors and analytical eigenvectors span the same space')

    def plot_graph_topology(self, options: dict = None, weights: bool = False, node_color: str = 'blue',
                                eigen_vecs: list = [1]):
        if self.M is None:
            self.adjacency_matrix()
        G = nx.from_numpy_matrix(self.M.toarray())

        if self.graph_type == 'RnXRn':

            if self.V is None:
                pos = nx.kamada_kawai_layout(G, dim=3)
                self.V = np.array([pos[v] for v in sorted(G)])
                self.E = np.array([(pos[u], pos[v]) for u, v in G.edges()])

            if self.pertube:
                edges = np.where(self.M.toarray())
                self.E = np.array([(self.V[edges[0][e]], self.V[edges[1][e]]) for e in range(len(edges[0]))])

            if node_color == 'eigenvectors':
                if self.L_eig_vecs is None:
                    self.eigs()
                # vmin = self.L_eig_vecs.min()
                # vmax = self.L_eig_vecs.max()
                for vidx in eigen_vecs:
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    # if self.pertube:
                    #     ax.scatter(*self.V.T, s=10, linewidths=0.3, c=np.single(self.L_eig_vecs[:, vidx]))
                    # else:
                    #     ax.scatter(*self.V.T, s=10, linewidths=0.3, c=self.L_eig_vecs[:, vidx], vmin=vmin, vmax=vmax)
                    ax.scatter(*self.V.T, s=10, linewidths=0.3, c=np.single(self.L_eig_vecs[:, vidx]))
                    ax.view_init(elev=50, azim=35)
                    ax.set_title(f'Product-Graph Topology, eigenvector {vidx+1} ')

                    for edge in self.E:
                        ax.plot(*edge.T, color="grey", linewidth=1)
                    plt.show()
            else:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(*self.V.T, s=10, linewidths=0.3, edgecolors='k', )
                ax.view_init(elev=50, azim=35)
                ax.set_title('Product-Graph Topology')

                for edge in self.E:
                    ax.plot(*edge.T, color="grey", linewidth=1)
                plt.show()

            ##########################################
            # pos = nx.kamada_kawai_layout(G, dim=3)
            # xyz = np.array([pos[v] for v in sorted(G)])
            # pts = mlab.points3d(
            #     xyz[:, 0],
            #     xyz[:, 1],
            #     xyz[:, 2],
            #     scale_factor=0.1,
            #     scale_mode="none",
            #     colormap="Blues",
            #     resolution=1,
            # )
            # pts.mlab_source.dataset.lines = np.array(list(G.edges()))
            # tube = mlab.pipeline.tube(pts, tube_radius=0.01)
            # mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8))
            # mlab.show()

        else:
            layout = nx.kamada_kawai_layout(G)
            if node_color == 'eigenvectors':
                if self.L_eig_vecs is None:
                    self.eigs()
                # vmin = self.L_eig_vecs.min()
                # vmax = self.L_eig_vecs.max()
                for vidx in eigen_vecs:
                    plt.title(f'{self.graph_type} Topology, eigenvector {vidx+1}')
                    nx.draw(G, layout, **options, node_color=np.single(self.L_eig_vecs[:, vidx]))
                    # nx.draw(G, layout, **options, node_color=self.L_eig_vecs[:, vidx], vmin=vmin, vmax=vmax)
                    if weights:
                        nx.draw_networkx_edge_labels(G, pos=layout)
                    plt.show()
            else:
                plt.title(f'{self.graph_type} Topology')
                nx.draw(G, layout, **options)
                if weights:
                    nx.draw_networkx_edge_labels(G, pos=layout)
                plt.show()


    def plot_eigenvalues(self, analytic=False):
        if self.L_eigs is None:
            self.eigs()

        k = np.arange(0, self.n)
        if analytic:
            ax = plt.subplot(121)
            ax.plot(k, self.eigs_anal)
            ax.set_title(f'sorted analytic eigenvalues of L_{self.graph_type}')
            ax.set_xlabel('k')

            ax = plt.subplot(122)
            ax.plot(k, self.eigs_anal)
            ax.set_title(f'sorted eigenvalues of L_{self.graph_type}')
            ax.set_xlabel('k')
        else:
            plt.plot(k, self.L_eigs)
            plt.title(f'sorted eigenvalues of L_{self.graph_type}')
            plt.xlabel('k')

        plt.show()

    def nodes(self, scatter=False) -> np.array:
        """
        :return: vertex array, in R^3
        """
        assert self.graph_type == 'RnXRn', 'Graph.nodes() is a function of graph product only'

        if self.nodes_xyz is None:
            G1 = nx.from_numpy_matrix(self.Rnx.adjacency_matrix().toarray())
            G2 = nx.from_numpy_matrix(self.Rny.adjacency_matrix().toarray())

            if scatter:
                # use cross product
                nodes1 = np.zeros((self.nx, 3))
                nodes2 = np.zeros((self.ny, 3))
                nodes1[:, 1:] = np.array(list(nx.kamada_kawai_layout(G1).values()))
                nodes2[:, 0:2] = np.array(list(nx.kamada_kawai_layout(G2).values()))
                nodes_xyz_cross = np.cross(np.repeat(nodes1, self.ny, 0), np.tile(nodes2, (self.nx, 1)))

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.set_title('Scatter Nodes Using Cross-Product ')
                ax.scatter(*nodes_xyz_cross.T, s=10, linewidths=0.3, edgecolors='k',)
                ax.view_init(elev=50, azim=35)
                plt.show()

            # Generate nodes using networkx package
            G = nx.cartesian_product(G1, G2)
            pos = nx.kamada_kawai_layout(G, dim=3)
            self.nodes_xyz = np.array([pos[v] for v in sorted(G)])
            # self.E = np.array([(pos[u], pos[v]) for u, v in G.edges()])

            if self.pertube:
                # generate permuted and noisy version
                idx_permute = np.random.permutation(self.n)
                self.nodes_xyz = self.nodes_xyz[idx_permute] + 0.1 * np.random.randn(*(self.nodes_xyz.shape))
                self.V = self.nodes_xyz
                hash = {k: v for k, v in zip(sorted(G), np.arange(0, len(sorted(G))))}
                self.E = self.E = np.array([(self.V[hash[u]], self.V[hash[v]]) for u, v in G.edges()])

        return self.nodes_xyz

    @staticmethod
    def mse(x, y):
        mse = np.sum((x - y)**2 / x.shape[0], 0)
        return mse

    @staticmethod
    def compare_eigenspaces(V1, V2, d):
        V1 = np.single(V1)
        V2 = np.single(V2)
        d = np.single(d)
        eigenvalue = d[0]
        eigenspace = np.concatenate((V1[:, 0][:, None], V2[:, 0][:, None]), 1)
        for i in np.arange(1, d.shape[0]):
            if d[i] == eigenvalue:
                eigenspace = np.concatenate((eigenspace, V1[:, i][:, None], V2[:, i][:, None]), 1)
            else:
                assert (rank(eigenspace) == eigenspace.shape[1] / 2), "subspaces are not equal"
                eigenvalue = d[i]
                eigenspace = np.concatenate((V1[:, i][:, None], V2[:, i][:, None]), 1)
        assert (rank(eigenspace) == eigenspace.shape[1] / 2), "subspaces are not equal"



class DmbBolGraph(ABC):
    def __init__(self, type: str = None, n: int = 30):
        """
        :param type: graph type - dumbbell/bolas
        :param n: complete graphs with number of vertices
        """
        assert type in ['dumbbell', 'bolas'], "graph type should be dumbbell/bolas"

        self.graph_type = type
        self.n_complete = n

        if self.graph_type == "dumbbell":
            self.n = 2 * n

        elif self.graph_type == "bolas":
            self.n = 3 * n - 1

        self.M = None
        self.M_complete = None
        self.N = None
        self.W = None
        self.D = None
        self.N_eigs = None
        self.N_eig_vecs = None
        self.W_eigs = None
        self.W_eig_vecs = None

    def complete_graph_adjacency_matrix(self) -> sparse.coo_matrix:
        """
        :return: adjacency matrix (scipy.sparse.coo_matrix)
        """
        n = self.n_complete

        if self.M_complete is None:
            M = np.ones((n, n))
            self.M_complete = sparse.coo_matrix(M - np.diag(np.diag(M)), shape=[n, n], dtype=np.int)
        return self.M_complete

    def adjacency_matrix(self) -> sparse.coo_matrix:
        """
        :return: adjacency matrix (scipy.sparse.coo_matrix)
        """
        n = self.n

        if self.M_complete is None:
            self.complete_graph_adjacency_matrix()

        if self.M is None:
            if self.graph_type == "dumbbell":
                M = block_diag(self.M_complete.toarray(), self.M_complete.toarray())
                M[self.n_complete-1, self.n_complete] = 1
                M[self.n_complete, self.n_complete-1] = 1

                self.M = sparse.coo_matrix(M, shape=[n, n], dtype=np.int)

            elif self.graph_type == "bolas":

                M_dumbbell = block_diag(self.M_complete.toarray(), self.M_complete.toarray())
                M_dumbbell[self.n_complete - 1, self.n_complete] = 1
                M_dumbbell[self.n_complete, self.n_complete - 1] = 1

                column = np.zeros((n, 1))
                column[0] = 1
                row = column.copy().T
                row[0, 2] = 1
                diag = toeplitz(column, row)
                diag = diag[self.n_complete-1: -self.n_complete-1, :]

                M = np.zeros((n, n))
                M[0:self.n_complete,0:self.n_complete+1] = M_dumbbell[0:self.n_complete,0: self.n_complete+1]
                M[self.n_complete: -self.n_complete,:] = diag
                M[-self.n_complete:, -self.n_complete-1:] = M_dumbbell[-self.n_complete:, -self.n_complete-1:]

                self.M = sparse.coo_matrix(M, shape=[n, n], dtype=np.int)
        return self.M

    def degree_matrix(self) -> sparse.coo_matrix:
        """
        :return: degree matrix (scipy.sparse.coo_matrix)
        """
        if self.M is None:
            self.adjacency_matrix()

        if self.D is None:
            n = self.n
            diag = sparse.coo_matrix(self.M.sum(0), shape=(1, n), dtype=np.int)
            self.D = sparse.diags(diag.toarray().squeeze(0), 0)
        return self.D

    def nornalized_laplacian_matrix(self) -> sparse.coo_matrix:
        """
        :return: Normalized Laplacian matrix (scipy.sparse.coo_matrix)
        """

        if self.D is None:
            self.degree_matrix()

        if self.N is None:
            n = self.n
            I = sparse.coo_matrix(np.diag(np.ones(n)), shape=[n, n], dtype=np.int)
            inv_sqrt_D = sparse.diags(np.diag(self.D.toarray())**(-0.5), 0)
            self.N = I - inv_sqrt_D @ self.M @ inv_sqrt_D
        return self.N
    
    def lazy_random_walk_matrix(self) -> sparse.coo_matrix:
        """
        :return: lazy random walk matrix (scipy.sparse.coo_matrix)
        """

        if self.D is None:
            self.degree_matrix()

        if self.W is None:
            n = self.n
            I = sparse.coo_matrix(np.diag(np.ones(n)), shape=[n, n], dtype=np.int)
            inv_D = sparse.diags(np.diag(self.D.toarray())**(-1), 0)
            self.W = 0.5 * (I + self.M @ inv_D)
        return self.W


    def eigs_laplacian(self) -> (np.array, np.array):
        """
        Compute eigenvalues and eigenvectors of the Laplacian matrix
        :return: eigenvalues, eigenvectors (np.array, np.array)
        """
        if self.N is None:
            self.nornalized_laplacian_matrix()

        if self.N_eigs is None:
            self.N_eigs, self.N_eig_vecs = eigh(self.N.toarray())
            idx = np.argsort(self.N_eigs)

            self.N_eigs     = self.N_eigs[idx]
            self.N_eig_vecs = self.N_eig_vecs[:, idx]

        return self.N_eigs, self.N_eig_vecs

    def eigs_rand_walk(self) -> (np.array, np.array):
        """
        Compute eigenvalues and eigenvectors of the random walk matrix
        :return: eigenvalues, eigenvectors (np.array, np.array)
        """
        if self.W is None:
            self.lazy_random_walk_matrix()

        if self.W_eigs is None:
            self.W_eigs, self.W_eig_vecs = eigh(self.W.toarray())
            idx = np.argsort(self.W_eigs)

            self.W_eigs     = self.W_eigs[idx]
            self.W_eig_vecs = self.W_eig_vecs[:, idx]

        return self.W_eigs, self.W_eig_vecs

    def plot_graph_topology(self, options: dict = None, p=None, time_step=[2,5,8]):
        if self.M is None:
            self.adjacency_matrix()
        G = nx.from_numpy_matrix(self.M.toarray())
        layout = nx.kamada_kawai_layout(G)

        if p is not None:
            for pidx in time_step:
                plt.title(f'{self.graph_type} Topology, time-step t={pidx + 1}')
                nx.draw(G, layout, **options, node_color=np.single(p[:,pidx]))
                plt.show()
        else:
            plt.title(f'{self.graph_type} Topology')
            nx.draw(G, layout, **options)
            plt.show()


    def plot_eigenvalues(self):
        if self.N_eigs is None:
            self.eigs_laplacian()

        if self.W_eigs is None:
            self.eigs_rand_walk()

        k = np.arange(0, self.n)

        plt.plot(k, self.N_eigs, 'go-', linewidth=2, markersize=5)
        plt.text(k[-1], self.N_eigs.max(), str(round(self.N_eigs.max(), 2)), fontsize=12)
        plt.title(f'Normalized Laplacian eigenvalues - {self.graph_type}')
        plt.xlabel('k')
        plt.show()

        plt.plot(k, self.W_eigs, 'go-', linewidth=2, markersize=5)
        plt.text(k[-1], self.W_eigs.max(), str(round(self.W_eigs.max(), 2)), fontsize=12)
        plt.plot(k, sorted(1-0.5*self.N_eigs), 'ro--', linewidth=1, markersize=2)
        plt.title(f'Lazy Random-Walk eigenvalues - {self.graph_type}')
        plt.legend((r'$eig(W_G)$', r'$1-0.5*eig(N_G) [sorted]$'))
        plt.xlabel('k')
        plt.show()


    def initial_mass_distribution(self):
        i_init = np.random.randint(self.n)
        p_init = np.zeros((self.n, 1))
        p_init[i_init, 0] = 1
        return p_init, i_init


    def Run_lazy_walk(self, eps=1e-5) -> (np.array, int):
        p_init, i_init = self.initial_mass_distribution()
        p = p_init
        if self.W is None:
            self.lazy_random_walk_matrix()
        W = self.W.toarray()
        p_step = W @ p
        p_curr = p
        while norm(p_step - p_curr)/self.n >= eps:
            p = np.concatenate((p, p_step), 1)
            p_curr = p_step
            p_step = W @ p_curr
        return p, i_init

    def upper_bound(self, p, i0):
        if self.D is None:
            self.degree_matrix()

        d = np.diag(self.D.toarray())
        pi = d/d.sum()
        diff = norm(p-pi[:, None], axis=0)
        plt.semilogy(diff)
        # plt.xlabel(r'$t$')
        # plt.ylabel(r'$\log\left\Vert p_t - \pi\right\Vert_2$')
        # plt.title(r'$L_2$ convergence')
        # plt.show()
        y0 = r'$\log\left\Vert p_t - \pi\right\Vert_2$'


        if self.W_eigs is None:
            self.eigs_rand_walk()
        diff = norm(p - pi[:, None], ord=np.inf, axis=0)
        w2 = self.W_eigs[-2]
        t = np.arange(0, p.shape[1])
        upper_bound = np.sqrt(max(d)/d[i0]) * w2**t
        plt.semilogy(diff)
        plt.semilogy(upper_bound)
        plt.xlabel(r'$t$')
        y1 = r'$\log\left\Vertp_t - \pi\right\Vert_\infty$'
        y2 = r'$\log\sqrt{\frac{\max\left(d\right)}{d\left[i_0\right]}}\omega^t_2$'
        plt.legend((y0, y1, y2))
        plt.title(r'$L_2 & L_\infty$ convergence')
        plt.show()
