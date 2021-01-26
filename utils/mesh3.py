from abc import ABC
import time
import numpy as np
import pyvista as pv
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import meshio
import scipy
from scipy.sparse.linalg import eigsh
from scipy.sparse import csc_matrix, csr_matrix

class Mesh(ABC):
    """
    Mesh structure with vertices & faces
    """

    def __init__(self, mesh_file: str = None, file_type: str ='ply', cls: str = 'uniform'):
        """
        Initialize the mesh from a file
        :param mesh_file: (str) '.ply'/'off' file
        """

        if mesh_file is not None:
            data_mesh = meshio.read(mesh_file, file_type)
            self.v = data_mesh.points
            self.f = data_mesh.cells_dict['triangle']

        self.unit_norm = True

        self.Avf = None
        self.Avv = None
        self.W = None
        self.D = None
        self.L = None
        self.M = None
        self.eig_val = None
        self.eig_vec = None
        self.Hn = None
        self.Hn_log = None
        if cls == 'uniform':
            self.cls = cls
        elif cls == 'half_cotangent':
            self.cls = cls
        else:
            assert cls in ['uniform',
                           'half_cotangent'], 'adjacency matrix method (cls) must be in [uniform, half_cotangent]'


    def vertex_face_adjacency(self) -> sparse.coo_matrix:
        """
        returns the sparse Boolean vertex-face adjacency matrix. Aij is true if vertex i is adjacent to face j
        :return: (scipy.sparse.coo_matrix) |V| X |F|
        """

        # compute to initialize
        if self.Avf is None:
            n_vs = len(self.v)
            n_fcs = len(self.f)

            adjs = np.concatenate([[np.array([i in self.f[j] for j in range(n_fcs)])
                                           for i in range(n_vs)]], 0)

            self.Avf = sparse.coo_matrix(adjs, shape=[n_vs, n_fcs], dtype=np.bool)
        return self.Avf


    def vertex_vertex_adjacency(self) -> sparse.coo_matrix:
        """
        returns the sparse Boolean vertex-vertex adjacency matrix. Aij is true if vertex i is adjacent to vertex j
        :return: (scipy.sparse.coo_matrix) |V| X |V|
        """
        # compute to initialize
        if self.Avv is None:
            Avf = self.vertex_face_adjacency().astype(np.int)

            # Note that 2 vertices are adjacent if they share 2 faces
            Avf_sq = Avf @ Avf.T
            inds_1 = Avf_sq <= 2
            inds_2 = Avf_sq > 0
            Avv = np.array((inds_1 - inds_2).todense()) == False
            self.Avv = sparse.coo_matrix(Avv, shape=Avv.shape)
        return self.Avv


    def weighted_adjacency(self) -> sparse.coo_matrix:
        """
        returns the sparse vertex-vertex weighted adjacency matrix.
        :param: cls: 'uniform' - Aij is 1 if vertex i is adjacent to vertex j
                     'half_cotangent' - use the half cotangent weight scheme
        :return: (scipy.sparse.coo_matrix) |V| X |V|
        """

        # compute to initialize
        if self.W is None:
            if self.cls == 'uniform':
                self.W = self.vertex_vertex_adjacency()
            elif self.cls == 'half_cotangent':
                cotangents = self.cotangent
                cotangents = cotangents + cotangents.T
                self.W = sparse.coo_matrix(cotangents / 2)
        return self.W

    def vertex_degree(self) -> sparse.coo_matrix:
        """
        returns a degree matrix (scipy.sparse.coo_matrix) with diagonal |V| holding the vertex degree per vertex.
        V is np.ndarray -->  each entry i denotes the degree of vertex i
        """

        # compute to initialize
        if self.D is None:
            n = len(self.v)
            diag = sparse.coo_matrix(self.weighted_adjacency().sum(0), shape=(1, n))
            self.D = sparse.diags(diag.toarray().squeeze(0), 0)
        return self.D


    def laplacian_matrix(self) -> sparse.coo_matrix:
        """
        :return: Laplacian matrix (scipy.sparse.coo_matrix)
        """
        if self.L is None:
            W = self.weighted_adjacency()
            D = self.vertex_degree()
            self.L = D - W

        return self.L

    @property
    def cotangent(self) -> np.ndarray:
        return self._cotangent()

    def _cotangent(self) -> np.ndarray:
        v = np.array(self.v)
        f = np.array(self.f)
        cotangents = np.zeros((len(v), len(v)))
        for i in range(len(f)):
            indices = f[i, :]
            triangle = v[indices, :]
            for j in range(3):
                cotangents[indices[0], indices[1]] += self.calculate_cotangent(triangle)

                indices = np.roll(indices, 1)
                triangle = v[indices, :]
        # v1 = v[f[:, 0], :]
        # v2 = v[f[:, 1], :]
        # v3 = v[f[:, 2], :]
        # cosine = ((v1 - v3) @ (v2 - v3).T) / np.linalg.norm(v1 - v3, axis=1) / np.linalg.norm(v2 - v3, axis=1)
        # return 1 / np.tan(np.arccos(cosine))
        return cotangents

    @property
    def barycenter_vertex_mass_matrix(self) -> sparse.coo_matrix:
        """
        returns a degree matrix (scipy.sparse.coo_matrix) with diagonal hoolding barycenter area
        of each vertex in the mesh
        """
        if self.M is None:
            self.M = sparse.diags(self._vertices_barycenters_areas(), 0)

        return self.M

    def _vertices_barycenters_areas(self) -> np.ndarray:
        """
        computing barycenter area of each vertex
        """
        Avf = self.vertex_face_adjacency()
        faces_areas = self.areas
        return (1 / 3) * Avf * faces_areas

    @property
    def areas(self) -> np.ndarray:
        """
        containing the area of each face
        """
        return self._faces_areas()

    def _faces_areas(self) -> np.ndarray:
        """
        computing the area of given face (Heron's formula)
        :return: (np.ndarray)
        """
        vertices = np.array(self.v)
        faces = np.array(self.f)
        faces_vertices = [
            (
                vertices[faces[f][0], :],
                vertices[faces[f][1], :],
                vertices[faces[f][2], :],
            )
            for f in range(faces.shape[0])]

        triangles = np.concatenate([
            np.expand_dims(np.array((
                np.sqrt(np.sum(np.power((tri[0] - tri[1]), 2))),
                np.sqrt(np.sum(np.power((tri[0] - tri[2]), 2))),
                np.sqrt(np.sum(np.power((tri[1] - tri[2]), 2))),
            )), 0) for tri in faces_vertices], 0)

        # Heron's formula
        s = np.sum(triangles, 1) / 2

        # area of each face
        areas = np.array([np.sqrt((s[t] * (s[t] - triangles[t, 0]) * (s[t] - triangles[t, 1]) * (s[t] - triangles[t, 2])))
            for t, tri in enumerate(triangles)])
        return areas

    def laplacian_spectrum(self, k: int = 1) -> None:
        """
        Compute eigenvalues and eigenvectors of the Laplacian matrix
        :return: eigenvalues, eigenvectors (np.array, np.array)
        """
        if self.eig_val is None or self.eig_vec is None:
            L = self.laplacian_matrix()
            M = self.barycenter_vertex_mass_matrix
            if self.cls == 'uniform':
                eig_val, eig_vec = eigsh(L, k, M, which='LM', sigma=0, tol=1e-7)
            else:
                eig_val, eig_vec = eigsh(L, k, M, which='LM', sigma=0, tol=1e-7)

            self.eig_val = np.round(eig_val, decimals=12)
            self.eig_vec = np.round(eig_vec, decimals=12)

            idx = np.argsort(self.eig_val)
            self.eig_val = self.eig_val[idx]
            self.eig_vec = self.eig_vec[:, idx]

    def render_mesh(self, k=1, scalar_func='eig', typpe='surface') -> None:
        self._render_mesh(scalar_func=scalar_func, typpe=typpe, cmap='hot', as_spheres=True, k=k)

    def _render_mesh(self, typpe: str, scalar_func: str = None, as_spheres: float = False, cmap: str = 'hot', k: int=1) -> None:
        """
        Utility method for rendering meshes

        :param typpe: (str) The rendering typpe to use, should be 'wireframe','points' or 'surface'
        :param scalar_func: (str) specifies the which scalar function to color the mesh
        :param as_spheres: (bool) whether to visualize the vertices as spheres
        :param cmap: (str) Colormap

        :return: None
        """
        vertices = self.v
        faces = self._get_faces_array()

        if scalar_func is not None:
            if scalar_func == 'eig':
                k = 1 if k < 1 else int(k)
                self.laplacian_spectrum(k=k)

                if k > 1:
                    m = int(np.ceil(k ** 0.5))
                    n = int(np.ceil(k / np.ceil(k ** 0.5)))
                    plotter = pv.Plotter(shape=(m, n))
                    p1, p2 = 0, 0
                    for idx in range(self.eig_vec.shape[1]):
                        if self.eig_vec[0, idx] < 0:
                            self.eig_vec[:, idx] = -self.eig_vec[:, idx]

                        if typpe == 'surface':
                            colors = np.array([np.mean(self.eig_vec[:, idx][faces[f, 1:]]) for f in range(faces.shape[0])])
                        else:
                            colors = self.eig_vec[:, idx]

                        plotter.subplot(p1, p2)
                        p1 = (p1 + 1) % m
                        p2 = (p2 + 1) % n
                        mesh = pv.PolyData(vertices, faces)
                        color_bar_args = {'fmt': "%.4f"}

                        plotter.add_mesh(mesh, style=typpe, cmap=cmap, render_points_as_spheres=as_spheres, scalars=colors,
                                         scalar_bar_args=color_bar_args)


                    # plotter.add_scalar_bar()
                    print(f'eigenvalues: {self.eig_val}')
                    plotter.show()
                else:
                    colors = self.eig_vec
                    mesh = pv.PolyData(vertices, faces)
                    color_bar_args = {'fmt': "%.4f"}
                    plotter = pv.Plotter()

                    plotter.add_mesh(mesh, style=typpe, cmap=cmap, render_points_as_spheres=as_spheres, scalars=colors,
                                     scalar_bar_args=color_bar_args)

                    plotter.add_scalar_bar()
                    plotter.show()

            if scalar_func == 'mean_curvature':
                self.mean_curvature()
                if typpe == 'surface':
                    Hn = np.array([np.mean(self.Hn[faces[f, 1:]]) for f in range(faces.shape[0])])
                    Hn_log = np.array([np.mean(self.Hn_log[faces[f, 1:]]) for f in range(faces.shape[0])])
                else:
                    Hn = self.Hn
                    Hn_log = self.Hn_log
                mesh = pv.PolyData(vertices, faces)
                color_bar_args = {'fmt': "%.4f"}
                # plotter = pv.Plotter()
                plotter = pv.Plotter(shape=(1, 2))
                # plotter = pv.Plotter(shape=(1, 2), off_screen=True)
                plotter.subplot(0, 0)
                plotter.add_mesh(mesh, style=typpe, cmap=cmap, render_points_as_spheres=as_spheres, scalars=Hn, scalar_bar_args=color_bar_args)
                plotter.add_text('Hn: signed mean curvature', font_size=10)
                # plotter.add_scalar_bar()

                plotter.subplot(0, 1)
                plotter.add_mesh(mesh, style=typpe, cmap=cmap, render_points_as_spheres=as_spheres, scalars=Hn_log, scalar_bar_args=color_bar_args)
                plotter.add_text('log(1 + |Hn|)*sign(Hn)', font_size=10)
                # plotter.add_scalar_bar()
                # plotter.screenshot(path)
                plotter.screenshot('test1')
                plotter.show()

            if scalar_func == "scalar_funcs":

                distances = self.distance_from_centroid()
                gaussian_narrow = self.gaussian_centroid(a=10, sig=0.1)
                gaussian_wide = self.gaussian_centroid(a=10, sig=0.3)

                self.laplacian_spectrum(k=50)
                M = self.barycenter_vertex_mass_matrix.toarray()

                f1_k10 = self.eig_vec[:, :10] @ self.eig_vec[:, :10].T @ M @ distances
                f1_k20 = self.eig_vec[:, :20] @ self.eig_vec[:, :20].T @ M @ distances
                f1_k50 = self.eig_vec[:, :50] @ self.eig_vec[:, :50].T @ M @ distances

                f2_k10 = self.eig_vec[:, :10] @ self.eig_vec[:, :10].T @ M @ gaussian_narrow
                f2_k20 = self.eig_vec[:, :20] @ self.eig_vec[:, :20].T @ M @ gaussian_narrow
                f2_k50 = self.eig_vec[:, :50] @ self.eig_vec[:, :50].T @ M @ gaussian_narrow

                f3_k10 = self.eig_vec[:, :10] @ self.eig_vec[:, :10].T @ M @ gaussian_wide
                f3_k20 = self.eig_vec[:, :20] @ self.eig_vec[:, :20].T @ M @ gaussian_wide
                f3_k50 = self.eig_vec[:, :50] @ self.eig_vec[:, :50].T @ M @ gaussian_wide
                if typpe == 'surface':
                    f1 = np.array([np.mean(distances[faces[f, 1:]]) for f in range(faces.shape[0])])
                    f2 = np.array([np.mean(gaussian_narrow[faces[f, 1:]]) for f in range(faces.shape[0])])
                    f3 = np.array([np.mean(gaussian_wide[faces[f, 1:]]) for f in range(faces.shape[0])])

                    f1_k10 = np.array([np.mean(f1_k10[faces[f, 1:]]) for f in range(faces.shape[0])])
                    f1_k20 = np.array([np.mean(f1_k20[faces[f, 1:]]) for f in range(faces.shape[0])])
                    f1_k50 = np.array([np.mean(f1_k50[faces[f, 1:]]) for f in range(faces.shape[0])])

                    f2_k10 = np.array([np.mean(f2_k10[faces[f, 1:]]) for f in range(faces.shape[0])])
                    f2_k20 = np.array([np.mean(f2_k20[faces[f, 1:]]) for f in range(faces.shape[0])])
                    f2_k50 = np.array([np.mean(f2_k50[faces[f, 1:]]) for f in range(faces.shape[0])])

                    f3_k10 = np.array([np.mean(f3_k10[faces[f, 1:]]) for f in range(faces.shape[0])])
                    f3_k20 = np.array([np.mean(f3_k20[faces[f, 1:]]) for f in range(faces.shape[0])])
                    f3_k50 = np.array([np.mean(f3_k50[faces[f, 1:]]) for f in range(faces.shape[0])])

                else:
                    f1 = distances
                    f2 = gaussian_narrow
                    f3 = gaussian_wide

                plotter = pv.Plotter(shape=(3, 4))
                color_bar_args = {'fmt': "%.4f"}

                plotter.subplot(0, 0)
                mesh = pv.PolyData(vertices, faces)
                plotter.add_mesh(mesh, style=typpe, cmap=cmap, render_points_as_spheres=as_spheres,scalars=f1, scalar_bar_args=color_bar_args)
                plotter.subplot(1, 0)
                mesh = pv.PolyData(vertices, faces)
                plotter.add_mesh(mesh, style=typpe, cmap=cmap, render_points_as_spheres=as_spheres, scalars=f2, scalar_bar_args=color_bar_args)
                plotter.subplot(2, 0)
                mesh = pv.PolyData(vertices, faces)
                plotter.add_mesh(mesh, style=typpe, cmap=cmap, render_points_as_spheres=as_spheres, scalars=f3, scalar_bar_args=color_bar_args)

                plotter.subplot(0, 1)
                mesh = pv.PolyData(vertices, faces)
                plotter.add_mesh(mesh, style=typpe, cmap=cmap, render_points_as_spheres=as_spheres, scalars=f1_k10,
                                 scalar_bar_args=color_bar_args)
                plotter.subplot(0, 2)
                mesh = pv.PolyData(vertices, faces)
                plotter.add_mesh(mesh, style=typpe, cmap=cmap, render_points_as_spheres=as_spheres, scalars=f1_k20,
                                 scalar_bar_args=color_bar_args)
                plotter.subplot(0, 3)
                mesh = pv.PolyData(vertices, faces)
                plotter.add_mesh(mesh, style=typpe, cmap=cmap, render_points_as_spheres=as_spheres, scalars=f1_k50,
                                 scalar_bar_args=color_bar_args)

                plotter.subplot(1, 1)
                mesh = pv.PolyData(vertices, faces)
                plotter.add_mesh(mesh, style=typpe, cmap=cmap, render_points_as_spheres=as_spheres, scalars=f2_k10,
                                 scalar_bar_args=color_bar_args)
                plotter.subplot(1, 2)
                mesh = pv.PolyData(vertices, faces)
                plotter.add_mesh(mesh, style=typpe, cmap=cmap, render_points_as_spheres=as_spheres, scalars=f2_k20,
                                 scalar_bar_args=color_bar_args)
                plotter.subplot(1, 3)
                mesh = pv.PolyData(vertices, faces)
                plotter.add_mesh(mesh, style=typpe, cmap=cmap, render_points_as_spheres=as_spheres, scalars=f2_k50,
                                 scalar_bar_args=color_bar_args)

                plotter.subplot(2, 1)
                mesh = pv.PolyData(vertices, faces)
                plotter.add_mesh(mesh, style=typpe, cmap=cmap, render_points_as_spheres=as_spheres, scalars=f3_k10,
                                 scalar_bar_args=color_bar_args)
                plotter.subplot(2, 2)
                mesh = pv.PolyData(vertices, faces)
                plotter.add_mesh(mesh, style=typpe, cmap=cmap, render_points_as_spheres=as_spheres, scalars=f3_k20,
                                 scalar_bar_args=color_bar_args)
                plotter.subplot(2, 3)
                mesh = pv.PolyData(vertices, faces)
                plotter.add_mesh(mesh, style=typpe, cmap=cmap, render_points_as_spheres=as_spheres, scalars=f3_k50,
                                 scalar_bar_args=color_bar_args)

                plotter.show()

            if scalar_func == "area_normalized_laplacian":

                distances = self.distance_from_centroid()  # distances
                gaussian_narrow = self.gaussian_centroid(a=10, sig=0.1)  # narrow gaussian
                gaussian_wide = self.gaussian_centroid(a=10, sig=0.3)  # wide gaussian

                L = self.laplacian_matrix()
                M = self.barycenter_vertex_mass_matrix

                norm_lap = (sparse.linalg.inv(csc_matrix(M)) @ csr_matrix(L)).toarray()

                f1 = norm_lap @ distances
                f1_log = np.log(np.abs(f1))
                f2 = norm_lap @ gaussian_narrow
                f2_log = np.log(np.abs(f2))
                f3 = norm_lap @ gaussian_wide
                f3_log = np.log(np.abs(f3))

                if typpe == 'surface':
                    f1 = np.array([np.mean(f1[faces[f, 1:]]) for f in range(faces.shape[0])])
                    f1_log = np.array([np.mean(f1_log[faces[f, 1:]]) for f in range(faces.shape[0])])
                    f2 = np.array([np.mean(f2[faces[f, 1:]]) for f in range(faces.shape[0])])
                    f2_log = np.array([np.mean(f2_log[faces[f, 1:]]) for f in range(faces.shape[0])])
                    f3 = np.array([np.mean(f3[faces[f, 1:]]) for f in range(faces.shape[0])])
                    f3_log = np.array([np.mean(f3_log[faces[f, 1:]]) for f in range(faces.shape[0])])


                plotter = pv.Plotter(shape=(2, 3))
                color_bar_args = {'fmt': "%.4f"}

                plotter.subplot(0, 0)
                mesh = pv.PolyData(vertices, faces)
                plotter.add_mesh(mesh, style=typpe, cmap=cmap, render_points_as_spheres=as_spheres, scalars=f1,
                                 scalar_bar_args=color_bar_args)
                plotter.subplot(1, 0)
                mesh = pv.PolyData(vertices, faces)
                plotter.add_mesh(mesh, style=typpe, cmap=cmap, render_points_as_spheres=as_spheres, scalars=f1_log,
                                 scalar_bar_args=color_bar_args)

                plotter.subplot(0, 1)
                mesh = pv.PolyData(vertices, faces)
                plotter.add_mesh(mesh, style=typpe, cmap=cmap, render_points_as_spheres=as_spheres, scalars=f2,
                                 scalar_bar_args=color_bar_args)
                plotter.subplot(1, 1)
                mesh = pv.PolyData(vertices, faces)
                plotter.add_mesh(mesh, style=typpe, cmap=cmap, render_points_as_spheres=as_spheres, scalars=f2_log,
                                 scalar_bar_args=color_bar_args)

                plotter.subplot(0, 2)
                mesh = pv.PolyData(vertices, faces)
                plotter.add_mesh(mesh, style=typpe, cmap=cmap, render_points_as_spheres=as_spheres, scalars=f3,
                                 scalar_bar_args=color_bar_args)
                plotter.subplot(1, 2)
                mesh = pv.PolyData(vertices, faces)
                plotter.add_mesh(mesh, style=typpe, cmap=cmap, render_points_as_spheres=as_spheres, scalars=f3_log,
                                 scalar_bar_args=color_bar_args)
                plotter.show()

            if scalar_func == 'hks':
                self.laplacian_spectrum(k=1000)

                plotter = pv.Plotter(shape=(2, 5))

                color_bar_args = {'fmt': "%.4f"}
                t_vec = self.expspace(start=1e-3, stop=5, n=10)
                print(f'time steps {t_vec}')
                for i, t in enumerate(t_vec):
                    # hks50 = self.hks(t=t, k=50)
                    # hks100 = self.hks(t=t, k=100)
                    hks1000 = self.hks(t=t, k=1000)

                    if typpe == 'surface':
                        # hks50 = np.array([np.mean(hks50[faces[f, 1:]]) for f in range(faces.shape[0])])
                        # hks100 = np.array([np.mean(hks100[faces[f, 1:]]) for f in range(faces.shape[0])])
                        hks1000 = np.array([np.mean(hks1000[faces[f, 1:]]) for f in range(faces.shape[0])])

                    # plotter.subplot(0, i)
                    # plotter.add_text(f'k={50}', font_size=10)
                    # mesh = pv.PolyData(vertices, faces)
                    # plotter.add_mesh(mesh, style=typpe, cmap=cmap, render_points_as_spheres=as_spheres, scalars=hks50,
                    #                  scalar_bar_args=color_bar_args)
                    # plotter.subplot(1, i)
                    # plotter.add_text(f'k={1000}', font_size=10)
                    # mesh = pv.PolyData(vertices, faces)
                    # plotter.add_mesh(mesh, style=typpe, cmap=cmap, render_points_as_spheres=as_spheres, scalars=hks100,
                    #                  scalar_bar_args=color_bar_args)
                    plotter.subplot(i//5, i%5)
                    plotter.add_text(f't{i+1}', font_size=10)
                    mesh = pv.PolyData(vertices, faces)
                    plotter.add_mesh(mesh, style=typpe, cmap=cmap, render_points_as_spheres=as_spheres, scalars=hks1000,
                                     scalar_bar_args=color_bar_args)

                plotter.show()

        else:
            colors = None
            mesh = pv.PolyData(vertices, faces)
            color_bar_args = {'fmt': "%.4f"}
            plotter = pv.Plotter()

            plotter.add_mesh(mesh, style=typpe, cmap=cmap, render_points_as_spheres=as_spheres, scalars=colors, scalar_bar_args=color_bar_args)

            plotter.add_scalar_bar()
            plotter.show()

    def _get_faces_array(self):
        return np.concatenate((np.expand_dims(len(self.f[0]) * np.ones((len(self.f)), ),1).astype(np.int),np.array(self.f)), 1)

    def calculate_cotangent(self, triangle):
        a = triangle[0, :]
        b = triangle[1, :]
        c = triangle[2, :]
        u = a - c
        v = b - c
        cosine = u @ v.T / np.linalg.norm(u) / np.linalg.norm(v)
        cotangent = 1/np.tan(np.arccos(cosine))
        return cotangent

    @property
    def vertex_normals(self) -> np.ndarray:
        return self._compute_vertex_normals()[0]

    def _compute_vertex_normals(self) -> (np.ndarray, np.ndarray):
        """
        computing the vertices normals.
        :return: (np.ndarray) Tuple containing an array of normal of given vertex
        """
        Af = self.areas
        normalize = False
        if self.unit_norm:
            self.unit_norm = False
            normalize = True

        normals = self.normals
        Avf = self.vertex_face_adjacency()
        weighted_normals = np.expand_dims(Af, 1) * normals
        vertices_normals = Avf @ weighted_normals
        norms = np.array([])
        if normalize:
            self.unit_norm = True
            norms = np.linalg.norm(vertices_normals, axis=1)
            vertices_normals = vertices_normals / np.expand_dims(norms, 1)
        return vertices_normals, norms

    @property
    def normals(self) -> np.ndarray:
        """
        :return: (np.ndarray) with the computed normals
        """
        return self._get_faces_normals()[0]

    def _get_faces_normals(self) -> (np.ndarray, np.ndarray):
        """
        :return: (np.ndarray, np.ndarray) Tuple the computed normals
                 np.ndarray containing the computed norms
        """
        v = np.array(self.v)
        f = np.array(self.f)

        # Compute the normal using cross product
        v1 = v[f[:, 0], :]
        v2 = v[f[:, 1], :]
        v3 = v[f[:, 2], :]
        normals = np.cross((v2 - v1), (v3 - v1))
        norms = np.array([])
        if self.unit_norm:
            normals, norms = self._normalize_vector(normals)
        return normals, norms

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        normalizing an input vector wrt L2-norm
        """
        norms = np.linalg.norm(vector, axis=1, keepdims=True)
        vector = vector / norms
        return vector, norms

    def distance_from_centroid(self) -> np.ndarray:
        """
        compute Euclidean distance to the vertices centroid.
        :return: (np.ndarray)
        """
        centroid = self.vertices_centroid
        vertices = self.v
        distances = vertices - np.expand_dims(centroid, 0)
        distances = np.sqrt(np.sum((np.power(distances, 2)), 1))
        return distances

    def gaussian_centroid(self, a=10, sig=0.2) -> np.ndarray:
        """
        compute Euclidean distance to the vertices centroid.
        :return: (np.ndarray)
        """
        distances = self.distance_from_centroid()
        gaussian = a*np.e**(-distances/sig**2)
        return gaussian


    def _compute_vertex_centroid(self) -> np.ndarray:
        """
        compute vertices centroid.
        :return: (np.ndarray) Coordinates of vertices centroid
        """
        vertices = self.v
        centroid = np.mean(vertices, 0)
        return centroid

    @property
    def vertices_centroid(self) -> np.ndarray:
        """
        contain vertices centroid.
        """
        return self._compute_vertex_centroid()

    def gps_desc(self):
        """ gps descriptor """
        self.laplacian_spectrum(k=1000)
        f_gps = 1/np.sqrt(self.eig_val[1:]) * self.eig_vec[:, 1:]
        return f_gps

    def dna_desc(self):
        """ dna descriptor """
        self.laplacian_spectrum(k=1000)
        f_dna = self.eig_val
        return f_dna

    def hks(self, t, k):
        """ heat kernel signature """
        exp_eig = np.diag(np.exp(-self.eig_val[:k] * t))
        hks = np.diag(self.eig_vec[:, :k] @ exp_eig @ self.eig_vec[:, :k].T)
        return hks

    # def hks_desc2(self, t=10):
    #     self.laplacian_spectrum(k=1000)
    #     f_hks5 = self.hks(t=t, k=5)
    #     f_hks50 = self.hks(t=t, k=50)
    #     f_hks200 = self.hks(t=t, k=200)
    #     f_hks1000 = self.hks(t=t, k=1000)
    #     return f_hks5, f_hks50, f_hks200, f_hks1000

    def hks_desc(self, start=1e-3, stop=10, n=10):
        """ hks descriptor """
        self.laplacian_spectrum(k=1000)
        t_vec = self.expspace(start, stop, n)
        f_hks5 = np.array([self.hks(t=t, k=5) for t in t_vec])
        f_hks50 = np.array([self.hks(t=t, k=50) for t in t_vec])
        f_hks200 = np.array([self.hks(t=t, k=200) for t in t_vec])
        f_hks1000 = np.array([self.hks(t=t, k=1000) for t in t_vec])
        return f_hks5, f_hks50, f_hks200, f_hks1000

    def shks_desc(self, start=1e-3, stop=10, n=10):
        self.laplacian_spectrum(k=1000)
        t_vec = self.expspace(start, stop, n)
        f_heat_trace = np.exp(-self.eig_val * stop)
        f_hks5 = np.array([self.hks(t=t, k=5) for t in t_vec]) / np.sum(f_heat_trace[:5])
        f_hks50 = np.array([self.hks(t=t, k=50) for t in t_vec]) / np.sum(f_heat_trace[:50])
        f_hks200 = np.array([self.hks(t=t, k=200) for t in t_vec]) / np.sum(f_heat_trace[:200])
        f_hks1000 = np.array([self.hks(t=t, k=1000) for t in t_vec]) / np.sum(f_heat_trace)
        return f_hks5, f_hks50, f_hks200, f_hks1000

    @staticmethod
    def expspace(start=1e-3, stop=5, n=10):
        return np.exp(np.linspace(np.log(start), np.log(stop), n))

    def mean_curvature(self):
        if self.Hn is None or self.Hn_log is None:
            M = self.barycenter_vertex_mass_matrix
            L = self.laplacian_matrix()
            V = self.v

            Hn0 = (sparse.linalg.inv(csc_matrix(M)) @ csr_matrix(L) @ csr_matrix(V)).toarray()
            Hn_unsigned = np.linalg.norm(Hn0, axis=1)
            normals = self.vertex_normals
            Hn_sign = np.sign((Hn0 * normals).sum(1))
            self.Hn = Hn_unsigned * Hn_sign
            self.Hn_log = np.log(1 + Hn_unsigned) * Hn_sign

    @property
    def mcurv_desc(self) -> np.ndarray:
        self.mean_curvature()
        return self.Hn
