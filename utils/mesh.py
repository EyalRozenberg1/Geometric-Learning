from abc import ABC
from utils.io import read_off

import numpy as np
import pyvista as pv
import scipy.sparse as sparse
import matplotlib.pyplot as plt


class Mesh(ABC):
    """
    Mesh structure with vertices & faces
    """

    def __init__(self, mesh_file: str = None, unit_norm: bool = True):
        """
        Initialize the mesh from a file
        :param mesh_file: (str) '.off' file
        :param unit_norm: (bool) Whether to normalize the computed normals to have an L2 norm of 1.
        """

        if mesh_file is not None:
            data_off = read_off(mesh_file)

            self.v = data_off[0]
            self.f = data_off[1]

        self.unit_norm = unit_norm

        self.Avf = None
        self.Avv = None
        self.vertices_degree = None


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

            self.Avf = sparse.coo_matrix(adjs,shape=[n_vs, n_fcs],dtype=np.bool)

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
            self.Avv = sparse.coo_matrix(Avv,shape=Avv.shape,dtype=np.bool)

        return self.Avv

    def vertex_degree(self) -> np.ndarray:
        """
        returns a numpy vector of size |V| holding the vertex degree per vertex.
        :return: (np.ndarray) shape (|V|, ) -->  each entry i denotes the degree of vertex i
        """

        # compute to initialize
        if self.vertices_degree is None:
            Avv = self.vertex_vertex_adjacency().sum(1)
            self.vertices_degree = np.array(Avv.sum(1)).squeeze()
        return self.vertices_degree


    def _render_mesh(self, typpe: str, scalar_func: str = None,as_spheres: float = False,cmap: str = 'hot') -> None:
        """
        Utility method for rendering meshes

        :param typpe: (str) The rendering typpe to use, should be 'wireframe','points' or 'surface'
        :param scalar_func: (str) specifies the which scalar function to color the mesh
        :param as_spheres: (bool) whether to visualize the vertices as spheres
        :param cmap: (str) Colormap

        :return: None
        """
        vertices = self._get_vertices_array()
        faces = self._get_faces_array()

        if scalar_func is not None:
            colors = None
            if scalar_func == 'degree':
                colors = np.sum(np.array(self.vertex_vertex_adjacency().todense()), 1)

            elif scalar_func == 'coo':
                colors = np.sqrt(np.sum(np.power(vertices, 2), 1))

            elif scalar_func == 'inds':
                colors = np.sum(faces, 1)

            elif scalar_func == 'face_area':
                colors = self.areas

            elif scalar_func == 'curvature':
                colors = -self.gaussian_curvature

            elif scalar_func == 'vertex_area':
                colors = self.barycenters_areas
                colors = np.array([np.mean(colors[faces[f, 1:]]) for f in range(faces.shape[0])])

            if scalar_func in ('degree', 'coo') and typpe == 'surface':
                colors = np.array([np.mean(colors[faces[f, 1:]]) for f in range(faces.shape[0])])
        else:
            colors = None

        mesh = pv.PolyData(vertices, faces)
        color_bar_args = {'fmt': "%.4f"}
        plotter = pv.Plotter()
        if scalar_func == 'curvature':
            plotter.add_mesh(mesh, style=typpe, cmap=cmap, render_points_as_spheres=as_spheres, scalars=colors, scalar_bar_args=color_bar_args, clim=[-100, 200])
        else:
            plotter.add_mesh(mesh, style=typpe, cmap=cmap, render_points_as_spheres=as_spheres, scalars=colors, scalar_bar_args=color_bar_args)

        plotter.add_scalar_bar()
        plotter.show()

    def render_wireframe(self) -> None:
        self._render_mesh(typpe='wireframe', as_spheres=False)

    def render_pointcloud(self, scalar_func: str = 'degree') -> None:
        self._render_mesh(scalar_func=scalar_func, typpe='points', cmap='hot', as_spheres=True)

    def render_surface(self, scalar_func: str = 'inds') -> None:
        self._render_mesh(scalar_func=scalar_func, typpe='surface', cmap='hot')


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


    def render_vertices_normals(self, normalize: bool, add_norms: bool = False, mag: float = 1.) -> None:
        typpe = 'surface' if add_norms else 'wireframe'
        self._render_normals(normalize=normalize, add_norms=add_norms, mag=mag,typpe=typpe, color_by='vertex')

    def _render_normals(self, normalize: bool, typpe: str, add_norms: bool = False, mag: float = 1., color_by: str = 'face'):
        """
        render normals for visualization
        """
        vertices     = self._get_vertices_array()
        faces        = self._get_faces_array()
        mesh         = pv.PolyData(vertices, faces)
        mesh.vectors = vertices

        norm            = self.unit_norm
        self.unit_norm  = normalize
        normals         = self.vertex_normals
        self.unit_norm  = norm

        if add_norms:
            if color_by == 'face':
                colors = self._get_faces_normals()[1]
            elif color_by == 'vertex':
                colors = self._compute_vertex_normals()[1]
                colors = np.array([np.mean(colors[faces[f, 1:]]) for f in range(faces.shape[0])])

            plotter = pv.Plotter()
            plotter.add_mesh(mesh, style=typpe, cmap='hot', scalars=colors)
            plotter.add_arrows(mesh.points, normals, mag=mag)

        else:
            plotter = pv.Plotter()
            plotter.add_mesh(mesh, style=typpe)
            plotter.add_arrows(mesh.points, normals, mag=mag)
        plotter.show()


    def render_get_faces_normals(self, normalize: bool, add_norms: bool = False, mag: float = 1.) -> None:
        """
        visualizing the normal vectors of all faces.
        """
        self._render_normals(normalize=normalize, add_norms=add_norms, mag=mag, typpe='surface', color_by='face')

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
            )), 0)
            for tri in faces_vertices
        ], 0)

        # Heron's formula
        s = np.sum(triangles, 1) / 2

        # area of each face
        areas = np.array([np.sqrt((s[t] * (s[t] - triangles[t, 0]) * (s[t] - triangles[t, 1]) * (s[t] - triangles[t, 2])))
            for t, tri in enumerate(triangles)])
        return areas

    @property
    def areas(self) -> np.ndarray:
        """
        containing the area of each face
        """
        return self._faces_areas()

    @property
    def barycenters_areas(self) -> np.ndarray:
        """
        barycenter area of each vertex in the mesh
        """
        return self._vertices_barycenters_areas()


    @property
    def barycenters(self) -> np.ndarray:
        """
        barycenters of each face
        """
        return self._faces_barycenters()


    def _vertices_barycenters_areas(self) -> np.ndarray:
        """
        computing barycenter area of each vertex
        """
        Avf = self.vertex_face_adjacency()
        faces_areas = self.areas
        return (1 / 3) * Avf * faces_areas

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

    @staticmethod
    def _calc_vertex_angle(main_vertex: np.ndarray, vertices: np.ndarray, main_vertex_ind: int, all_vertices_inds: np.ndarray) -> float:
        """
        computing vertex's angle in given face
        """

        # Compute angle
        other_vertices = np.setdiff1d(np.arange(3),np.where(all_vertices_inds == main_vertex_ind)[0])
        vertex_1 = vertices[other_vertices[0]]
        vertex_2 = vertices[other_vertices[1]]

        a = vertex_1 - main_vertex
        b = vertex_2 - main_vertex
        c = vertex_2 - vertex_1
        size_a = np.sqrt(np.sum(np.power(a, 2)))
        size_b = np.sqrt(np.sum(np.power(b, 2)))
        size_c = np.sqrt(np.sum(np.power(c, 2)))

        angle = np.arccos((((size_a ** 2) + (size_b ** 2) - (size_c ** 2)) /
                           (2 * size_a * size_b)))
        return angle

    def _euler_characteristic(self) -> int:
        """
        compute Euler Characteristic
        :return: (int)
        """
        n_v = len(self.v)
        n_f = len(self.f)
        n_e = self.vertex_vertex_adjacency().todense()
        n_e = int(np.sum([np.sum(n_e[r, r:]) for r in range(n_v)]).item())
        return n_v - n_e + n_f

    @property
    def euler_characteristic(self) -> int:
        """
        containing the Euler Characteristic
        :return: (int)
        """
        return self._euler_characteristic()

    def _compute_gaussian_curvature(self) -> np.ndarray:
        """
        compute the Gaussian curvature
        :return: (np.ndarray)
        """
        Avf = np.array(self.vertex_face_adjacency().todense().astype(np.int))

        n_vertex = len(self.v)
        faces_per_vertex = np.array([np.where(Avf[v, :] == 1)[0] for v in range(n_vertex)])
        vertices_per_vertex = [[np.where(Avf[:, f] == 1)[0] for f in faces]for faces in faces_per_vertex]
        vertices_array = self._get_vertices_array()

        angels_per_vertex = [np.array([self._calc_vertex_angle(main_vertex=self.v[v], vertices=vertices_array[f],
                                                               main_vertex_ind=v, all_vertices_inds=f) for f in
                                       vertices_per_vertex[v]])for v, vertices in enumerate(vertices_per_vertex)]

        # vertices' areas
        Av = self._vertices_barycenters_areas()
        curvatures = ((2 * np.pi) - np.array([np.sum(angles) for angles in angels_per_vertex])) / Av
        return curvatures

    @property
    def gaussian_curvature(self) -> np.ndarray:
        """
        containing Gaussian curvature
        """
        return self._compute_gaussian_curvature()


    def _compute_vertex_centroid(self) -> np.ndarray:
        """
        compute vertices centroid.
        :return: (np.ndarray) Coordinates of vertices centroid
        """
        vertices = self._get_vertices_array()
        centroid = np.mean(vertices, 0)
        return centroid

    @property
    def vertices_centroid(self) -> np.ndarray:
        """
        contain vertices centroid.
        """
        return self._compute_vertex_centroid()

    def distance_from_centroid(self) -> np.ndarray:
        """
        compute Euclidean distance to the vertices centroid.
        :return: (np.ndarray)
        """
        centroid = self.vertices_centroid
        vertices = self._get_vertices_array()
        distances = vertices - np.expand_dims(centroid, 0)
        distances = np.sqrt(np.sum((np.power(distances, 2)), 1))
        return distances

    def render_distance_from_centroid(self) -> None:
        """
        visualizing Euclidean distance to the vertices centroid
        """
        centroid = self.vertices_centroid
        vertices = self._get_vertices_array()
        distances = self.distance_from_centroid()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        im = ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=distances,
                        zorder=0, s=10, alpha=0.2)
        ax.scatter(centroid[0], centroid[1], centroid[2], c='k', zorder=1, s=250,
                   alpha=1.0)
        fig.colorbar(im)
        plt.show()

    def _get_vertices_array(self):
        return np.array(self.v)

    def _get_faces_array(self):
        return np.concatenate((np.expand_dims(len(self.f[0]) * np.ones((len(self.f)), ),1).astype(np.int),np.array(self.f)), 1)

    def _faces_barycenters(self) -> np.ndarray:
        # Get the coordinates
        faces = np.array(self.f)
        faces_vertices = np.array(self.v)
        # Compute the barycenters
        barycenters = np.array([np.mean(np.concatenate([np.expand_dims(np.array(faces_vertices[v]), 0)for v in face], 0), 0)for face in faces])
        return barycenters
