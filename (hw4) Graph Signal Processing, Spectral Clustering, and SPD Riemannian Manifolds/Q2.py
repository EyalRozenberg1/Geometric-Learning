from pathlib import Path
root = Path(Path(__file__).resolve().parents[1])

import matplotlib
matplotlib.use('TkAgg')
import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data_dir = os.path.join(root, 'data', 'rings')
results_dir = os.path.join(root, 'figs', 'Ex4')

Q = "5.g"

if Q == "4.a":
    ring5 = np.load(os.path.join(data_dir, 'ring5.npy')).T
    plt.figure('ring5.py scatter-points', figsize=(7, 5))
    ax = plt.axes(projection='3d')
    ax.scatter(*ring5.T)
    plt.title('scatter-points')
    plt.savefig(os.path.join(results_dir, 'ring5'))


    kmeans = KMeans(n_clusters=5).fit(ring5)
    clusters = kmeans.predict(ring5)
    plt.figure('ring5.py scatter-points with kmeans(k=5)', figsize=(7, 5))
    ax = plt.axes(projection='3d')
    ax.scatter(*ring5.T, c=clusters)
    plt.title('K-means(K=5)')
    plt.savefig(os.path.join(results_dir, 'ring5_Kmeans'))

if Q == "4.c":
    ring5 = np.load(os.path.join(data_dir, 'ring5.npy')).T
    from utils.graphQ4 import Graph
    Ring5 = Graph(data=ring5)
    ring5_embedding = Ring5.calc_embedding().T
    kmeans = KMeans(n_clusters=5).fit(ring5_embedding)

    clusters = kmeans.predict(ring5_embedding)
    plt.figure('ring5.py first 3 embeddings', figsize=(7, 5))
    ax = plt.axes(projection='3d')
    ax.scatter(*ring5_embedding[:, :3].T, c=clusters)
    plt.title('Embeddings Space (first three coordinates) for Spectral Clustering')
    plt.savefig(os.path.join(results_dir, 'ring5_embeddings_space'))

    plt.figure('ring5.py embedding scatter-points with kmeans(k=5)', figsize=(7, 5))
    ax = plt.axes(projection='3d')
    ax.scatter(*ring5.T, c=clusters)
    plt.title('Spectral Clustering')
    plt.savefig(os.path.join(results_dir, 'ring5_spectral_clustering'))

if Q == "5.c":
    ring2 = np.load(os.path.join(data_dir, 'ring2.npy')).T
    plt.figure('ring2.py scatter-points', figsize=(7, 5))
    ax = plt.axes(projection='3d')
    ax.scatter(*ring2.T)
    plt.title('scatter-points')
    plt.savefig(os.path.join(results_dir, 'ring2'))

    kmeans = KMeans(n_clusters=2).fit(ring2)
    clusters = kmeans.predict(ring2)
    plt.figure('ring2.py scatter-points with kmeans(k=2)', figsize=(7, 5))
    ax = plt.axes(projection='3d')
    ax.scatter(*ring2.T, c=clusters)
    plt.title('K-means(K=2)')
    plt.savefig(os.path.join(results_dir, 'ring2_Kmeans'))

    from utils.graphQ4 import Graph
    Ring2 = Graph(data=ring2, sigma=0.05)
    ring2_embedding = Ring2.calc_embedding().T
    kmeans = KMeans(n_clusters=2).fit(ring2_embedding)

    clusters = kmeans.predict(ring2_embedding)
    plt.figure('ring2.py first 2 embeddings', figsize=(7, 5))
    ax = plt.axes(projection='3d')
    ax.scatter(*ring2_embedding[:, :2].T, c=clusters)
    plt.title('Embeddings Space (first two coordinates) for Spectral Clustering')
    plt.savefig(os.path.join(results_dir, 'ring2_embeddings_space'))

    plt.figure('ring2.py embedding scatter-points with kmeans(k=2)', figsize=(7, 5))
    ax = plt.axes(projection='3d')
    ax.scatter(*ring2.T, c=clusters)
    plt.title('Spectral Clustering')
    plt.savefig(os.path.join(results_dir, 'ring2_spectral_clustering'))

if Q == "5.g":
    ring2 = np.load(os.path.join(data_dir, 'ring2.npy')).T
    from utils.graphQ4 import Graph
    Ring2 = Graph(data=ring2, sigma=0.05, normalized=True)
    ring2_embedding = Ring2.calc_embedding().T
    kmeans = KMeans(n_clusters=2).fit(ring2_embedding)

    clusters = kmeans.predict(ring2_embedding)
    plt.figure('ring2.py first 2 embeddings with normalized Laplacian', figsize=(7, 5))
    ax = plt.axes(projection='3d')
    ax.scatter(*ring2_embedding[:, :2].T, c=clusters)
    plt.title('Embeddings Space (first two coordinates) for Spectral Clustering')
    plt.savefig(os.path.join(results_dir, 'ring2_embeddings_space_normL'))

    plt.figure('ring2.py embedding scatter-points with kmeans(k=2)', figsize=(7, 5))
    ax = plt.axes(projection='3d')
    ax.scatter(*ring2.T, c=clusters)
    plt.title('Spectral Clustering')
    plt.savefig(os.path.join(results_dir, 'ring2_spectral_clustering_normL'))
