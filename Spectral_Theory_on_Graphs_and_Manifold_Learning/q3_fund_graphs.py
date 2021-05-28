from utils.graph import Graph

section = 1
if section == 1:
    n = 201
    Pn = Graph('Pn', n)
    Rn = Graph('Rn', n)

    options_top = {
        'node_size': 5,
        'width': 0.5,
        'with_labels': True,
        'font_weight': 'normal',
        'font_size': 5}

    Pn.plot_graph_topology(options=options_top)
    Rn.plot_graph_topology(options=options_top)

    Pn.plot_eigenvalues()
    Rn.plot_eigenvalues()

    options_top['node_size'] = 100
    options_top['with_labels'] = False
    Pn.plot_graph_topology(options=options_top, node_color='eigenvectors', eigen_vecs=[0, 1, 4, 9])
    Rn.plot_graph_topology(options=options_top, node_color='eigenvectors', eigen_vecs=[0, 1, 4, 9])

    Pn.eigs_compare_analytic()
    Rn.eigs_compare_analytic()

if section == 2:
    """ Graph Product"""
    n_x = 71
    n_y = 31

    G = Graph(type='RnXRn', nx=n_x, ny=n_y)
    G.nodes(scatter=True)
    G.plot_graph_topology()
    G.plot_eigenvalues()
    G.plot_graph_topology(node_color='eigenvectors', eigen_vecs=[0, 1, 4, 6, 9])
    G.eigs_compare_analytic()

if section == 3:
    """ Noisy Graph Product"""
    n_x = 71
    n_y = 31
    G = Graph(type='RnXRn', nx=n_x, ny=n_y, pertube=True)
    G.plot_graph_topology()
    G.plot_eigenvalues()
    G.plot_graph_topology(node_color='eigenvectors', eigen_vecs=[0, 1, 4, 6, 9])
