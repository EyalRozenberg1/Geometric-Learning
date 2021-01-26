from utils.graph import DmbBolGraph as Graph

n = 30

D = Graph('dumbbell', n)
options_top = {
    'node_size': 30,
    'width': 0.2,
    'with_labels': True,
    'font_weight': 'normal',
    'font_size': 20}
D.plot_graph_topology(options=options_top)
D.plot_eigenvalues()
options_top['with_labels'] = False
options_top['node_size'] = 50
p,i0 = D.Run_lazy_walk(eps=3e-6)
l = p.shape[1]
a = l**(1/6)
D.plot_graph_topology(options=options_top, p=p, time_step=[0, round(a**2), round(a**3), round(a**4), round(a**5), l-1])
D.upper_bound(p, i0)

B = Graph('bolas', n)
options_top = {
    'node_size': 20,
    'width': 0.1,
    'with_labels': True,
    'font_weight': 'normal',
    'font_size': 10}
B.plot_graph_topology(options=options_top)
B.plot_eigenvalues()

options_top['with_labels'] = False
options_top['node_size'] = 40
p, i0 = B.Run_lazy_walk(eps=1e-6)
l = p.shape[1]
a = l**(1/6)
B.plot_graph_topology(options=options_top, p=p, time_step=[0, round(a**2), round(a**3), round(a**4), round(a**5), l-1])
B.upper_bound(p, i0)
