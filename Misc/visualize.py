import networkx as nx
import matplotlib.pyplot as plt
import torch

from Misc.utils import edge_tensor_to_list

colors_feat = ['red', 'orange', 'yellow', 'blue', 'green', 'grey', 'pink', 'magenta', 'maroon', 'purple']*3
colors_type = ['red', 'orange', 'yellow', 'blue', 'green', 'grey', 'pink', 'magenta']
cololrs_edges = ['orange', 'red', 'yellow', 'blue', 'green', 'grey', 'pink', 'purple', 'magenta', 'maroon']
default_color = 'white'
default_color_edge = 'black'

max_value_color = 'magenta'

def visualize(data, name,  v_feat_dim = None, e_feat_dim = None):
    G = nx.DiGraph(directed=True)

    edge_list = edge_tensor_to_list(data.edge_index)

    if len(data.edge_attr.shape) == 1:
        data.edge_attr = torch.unsqueeze(data.edge_attr, 1)

    for i, edge in enumerate(edge_list):
        G.add_edge(edge[0], edge[1], color=default_color_edge if (e_feat_dim is None) else cololrs_edges[data.edge_attr[i, e_feat_dim]])

    color_map_vertices = [default_color for _ in G]
    color_map_edges = [default_color_edge for _ in range(data.edge_index.shape[1])]

    if v_feat_dim is not None:
        node_types = data.x[:, v_feat_dim].tolist()
        for i, node in enumerate(G):
            color_map_vertices[i] = colors_feat[node_types[node]]

    edges = G.edges()
    color_map_edges = [G[u][v]['color'] for u,v in edges]

    nx.draw_kamada_kawai(G, node_color=color_map_vertices,  edge_color=color_map_edges, with_labels=True, font_weight='bold', arrows=True)
    plt.savefig(f'Imgs/{name}.png')
    plt.close()
