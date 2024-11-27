import matplotlib.pyplot as plt
import networkx as nx
import torch

from Misc.config import config
from Misc.utils import edge_tensor_to_list
from GraphGen.mutate_graph import MutateGraph
from Exp.preparation import load_dataset
from Exp.parser import parse_args

p_feat = 0.2
p_edge_drop = 0.05
p_edge_add = 0.05

default_color = 'white'
default_color_edge = 'black'

below_line_delta = 0
above_line_delta = 0
baseline_color = 'r'
student_colors = ['m', 'y']
teacher_color = 'g'

def visualize(data, name, colors, v_feat_dim = None, e_feat_dim = None, pos = None):
    if len(data.edge_attr.shape) == 1:
        data.edge_attr = torch.unsqueeze(data.edge_attr, dim = 1)
    
    G = nx.DiGraph(directed=True)

    edge_list = edge_tensor_to_list(data.edge_index)

    for i, edge in enumerate(edge_list):
        G.add_edge(edge[0], edge[1], color=default_color_edge if (e_feat_dim is None) else colors[data.edge_attr[i, e_feat_dim]])

    color_map_vertices = [default_color for _ in G]
    color_map_edges = [default_color_edge for _ in range(data.edge_index.shape[1])]

    if v_feat_dim is not None:
        node_types = data.x[:, v_feat_dim].tolist()
        for i, node in enumerate(G):
            color_map_vertices[i] = colors[node_types[node]]

    edges = G.edges()
    color_map_edges = [G[u][v]['color'] for u,v in edges]

    if pos is None:
        pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos = pos, node_color=color_map_vertices,  edge_color=color_map_edges,arrows=False)
    plt.savefig(f'Imgs/{name}.pdf')
    plt.savefig(f'Imgs/{name}.jpg')

    plt.close()
    return pos
    
def plot(student_perf, teacher_perf, xs, ys, student_names, errors, name_plot, name_teacher, y_label):
    plt.style.use('ggplot')    
    fig, ax = plt.subplots()
    plt.xticks(xs[0])
    
    for x, y, errors, student_name, student_color in zip(xs, ys, errors, student_names, student_colors):
        plt.plot(x, y, student_color + 's', label = student_name)
        ax.errorbar(x, y, yerr=errors, color=student_color, capsize=5)

    if teacher_perf > student_perf:
        teacher_line_delta = below_line_delta
        student_line_delta = above_line_delta
    else:
        teacher_line_delta = above_line_delta
        student_line_delta = below_line_delta
        
    #  GIN baseline
    plt.axhline(student_perf, color=baseline_color)
    center_x = x[0] + ((x[1]-x[0])/2)
    plt.text(center_x, student_perf + student_line_delta, 'GIN')

    # Teacher comparison
    plt.axhline(teacher_perf, color=teacher_color)
    plt.text(center_x, teacher_perf + teacher_line_delta, name_teacher)

    plt.xlabel("Relative Distillation Dataset Size")
    
    plt.ylabel(y_label)
    plt.legend(loc=(0.75,0.8))
    plt.tight_layout()

    plt.savefig(f'Imgs/{name_plot}.pdf')
    plt.savefig(f'Imgs/{name_plot}.jpg')

    plt.close()

def plot_experiment_results():
    # DSS
    gin_perf = 0.194
    teacher_perf = 0.0928
    x = [0, 50, 100, 150, 200]
    y = [gin_perf, 0.1409, 0.1271, 0.1134, 0.1146]
    errors = [0, 0.0084, 0.0053, 0.0014, 0.0023]
    plot(gin_perf, teacher_perf, [x], [y], [r"GIN$_\mathrm{S}$"], [errors], "ZINC_DSS", r"DSS$_\mathrm{T}$", "Mean Average Error")

    # CWN
    gin_perf = 0.194
    teacher_perf = 0.1266
    x = [0, 50, 100, 150]
    y = [gin_perf, 0.1652, 0.1477, 0.1365]
    errors = [0, 0.0100, 0.0095, 0.0060]
    plot(gin_perf, teacher_perf, [x], [y], [r"GIN$_\mathrm{S}$"], [errors], "ZINC_CWN", r"CWN$_\mathrm{T}$", "Mean Average Error")

    # MOLHIV
    gin_perf = 75.82
    teacher_perf = 79.06
    x1 = [0, 3, 10, 20]
    x2 = [0, 3, 10, 20]
    y1 = [gin_perf, 75.17, 76.66, 75.04]
    y2 = [gin_perf, 75.25, 74.42, 75.6]
    errors1 = [0, 1.68, 2.46, 2.03]
    errors2 = [0, 1.82, 1.56, 1.57]

    plot(gin_perf, teacher_perf, [x1,x2 ], [y1, y2], [r"GIN$_{5,\mathrm{S}}$", r"GIN$_{23,\mathrm{S}}$"], [errors1, errors2], "MOLHIV_CWN", r"CWN$_\mathrm{T}$", "ROC-AUC")
    

def plot_graphs():
    parameters = parse_args({
        "--dataset": "ZINC",
        "--batch_size": 1
        })
    
    dataset = load_dataset(parameters, config)[0].dataset
    
    node_features = [list(range(21))]
    edge_features = [list(range(4))]
            
    colors = [i/21 for i in range(21)]

    graph = dataset[0]
    print(f"graph: {graph}")
    
    pos = visualize(graph, "graph_1", colors, v_feat_dim = 0, e_feat_dim = 0)
    
    mutate = MutateGraph(node_features, possible_edge_features = edge_features, train_set = dataset, prob_node_feat_mut = p_feat, prob_edge_drop = p_edge_drop, prob_edge_add = p_edge_add)
    transformed_graph = mutate(graph)
    print(f"transformed_graph: {transformed_graph}")
    visualize(transformed_graph, "graph_1_transformed_layout1", colors, v_feat_dim = 0, e_feat_dim = 0, pos = pos)
    visualize(transformed_graph, "graph_1_transformed_layout2", colors, v_feat_dim = 0, e_feat_dim = 0)
    
    graph = dataset[0]        
    mutate = MutateGraph(node_features, possible_edge_features = edge_features, train_set = dataset, prob_node_feat_mut = p_feat, prob_edge_drop = p_edge_drop, prob_edge_add = p_edge_add)
    transformed_graph = mutate(graph)
    print(f"transformed_graph: {transformed_graph}")
    visualize(transformed_graph, "graph_1_transformed2_layout1", colors, v_feat_dim = 0, e_feat_dim = 0, pos = pos)
    visualize(transformed_graph, "graph_1_transformed2_layout2", colors, v_feat_dim = 0, e_feat_dim = 0)
    
def main():
    plot_graphs()
    plot_experiment_results()

if __name__ == "__main__":
    main()