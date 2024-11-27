import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch.nn import Linear, ReLU, ModuleList, Parameter
from torch_geometric.nn.inits import uniform

from Models.conv import GNN_node, GNN_node_Virtualnode

from torch_scatter import scatter_mean

class GNN(torch.nn.Module):

    def __init__(self, num_classes, num_tasks, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean",
                    node_encoder = lambda x: x, edge_encoder = lambda x: x, use_node_encoder = True, graph_features = 0, num_mlp_layers = 1, weights = None, dim_pooling = False, freeze_mlp = False):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()
        
        print(f"JK: {JK}")

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.use_node_encoder = use_node_encoder
        self.graph_features = graph_features
        self.num_mlp_layers = num_mlp_layers
        self.dim_pooling = dim_pooling

        if self.num_layer < 1:
            raise ValueError("Number of GNN layers must be at least 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, node_encoder=node_encoder, edge_encoder=edge_encoder)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, node_encoder=node_encoder, edge_encoder=edge_encoder)

        self.graph_emb_dim = emb_dim if (JK != "concat") else (emb_dim*(num_layer+1))
        if self.graph_pooling == "set2set":
            self.graph_emb_dim = self.graph_emb_dim*2
        
        self.set_mlp(graph_features, weights = weights, train_mlp = not freeze_mlp)
        
        ### Pooling function to generate whole-graph embeddings
        print(f"graph_pooling: {graph_pooling}")
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            in_dim = emb_dim if (JK != "concat") else (emb_dim*(num_layer+1))
            self.set2set = Set2Set(in_dim, processing_steps=6)
            self.pool = lambda x, y: self.set2set(x, y, dim = 0)  
        
        self.mean_pool = global_mean_pool
           
        # Dimensional pooling (for CWN)
        if self.dim_pooling:
            print("CWN: Using dimensional pooling.")
            self.lin_per_dim = torch.nn.ModuleList()
            self.relu = torch.nn.ReLU()
            for _ in range(3):
                self.lin_per_dim.append(torch.nn.Linear(self.graph_emb_dim, self.graph_emb_dim))

    def set_mlp(self, graph_features = 0, weights = None, train_mlp = True):
        print(f"set_mlp({graph_features}, weights are none: {weights is None})")
        print(f"nr layers to generate: {self.num_mlp_layers}")
        hidden_size = self.emb_dim * 2
        final_layers = ModuleList([])
        final_layers.requires_grad = False

        for i in range(self.num_mlp_layers):
            in_size = hidden_size if i > 0 else self.graph_emb_dim
            out_size = hidden_size if i < self.num_mlp_layers - 1 else self.num_classes*self.num_tasks

            new_linear_layer = Linear(in_size, out_size)
            print(f"{in_size}->{out_size}")

            if weights is not None:
                print(f"Loading weights into {i}th layer, training: {train_mlp}")
                new_linear_layer.weight.requires_grad = False

                print(weights[f"final_layers.{i*2}.weight"].shape)
                print(weights[f"final_layers.{i*2}.bias"].shape)
                new_linear_layer.bias.requires_grad = False

                if i == 0:
                    new_linear_layer.weight[:, -self.emb_dim:] = weights[f"final_layers.{i*2}.weight"].detach().clone()[:, -self.emb_dim:]
                    new_linear_layer.bias = torch.nn.Parameter(weights[f"final_layers.{i*2}.bias"].detach().clone())
                else:
                    new_linear_layer.weight = torch.nn.Parameter(weights[f"final_layers.{i*2}.weight"].detach().clone())
                    new_linear_layer.bias = torch.nn.Parameter(weights[f"final_layers.{i*2}.bias"].detach().clone())

                if train_mlp:
                    new_linear_layer.weight.requires_grad = True
                    new_linear_layer.bias.requires_grad = True

            final_layers.append(new_linear_layer)

            if self.num_mlp_layers > 0 and i < self.num_mlp_layers - 1:
                final_layers.append(ReLU())

        if train_mlp:
            final_layers.requires_grad = True
        self.final_layers = final_layers

    def forward(self, batched_data):
        h_node, node_emb = self.gnn_node(batched_data)
        
        node_emb = list(map(lambda node_emb: self.mean_pool(node_emb, batched_data.batch), node_emb))

        if self.dim_pooling:
            dimensional_pooling = []
            for dim in range(3):
                multiplier = torch.unsqueeze(batched_data.node_type[:, dim], dim=1)
                single_dim = h_node * multiplier
                single_dim = self.pool(single_dim, batched_data.batch)
                single_dim = self.relu(self.lin_per_dim[dim](single_dim))
                dimensional_pooling.append(single_dim)
            h_graph = sum(dimensional_pooling)
        else:
            h_graph = self.pool(h_node, batched_data.batch)

        prediction = h_graph
            
        for i, layer in enumerate(self.final_layers):
            prediction = layer(prediction)         

        if self.num_tasks == 1:
            prediction = prediction.view(-1, self.num_classes)
        else:
            prediction.view(-1, self.num_tasks, self.num_classes)
            
        return prediction, h_graph, node_emb

if __name__ == '__main__':
    GNN(num_tasks = 10)