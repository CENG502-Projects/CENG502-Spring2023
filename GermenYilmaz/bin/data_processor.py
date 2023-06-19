import train_controller
from ogb.nodeproppred import Evaluator, NodePropPredDataset
from spektral.datasets.ogb import OGB
from spektral.transforms import AdjToSpTensor, GCNFilter
import os
import torch
from torch_geometric.data import Data
import random

class data_processor():

    def __init__(self, train_controller) -> None:
        self.train_controller = train_controller

        self.SEED = self.train_controller.SEED
        self.device = self.train_controller.device

        self.V_percentage = self.train_controller.V_percentage
        self.train_edge_percentage = self.train_controller.train_edge_percentage

        self.download_data()
        self.convert2torch()

        self.node_split()
        self.edge_split()

        self.edge_validation()
        self.node_degree()
        pass

    def download_data(self):
        # Set download location to cwd
        
        dataset_name = "ogbn-arxiv"
        current_directory = os.getcwd()
        parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
        download_root = os.path.join(parent_directory, "ogb_datasets")  # Specify the desired parent directory

        ogb_dataset = NodePropPredDataset(dataset_name, root=download_root)

        self.dataset = OGB(ogb_dataset, transforms=[GCNFilter(), AdjToSpTensor()])

    def convert2torch(self):
        # convert tf dataset to torch tensor

        # Get the node features, edge indices, and labels
        features = self.dataset[0].x
        edge_indices = self.dataset[0].a.indices
        labels = self.dataset[0].y

        # Convert TensorFlow tensors to PyTorch Tensors
        features_torch = torch.from_numpy(features)
        edge_indices_torch = torch.from_numpy(edge_indices.numpy().T).long()  # Transpose to fit PyG's edge_index format and convert to long
        labels_torch = torch.from_numpy(labels)

        # Create a PyTorch Geometric Data object
        self.data = Data(x=features_torch, edge_index=edge_indices_torch, y=labels_torch)
        pass

    def node_split(self):
        # Get the number of nodes in your graph.
        self.num_nodes = self.data.num_nodes

        # Create a random permutation of indices [0, 1, 2, ..., num_nodes-1].
        perm = list(range(self.num_nodes))

        random.Random(self.SEED).shuffle(perm)

        # Calculate the index at which to split the permutation.
        split_idx = int(self.num_nodes * self.V_percentage)

        # Split the permutation into indices for V (95%) and V_new (5%).
        self.V = torch.tensor(perm[:split_idx]).to(self.device)
        self.V_new =torch.tensor(perm[split_idx:]).to(self.device)

    def edge_split(self):
        # Get the number of edges
        num_edges = self.data.edge_index.size(1)

        # Create a list of indices representing the edges
        edge_indices = list(range(num_edges))

        # Shuffle the indices randomly
        random.Random(self.SEED).shuffle(edge_indices)

        # Define the percentage of edges to be used for training
        num_train_edges = int(self.train_edge_percentage * num_edges)

        # Split the indices into two sets: for training and validation
        train_edge_indices = edge_indices[:num_train_edges]
        val_edge_indices = edge_indices[num_train_edges:]

        # Function to create a new edge_index tensor based on selected indices
        def create_edge_index_subset(edge_index, selected_indices):
            return edge_index[:, selected_indices]

        # Create new edge_index tensors for training and validation
        self.E_train = create_edge_index_subset(self.data.edge_index, train_edge_indices).to(self.device)
        self.E_val = create_edge_index_subset(self.data.edge_index, val_edge_indices).to(self.device)
        self.E_all = self.data.edge_index.to(self.device).to(self.device)

    def edge_validation(self):
        v_mask = torch.zeros(self.num_nodes, dtype=torch.bool).to(self.device)
        v_mask[self.V] = True
        v_mask = v_mask.to(self.device)

        source_nodes = self.E_val[0, :]
        target_nodes = self.E_val[1, :]
        can_exist_in_V = v_mask[source_nodes] & v_mask[target_nodes]
        valid_edges_in_V = self.E_val[:, can_exist_in_V]

        start_nodes = torch.unique(valid_edges_in_V)

        self.start_node_dict_val = {}
        for start_node in start_nodes:
            start_node = int(start_node)
            mask = valid_edges_in_V[0] == start_node
            edges = valid_edges_in_V[:, mask]
            self.start_node_dict_val[start_node] = edges

    def node_degree(self):
        self.node_degree = torch.zeros(self.num_nodes).to(self.device)
        self.node_degree[self.data.edge_index[0, :]] += 1
        self.node_degree[self.data.edge_index[1, :]] += 1
        pass