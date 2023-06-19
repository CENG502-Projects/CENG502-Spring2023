import torch
import pickle   
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import degree

def save_all(model, train_losses, val_losses, epoch, save_path):
  # save the model
  model_save_path = save_path + "/model_"+str(epoch)+".pt"
  torch.save(model.state_dict(), model_save_path)

  # save the train loss
  trainloss_save_path = save_path + "/train_losses_" + str(epoch) + ".pkl"
  with open(trainloss_save_path, 'wb') as f:
      pickle.dump(train_losses, f)

  # save the val loss
  valloss_save_path = save_path + "/val_losses_" + str(epoch) + ".pkl"
  with open(valloss_save_path, 'wb') as f:
      pickle.dump(val_losses, f)

import torch

def calculate_recall_per_node(all_edges, all_scores, positive_edge_indicator, K, start_node):
    """
    Calculate recall for each individual starting node using tensor operations on GPU.

    Parameters:
    - all_edges: Tensor of shape [2, num_edges], containing edges (source -> target).
    - all_scores: Tensor of shape [num_edges], containing scores for each edge.
    - positive_edge_indicator: Tensor of shape [num_edges], containing 1 for positive edges and 0 for negative edges.
    - K: The number of top edges to consider for calculating recall.

    Returns:
    - recall_per_node: Dictionary with nodes as keys and recall as values.
    """

    #print("all_edges")
    #print(all_edges.shape)

    #print("all_scores")
    #print(all_scores.shape)

    #print("positive_edge_indicator")
    #print(positive_edge_indicator.shape)





    # Sort scores in descending order
    sorted_indices = torch.argsort(all_scores, descending=True)

    positive_edge_indicator = positive_edge_indicator[sorted_indices]

    recall =  positive_edge_indicator[:K].sum() / positive_edge_indicator.sum()




    # Create bins for each unique start node
    #bins = torch.zeros_like(all_scores).scatter_(0, all_edges[0, :], 1).cumsum(0)

    # Create a mask for top K elements in each bin
    #top_k_mask = (bins <= K).gather(0, torch.argsort(bins.gather(0, sorted_indices)))

    # Compute recalls by start node
    #top_k_sorted_positive_indicators = positive_edge_indicator[sorted_indices][top_k_mask]
    #recall = (top_k_sorted_positive_indicators.view(len(start_nodes), -1).sum(1) / K).cpu().numpy()

    # Create recall_per_node dictionary
    #recall_per_node = {node.item(): recall for node, recall in zip(start_nodes, recalls)}

    return recall

# Example usage remains the same
import torch

def remove_common_edges(E_all, B):
    return B
    # Compute the pairwise equality
    pairwise_equality = torch.eq(E_all.unsqueeze(2), B.unsqueeze(1))

    # Determine the columns where all rows are True (i.e., both elements in column are equal)
    column_equality = torch.all(pairwise_equality, dim=0)

    # Clear intermediate tensor
    del pairwise_equality

    # Use in-place operation to set intersection elements to 0
    intersection = B[:, column_equality.any(dim=0)]
    intersection[:] = 0

    # Create a new tensor without intersection elements
    B_without_intersection = B[:, ~column_equality.any(dim=0)].clone()

    return B_without_intersection

  # Display the intersection and B without intersection
  #print("Intersection:")
  #print(intersection.cpu())
  #print("B without intersection:")
  #print(B_without_intersection.cpu())




def analyzer(recall_at_k, node_degrees):
    # Assuming recall_at_k is a 2D array where each row contains [node_index, recall_value]
    # Assuming node_degrees is an array where each element is the degree of the corresponding node
    
    # Find unique degrees
    unique_degrees = np.unique(node_degrees)
    
    # Calculate average recalls for each unique degree using vectorized operations
    sums = np.bincount(node_degrees, weights=recall_at_k[:, 1])
    counts = np.bincount(node_degrees)
    avg_recalls = sums / counts
    
    # Filter out NaN values (when counts is 0)
    avg_recalls = avg_recalls[np.isfinite(avg_recalls)]
    
    # Stack unique_degrees and avg_recalls into a 2D array
    degree_recall_array = np.column_stack((unique_degrees, avg_recalls))
    
    return degree_recall_array

def renormalize(edge_index, num_nodes):
    # Convert to PyTorch tensor for calculation
    edge_index = edge_index.clone().detach()

    # Calculate degree and create Degree Matrix D
    row, col = edge_index
    deg = degree(row, num_nodes, dtype=edge_index.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

    # Renormalize
    row, col = edge_index
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    return edge_index, edge_weight


def random_edge_sampler(data, percent):
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges)
    preserve_nnz = int(num_edges * percent)

    # Indices for kept edges
    kept_indices = perm[:preserve_nnz]
    kept_edges = edge_index[:, kept_indices]
    kept_edges, kept_weights = renormalize(kept_edges, num_nodes)
    data_kept = Data(edge_index=kept_edges, edge_attr=kept_weights)

    # Indices for dropped edges
    dropped_indices = perm[preserve_nnz:]
    dropped_edges = edge_index[:, dropped_indices]
    dropped_edges, dropped_weights = renormalize(dropped_edges, num_nodes)
    data_dropped = Data(edge_index=dropped_edges, edge_attr=dropped_weights)

    return data_kept, data_dropped