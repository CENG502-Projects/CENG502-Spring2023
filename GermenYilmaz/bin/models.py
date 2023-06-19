import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv



# Define a simple GNN model
class GCN(torch.nn.Module):
    def __init__(self, num_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 256)
        self.conv2 = GCNConv(256, 256)
        self.conv3 = GCNConv(256, 256)

        self.scoring = torch.nn.Sequential(
            torch.nn.Linear(2 * 256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )

    def forward(self, data, edge_index):
        x = self.conv1(data.x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv3(x, edge_index)
        return x

    def decode(self, z, indices):
        start, end = indices
        edge_features = torch.cat([z[start], z[end]], dim=1)
        return self.scoring(edge_features).squeeze(-1)


def bpr_loss(pos_logit, neg_logit):
    #return F.log(F.sigmoid(pos_logit - neg_logit)).sum()
    return -F.logsigmoid(pos_logit - neg_logit).sum()

