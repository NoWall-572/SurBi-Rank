import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx


class AdaptiveSurBiGCN(nn.Module):
    def __init__(self, num_features=5, hidden_dim=64):
        super(AdaptiveSurBiGCN, self).__init__()
        self.gc1 = nn.Linear(num_features, hidden_dim)
        self.gc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gc3 = nn.Linear(hidden_dim, hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x, adj):
        # Layer 1
        x = F.relu(self.gc1(x))
        x = torch.spmm(adj, x)

        # Layer 2
        x = F.relu(self.gc2(x))
        x = torch.spmm(adj, x)

        # Layer 3
        x = F.relu(self.gc3(x))
        x = torch.spmm(adj, x)

        graph_embedding = torch.mean(x, dim=0, keepdim=True)

        out = F.relu(self.fc1(graph_embedding))
        r = torch.sigmoid(self.fc2(out)) 

        return r.squeeze()

def extract_features(G):
    n = len(G)
    dc = np.array(list(nx.degree_centrality(G).values()))
    cc = np.array(list(nx.closeness_centrality(G).values()))
    try:
        ec = np.array(list(nx.eigenvector_centrality(G, max_iter=600).values()))
    except:
        ec = np.zeros(n)

    clust = np.array(list(nx.clustering(G).values()))

    # PageRank
    pr = np.array(list(nx.pagerank(G, alpha=0.85).values()))

    features = np.vstack([dc, cc, ec, clust, pr]).T

    f_min, f_max = features.min(axis=0), features.max(axis=0)
    denom = f_max - f_min
    denom[denom == 0] = 1.0
    features = (features - f_min) / denom

    return torch.FloatTensor(features)


def normalize_adj(adj_matrix):
    if not isinstance(adj_matrix, np.ndarray):
        adj_matrix = np.array(adj_matrix)

    adj = adj_matrix + np.eye(adj_matrix.shape[0])
    row_sum = np.array(adj.sum(1))

    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.

    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)

    return torch.FloatTensor(norm_adj)
