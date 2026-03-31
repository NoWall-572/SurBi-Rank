import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import networkx as nx
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.linalg as la
import scipy.stats as stats
from tqdm import tqdm
from core.config import Config
from core.network_env import SwarmSystem
from models.gcn_module import AdaptiveSurBiGCN, extract_features, normalize_adj


def calculate_natural_connectivity(adj):
    evals = np.real(la.eigvals(adj))
    return np.log(np.sum(np.exp(evals)) / len(adj) + 1e-9)


def get_ground_truth_ranking(G):
    n = len(G)
    adj_orig = nx.to_numpy_array(G)
    phi_0 = calculate_natural_connectivity(adj_orig)
    drops = np.zeros(n)

    degrees_init = dict(G.degree())
    K_nodes = sorted(degrees_init, key=degrees_init.get, reverse=True)[:max(1, int(n * 0.05))]

    for i in range(n):
        active_mask = np.ones(n, dtype=bool)
        active_mask[i] = False
        while True:
            nodes_idx = np.where(active_mask)[0]
            if len(nodes_idx) < 2: break

            sub_adj = adj_orig[np.ix_(nodes_idx, nodes_idx)]
            curr_degrees = np.sum(sub_adj > 1e-6, axis=1)

            to_fail_local = np.where(curr_degrees <= 2)[0]
            if len(to_fail_local) == 0:
                break

            to_fail_global = nodes_idx[to_fail_local]
            active_mask[to_fail_global] = False

        final_nodes = np.where(active_mask)[0]
        if len(final_nodes) == 0:
            final_nc = 0
        else:
            G_res = nx.from_numpy_array(adj_orig[np.ix_(final_nodes, final_nodes)])
            global_to_local = {v: k for k, v in enumerate(final_nodes)}
            alive_K = [global_to_local[k] for k in K_nodes if k in global_to_local]

            if not alive_K:
                final_nc = 0
            else:
                functional_nodes_local = set()
                for k_local in alive_K:
                    functional_nodes_local.update(nx.node_connected_component(G_res, k_local))
                if len(functional_nodes_local) < len(G_res) * 0.1: 
                    final_nc = 0
                else:
                    func_idx = list(functional_nodes_local)
                    adj_functional = nx.to_numpy_array(G_res.subgraph(func_idx))
                    final_nc = calculate_natural_connectivity(adj_functional)

        drops[i] = max(0, phi_0 - final_nc)

    return drops


def find_optimal_r(G):
    n = len(G)
    adj = nx.to_numpy_array(G)
    true_drops = get_ground_truth_ranking(G)

    phi_0 = calculate_natural_connectivity(adj)
    bi_scores = np.zeros(n)
    for i in range(n):
        adj_sub = np.delete(np.delete(adj, i, 0), i, 1)
        bi_scores[i] = max(0, phi_0 - calculate_natural_connectivity(adj_sub))
    if np.max(bi_scores) > 1e-9: bi_scores /= np.max(bi_scores)

    try:
        ks = list(nx.core_number(G).values())
        ec = list(nx.eigenvector_centrality(G, max_iter=1000).values())
    except:
        ks = [d for n, d in G.degree()]
        ec = [d for n, d in G.degree()]

    si_scores = np.zeros(n)
    nodes = list(G.nodes())
    for i in range(n):
        nbs = list(G.neighbors(nodes[i]))
        sum_ks = sum([ks[nodes.index(nb)] for nb in nbs])
        si_scores[i] = sum_ks * ec[i]
    if np.max(si_scores) > 1e-9: si_scores /= np.max(si_scores)

    best_r = 0.5
    max_tau = -1.0

    for r in np.arange(0.0, 1.01, 0.1): 
        surbi_scores = r * bi_scores + (1 - r) * si_scores
        tau, _ = stats.kendalltau(true_drops, surbi_scores)

        num_ties = np.sum(np.isclose(np.diff(surbi_scores), 0)) 
        tau -= 0.01 * num_ties

        if np.isnan(tau): tau = 0.0
        if tau > max_tau:
            max_tau = tau
            best_r = r

    return best_r, max_tau


def generate_random_graph():
    n = np.random.randint(50, 150)  
    graph_type = np.random.choice(['BA', 'RG', 'WS', 'ER'])
    if graph_type == 'BA':
        m = np.random.randint(2, 5)
        G = nx.barabasi_albert_graph(n, m)
    elif graph_type == 'RG':
        G = nx.random_geometric_graph(n, radius=np.random.uniform(0.15, 0.35))
    elif graph_type == 'WS':
        k = np.random.randint(2, 6)
        p = np.random.uniform(0.1, 0.3)
        G = nx.watts_strogatz_graph(n, k, p)
    elif graph_type == 'ER':
        p = np.random.uniform(0.05, 0.1)
        G = nx.erdos_renyi_graph(n, p)

    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        G = nx.convert_node_labels_to_integers(G)
    return G


def main():
    print("=== Phase 1: Dataset Generation & Optimal r Search (Kendall's Tau) ===")
    dataset = []
    tau_list = []
    for _ in tqdm(range(500)): 
        G = generate_random_graph()
        if len(G) < 10: continue
        best_r, best_tau = find_optimal_r(G)
        dataset.append((G, best_r))
        tau_list.append(best_tau)

    print(f"Average Kendall's Tau achieved: {np.mean(tau_list):.4f}")

    print("\n=== Phase 2: Training GCN Model ===")
    processed_dataset = []
    for G, label_r in dataset:
        features = extract_features(G)
        adj_norm = normalize_adj(nx.to_numpy_array(G))
        target = torch.tensor([label_r], dtype=torch.float32)
        processed_dataset.append({'x': features, 'adj': adj_norm, 'y': target})

    model = AdaptiveSurBiGCN(num_features=5, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    loss_func = nn.MSELoss()

    model.train()
    for epoch in range(100):
        np.random.shuffle(processed_dataset)
        total_loss = 0
        for data in processed_dataset:
            optimizer.zero_grad()
            pred_r = model(data['x'], data['adj'])
            if pred_r.dim() == 0: pred_r = pred_r.unsqueeze(0)
            loss = loss_func(pred_r, data['y'])

            print(f"Epoch {epoch + 1}, Predicted r: {pred_r.item():.4f}, True r: {data['y'].item():.4f}, Loss: {loss.item():.5f}")

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/100, Average Loss: {total_loss / len(processed_dataset):.5f}")

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, 'gcn_r_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to '{model_path}'")

if __name__ == "__main__":
    main()
