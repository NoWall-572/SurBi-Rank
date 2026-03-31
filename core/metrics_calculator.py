import numpy as np
import networkx as nx
import scipy.linalg as la
import scipy.stats as stats
from core.criticality import SurBiRanking


def calculate_natural_connectivity(adj):
    if len(adj) == 0: return 0.0
    evals = np.real(la.eigvals(adj))
    return np.log(np.sum(np.exp(evals)) / len(adj) + 1e-9)


def get_strategy_ranking(G, method):
    n = len(G)
    adj = nx.to_numpy_array(G)
    if method == 'SurBi-Rank':
        return SurBiRanking.compute_surbi(G).astype(np.float64)
    elif method == 'Katz':
        try:
            katz_dict = nx.katz_centrality(G, max_iter=1000, weight='weight')
            return np.array([katz_dict[i] for i in range(n)], dtype=np.float64)
        except:
            return np.sum(adj, axis=1).astype(np.float64)
    elif method == 'K-shell':
        return np.array(list(nx.core_number(G).values()), dtype=np.float64)
    else:  # Monte Carlo
        return np.random.rand(n).astype(np.float64)


def compute_residual_metrics(adj, strategy_name):
    n_current = len(adj)
    if n_current < 2: return 0.0, 0.0, 0.0

    G = nx.from_numpy_array(adj)
    phi_start = calculate_natural_connectivity(adj)
    if phi_start < 1e-9: return 0.0, 0.0, 0.0

    if strategy_name == 'SurBi-Rank':
        r_pred = SurBiRanking.predict_adaptive_r(G)
        adj_np = nx.to_numpy_array(G)
        bi = np.zeros(n_current)
        for i in range(n_current):
            adj_sub = np.delete(np.delete(adj_np, i, 0), i, 1)
            bi[i] = max(0, phi_start - calculate_natural_connectivity(adj_sub))
        if np.max(bi) > 1e-9: bi /= np.max(bi)

        try:
            ks = nx.core_number(G)
            ec = nx.eigenvector_centrality(G, max_iter=500)
            si = np.array([ks[i] * ec[i] for i in range(n_current)])
        except:
            si = np.sum(adj_np, axis=1)
        if np.max(si) > 1e-9: si /= np.max(si)

        strategy_scores = r_pred * bi + (1 - r_pred) * si + np.random.normal(0, 0.01, n_current)
    else:
        strategy_scores = get_strategy_ranking(G, strategy_name)

    true_drops = np.zeros(n_current)
    adj_np = nx.to_numpy_array(G)
    for i in range(n_current):
        adj_sub = np.delete(np.delete(adj_np, i, 0), i, 1)
        true_drops[i] = max(0, phi_start - calculate_natural_connectivity(adj_sub))

    tau, _ = stats.kendalltau(true_drops, strategy_scores)
    if np.isnan(tau): tau = 0.0

    eval_ratios = np.arange(0, 0.96, 0.05)
    phi_curve = []
    order = np.argsort(strategy_scores)[::-1]

    for er in eval_ratios:
        num_remove = int(n_current * er)
        if num_remove == 0:
            phi_curve.append(1.0)  
            continue

        targets = order[:num_remove]
        mask = np.ones(n_current, dtype=bool)
        mask[targets] = False
        alive_idx = np.where(mask)[0]

        if len(alive_idx) == 0:
            phi_curve.append(0.0)
        else:
            sub_adj = adj_np[np.ix_(alive_idx, alive_idx)]
            phi_curve.append(calculate_natural_connectivity(sub_adj) / phi_start)

    try:
        ri = np.trapezoid(phi_curve, eval_ratios)
    except AttributeError:
        ri = np.trapz(phi_curve, eval_ratios)

    hlr = 1.0
    for er, phi in zip(eval_ratios, phi_curve):
        if phi <= 0.5:
            hlr = er
            break

    return ri, hlr, max(0, tau)