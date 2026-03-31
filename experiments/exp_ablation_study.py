import sys
import os
import numpy as np
import networkx as nx
import pandas as pd
import scipy.linalg as la
import scipy.stats as stats
import multiprocessing
from multiprocessing.pool import ThreadPool

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import Config
from core.network_env import SwarmSystem
from core.optimization import StaticTopologyProblem, GraphSmartSampling
from core.criticality import SurBiRanking
from utils.decision_maker import critic_weights, select_knee_point

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def calculate_natural_connectivity(adj):
    if len(adj) == 0: return 0.0
    evals = np.real(la.eigvals(adj))
    return np.log(np.sum(np.exp(evals)) / len(adj) + 1e-9)


def main(pool):
    print("=== Ablation Study: Rigorous Component Analysis ===")
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("\n[1/4] Generating Physical Swarm Environment & Optimizing G0...")
    swarm = SwarmSystem()
    swarm.generate_topology()

    problem = StaticTopologyProblem(swarm)
    algorithm = NSGA2(
        pop_size=40,
        sampling=GraphSmartSampling(swarm),
        crossover=TwoPointCrossover(prob=0.8),
        mutation=BitflipMutation(prob=0.05),
        eliminate_duplicates=True
    )
    res = minimize(problem, algorithm, ('n_gen', 20), runner=pool, verbose=False)

    if res.F is None or len(res.F) == 0:
        print("Error: Optimization failed.")
        return

    weights = critic_weights(res.F)
    best_idx = select_knee_point(res.F, weights)
    G0_adj = problem._decode(res.X[best_idx])

    print("\n[2/4] Calculating Ground Truth (BI) and Local Topology (SI)...")
    n_nodes = len(G0_adj)
    G0 = nx.from_numpy_array(G0_adj)
    phi_0 = calculate_natural_connectivity(G0_adj)

    true_drops = np.zeros(n_nodes)
    adj_orig = nx.to_numpy_array(G0)
    
    degrees_init = dict(G0.degree())
    K_nodes = sorted(degrees_init, key=degrees_init.get, reverse=True)[:max(1, int(n_nodes * 0.05))]

    for i in range(n_nodes):
        active_mask = np.ones(n_nodes, dtype=bool)
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

        true_drops[i] = max(0, phi_0 - final_nc)

    bi = np.zeros(n_nodes)
    for i in range(n_nodes):
        adj_sub = np.delete(np.delete(adj_orig, i, 0), i, 1)
        bi[i] = max(0, phi_0 - calculate_natural_connectivity(adj_sub))
    if np.max(bi) > 1e-9: bi /= np.max(bi)
    try:
        ks = nx.core_number(G0)
        ec = nx.eigenvector_centrality(G0, max_iter=1000)
        si = np.array([ks[i] * ec[i] for i in range(n_nodes)])
    except:
        si = np.sum(G0_adj, axis=1)
    if np.max(si) > 1e-9: si /= np.max(si)

    r_gcn = SurBiRanking.predict_adaptive_r(G0)
    print(f"      -> GCN Predicted Adaptive r: {r_gcn:.4f}")

    if abs(r_gcn - 1.0) < 1e-4:
        print("      [Note] GCN predicted r is close to 1.0, Proposed model will degrade to Only BI. This is the true output of the model.")

    best_r_theory = 1.0
    min_ri_theory = float('inf')
    ratios = np.arange(0, 1.01, 0.1)

    for test_r in np.arange(0.0, 1.01, 0.05):
        scores = test_r * bi + (1 - test_r) * si
        order = np.argsort(scores)[::-1]
        phi_curve = []
        for ratio in ratios:
            num_remove = int(n_nodes * ratio)
            if num_remove == 0: phi_curve.append(1.0); continue
            if num_remove >= n_nodes: phi_curve.append(0.0); continue
            mask = np.ones(n_nodes, dtype=bool)
            mask[order[:num_remove]] = False
            alive_idx = np.where(mask)[0]
            sub_adj = G0_adj[np.ix_(alive_idx, alive_idx)]
            phi_curve.append(calculate_natural_connectivity(sub_adj) / phi_0)
        try:
            ri = np.trapezoid(phi_curve, ratios)
        except:
            ri = np.trapz(phi_curve, ratios)
        if ri < min_ri_theory:
            min_ri_theory = ri
            best_r_theory = test_r

    ablation_groups = {
        f'0. Theoretical Upper Bound (r={best_r_theory:.2f})': best_r_theory, 
        f'1. Proposed (GCN Output r={r_gcn:.2f})': r_gcn,  
        '2. w/o SI (Only BI, Fixed r=1.0)': 1.0,  
        '3. w/o BI (Only SI, Fixed r=0.0)': 0.0, 
        '4. w/o GCN (Fixed r=0.3)': 0.3 
    }

    print("\n[3/4] Executing 10%-step Static Attacks for all groups...")
    results = []

    for name, r_val in ablation_groups.items():
        scores = r_val * bi + (1 - r_val) * si

        tau, _ = stats.kendalltau(true_drops, scores)
        if np.isnan(tau): tau = 0.0

        order = np.argsort(scores)[::-1]
        phi_curve = []

        for ratio in ratios:
            num_remove = int(n_nodes * ratio)
            if num_remove == 0: phi_curve.append(1.0); continue
            if num_remove >= n_nodes: phi_curve.append(0.0); continue

            targets = order[:num_remove]
            mask = np.ones(n_nodes, dtype=bool)
            mask[targets] = False
            alive_idx = np.where(mask)[0]

            sub_adj = G0_adj[np.ix_(alive_idx, alive_idx)]
            phi_curve.append(calculate_natural_connectivity(sub_adj) / phi_0)

        try:
            ri = np.trapezoid(phi_curve, ratios)
        except:
            ri = np.trapz(phi_curve, ratios)

        results.append({
            'Ablation Group': name,
            "Kendall's Tau (↑)": round(tau, 4),
            'Resilience Integral RI (↓)': round(ri, 4),
            'NC Retained @ 10% Attack (↓)': round(phi_curve[1], 4)
        })

    print("\n[4/4] Ablation Study Results:\n")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    res_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(res_dir, exist_ok=True)
    save_path = os.path.join(res_dir, 'table_ablation_study.csv')
    df.to_csv(save_path, index=False)
    print(f"\n[Completed] Ablation study table saved to: {save_path}")


if __name__ == "__main__":
    n_threads = max(1, multiprocessing.cpu_count() - 1)
    with ThreadPool(n_threads) as pool:
        main(pool)