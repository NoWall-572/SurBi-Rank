import sys, os, numpy as np, networkx as nx, pickle, pandas as pd, copy
from multiprocessing.pool import ThreadPool
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import Config
from core.optimization import DynamicTopologyProblem, GraphSmartSampling
from core.metrics_calculator import compute_residual_metrics, get_strategy_ranking
from utils.decision_maker import critic_weights, select_knee_point
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation


def main(pool):
    print("=== Part 2 (SIM): Dynamic Lifecycle (Baseline vs Dynamic) ===")
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with open(os.path.join(PROJECT_ROOT, 'data_step1_static.pkl'), 'rb') as f:
        swarm = pickle.load(f)['swarm']
    with open(os.path.join(PROJECT_ROOT, 'data_step2_candidates.pkl'), 'rb') as f:
        candidates_data = pickle.load(f)
        first_key = list(candidates_data.keys())[0]
        G0_adj = candidates_data[first_key]['adj']

    adj_base = G0_adj.copy()
    adj_dyn = G0_adj.copy()
    dead_nodes = set()
    all_results = []

    print("[Init] Evaluating initial topology...")
    ri, hlr, tau = compute_residual_metrics(G0_adj, 'SurBi-Rank')
    for name in ['Baseline', 'Dynamic']:
        all_results.append({'Strategy': name, 'Round': 0, 'Stage': 'Init', 'RI': ri, 'HLR': hlr, 'Tau': tau})

    for r in range(1, 6):
        print(f"\n>>> Executing Round {r} ...")

        # --- Attack Phase ---
        G = nx.from_numpy_array(adj_dyn)
        scores = get_strategy_ranking(G, 'SurBi-Rank')
        for d in dead_nodes: scores[d] = -np.inf
        targets = np.argsort(scores)[::-1][:max(1, int(Config.NUM_NODES * 0.05))]
        dead_nodes.update(targets)

        for t in targets:
            adj_base[t, :] = 0;
            adj_base[:, t] = 0
            adj_dyn[t, :] = 0;
            adj_dyn[:, t] = 0

        ri_b, hlr_b, tau_b = compute_residual_metrics(adj_base, 'SurBi-Rank')
        all_results.append(
            {'Strategy': 'Baseline', 'Round': r, 'Stage': 'Attack', 'RI': ri_b, 'HLR': hlr_b, 'Tau': tau_b})

        ri_d, hlr_d, tau_d = compute_residual_metrics(adj_dyn, 'SurBi-Rank')
        all_results.append(
            {'Strategy': 'Dynamic', 'Round': r, 'Stage': 'Attack', 'RI': ri_d, 'HLR': hlr_d, 'Tau': tau_d})

        print(f"  [Repair] Starting dynamic restructuring...")

        swarm_attacked = copy.deepcopy(swarm)
        for d in dead_nodes:
            swarm_attacked.reliability_matrix[d, :] = 0
            swarm_attacked.reliability_matrix[:, d] = 0

        n_alive = Config.NUM_NODES - len(dead_nodes)
        new_target_edges = int(Config.TARGET_EDGE_COUNT * (n_alive / Config.NUM_NODES))

        problem = DynamicTopologyProblem(swarm_attacked, adj_dyn, list(dead_nodes), target_edge_count=new_target_edges)

        algorithm = NSGA2(
            pop_size=100,  
            sampling=GraphSmartSampling(swarm_attacked),  
            crossover=TwoPointCrossover(prob=0.8),
            mutation=BitflipMutation(prob=0.05),
            eliminate_duplicates=True
        )

        res = minimize(problem, algorithm, ('n_gen', 50), runner=pool, verbose=False)

        if res.F is not None and len(res.F) > 0:
            idx = select_knee_point(res.F, critic_weights(res.F))
            adj_dyn = problem._decode(res.X[idx])
            print(f"  [Repair] Refactoring successful!")
        else:
            print(f"  [Repair] Refactoring failed, keeping the original topology.")

        all_results.append(
            {'Strategy': 'Baseline', 'Round': r, 'Stage': 'Repair', 'RI': ri_b, 'HLR': hlr_b, 'Tau': tau_b})

        ri_d2, hlr_d2, tau_d2 = compute_residual_metrics(adj_dyn, 'SurBi-Rank')
        all_results.append(
            {'Strategy': 'Dynamic', 'Round': r, 'Stage': 'Repair', 'RI': ri_d2, 'HLR': hlr_d2, 'Tau': tau_d2})

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(PROJECT_ROOT, 'results', 'part2_data.csv'), index=False)
    print("\n[Done] Data saved to results/part2_data.csv")


if __name__ == "__main__":
    import multiprocessing

    with ThreadPool(max(1, multiprocessing.cpu_count() - 1)) as pool:
        main(pool)