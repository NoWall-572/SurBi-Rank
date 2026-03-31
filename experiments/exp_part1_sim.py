# 文件路径: experiments/exp_part1_sim.py
import sys, os, pickle, numpy as np, pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.metrics_calculator import compute_residual_metrics, get_strategy_ranking


def main():
    print("=== Part 1 (SIM): Static Cascading Failure (Residual Metrics) ===")
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(PROJECT_ROOT, 'data_step2_candidates.pkl'), 'rb') as f:
        candidates_data = pickle.load(f)
        first_key = list(candidates_data.keys())[0]
        G0_adj = candidates_data[first_key]['adj']

    from core.metrics_calculator import calculate_natural_connectivity
    global_phi_0 = calculate_natural_connectivity(G0_adj)
    print(f"Global Initial NC (phi_0): {global_phi_0:.4f}")

    strategies = ['SurBi-Rank', 'Katz', 'K-shell', 'Monte Carlo']
    ratios = np.arange(0, 1.01, 0.05)
    all_results = []

    for strategy in strategies:
        print(f"Simulating Strategy: {strategy}")
        import networkx as nx
        G0 = nx.from_numpy_array(G0_adj)
        # 获取初始排名，用于决定移除顺序
        initial_order = np.argsort(get_strategy_ranking(G0, strategy))[::-1]

        for r in ratios:
            num_remove = int(len(G0_adj) * r)
            mask = np.ones(len(G0_adj), dtype=bool)
            mask[initial_order[:num_remove]] = False
            alive_idx = np.where(mask)[0]

            if len(alive_idx) < 2: break

            # 获取当前残余拓扑
            curr_adj = G0_adj[np.ix_(alive_idx, alive_idx)]

            # 【核心要求】在当前残余拓扑上，计算三个新指标
            ri, hlr, tau = compute_residual_metrics(curr_adj, strategy)
            all_results.append({'Strategy': strategy, 'Failed_Ratio': r * 100, 'RI': ri, 'HLR': hlr, 'Tau': tau})

    df = pd.DataFrame(all_results)
    os.makedirs(os.path.join(PROJECT_ROOT, 'results'), exist_ok=True)
    df.to_csv(os.path.join(PROJECT_ROOT, 'results', 'part1_data.csv'), index=False)
    print("Data saved to results/part1_data.csv")


if __name__ == "__main__":
    main()