# 文件路径: experiments/exp3_static_topology.py

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pickle
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation

from core.config import Config
from core.network_env import SwarmSystem
from core.optimization import StaticTopologyProblem, GraphSmartSampling
from utils.decision_maker import critic_weights, select_knee_point


def main():
    print("=== Phase 3: Static Topology Optimization & Decision Making ===")

    # 1. 初始化物理环境
    print("[1/4] Generating Physical Swarm Environment...")
    swarm = SwarmSystem()
    swarm.generate_topology()

    # 2. 设置 NSGA-II 优化器
    print(f"[2/4] Starting NSGA-II Optimization (Pop: {Config.POP_SIZE}, Gen: {Config.N_GEN})...")
    problem = StaticTopologyProblem(swarm)
    algorithm = NSGA2(
        pop_size=Config.POP_SIZE,
        sampling=GraphSmartSampling(swarm),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(),
        eliminate_duplicates=True
    )

    start_time = time.time()
    res = minimize(
        problem,
        algorithm,
        ('n_gen', Config.N_GEN),
        verbose=True
    )
    print(f"Optimization finished in {time.time() - start_time:.2f} seconds.")

    if res.X is None:
        print("Error: No feasible solution found. Check constraints.")
        return

    # 3. CRITIC + Knee Point 自动化决策
    print("\n[3/4] Executing CRITIC Decision Making...")
    pareto_F = res.F
    pareto_X = res.X

    weights = critic_weights(pareto_F)
    print(f"  -> CRITIC Weights: f1(Conn)={weights[0]:.4f}, f2(Rel)={weights[1]:.4f}, f3(RI)={weights[2]:.4f}")

    best_idx = select_knee_point(pareto_F, weights)
    best_F = pareto_F[best_idx]
    print(f"  -> Selected Knee Point (ID: {best_idx}): f1={best_F[0]:.4f}, f2={best_F[1]:.4f}, f3={best_F[2]:.4f}")

    # 4. 解码并保存结果
    print("\n[4/4] Saving Optimal Topology G0...")
    adj_matrix = problem._decode(pareto_X[best_idx])

    candidates = {
        best_idx: {
            'id': best_idx,
            'adj': adj_matrix,
            'binary_X': pareto_X[best_idx],
            'metrics': best_F,
            'selected_by': ['CRITIC_Optimal'],
            'weights': weights
        }
    }

    # 使用绝对路径保存到项目根目录，确保 exp2 能准确读到
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 保存 swarm 物理环境 (供后续动态重构使用)
    with open(os.path.join(PROJECT_ROOT, 'data_step1_static.pkl'), 'wb') as f:
        pickle.dump({'swarm': swarm, 'pareto_F': pareto_F, 'pareto_X': pareto_X}, f)

    # 保存最优拓扑 G0
    save_path = os.path.join(PROJECT_ROOT, 'data_step2_candidates.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(candidates, f)

    print(f"Success! Optimal topology saved to '{save_path}'")


if __name__ == "__main__":
    main()