import numpy as np
import os
import pickle


class Config:
    # --- 1. 大规模场景设置 ---
    NUM_UAV = 120
    NUM_USV = 80
    NUM_NODES = NUM_UAV + NUM_USV

    MAP_SIZE_X = 2000
    MAP_SIZE_Y = 2000
    Z_MIN_UAV = 50
    Z_MAX_UAV = 500

    # --- 2. 通信模型 ---
    COMM_RANGE = 1000.0
    ALPHA = 2.1
    THETA_THRESHOLD = 0.1
    NOISE_POWER = 1e-9
    P_TX = 5.0

    JAMMERS = [
        {'pos': np.array([1000, 1000, 0]), 'power': 10.0},
        {'pos': np.array([500, 1500, 200]), 'power': 5.0},
        {'pos': np.array([1500, 500, 0]), 'power': 5.0}
    ]

    # --- 3. GCN ---
    R_VALUE = 0.3
    GCN_HIDDEN_DIM = 64
    GCN_LR = 0.01
    GCN_EPOCHS = 200
    TRAIN_GRAPH_NUM = 50
    TRAIN_GRAPH_SIZE = 50

    # --- 4. 优化算法 ---
    POP_SIZE = 100
    N_GEN = 50

    # 边数约束 (等式)
    TARGET_EDGE_COUNT = int(NUM_NODES * 8 / 2)
    if os.path.exists('../optimized_hyperparams.pkl'):
        try:
            with open('../optimized_hyperparams.pkl', 'rb') as f:
                params = pickle.load(f)
                if 'N_star' in params:
                    TARGET_EDGE_COUNT = int(params['N_star'])
        except:
            pass
    MAX_AVG_DEGREE = (TARGET_EDGE_COUNT * 2) / NUM_NODES
    MAX_EDGES = TARGET_EDGE_COUNT

    # --- 5. 能量模型参数 (新增，对应 Eq. 39) ---
    # 假设参数，需保证约束有意义（即全连通可能耗尽电量，稀疏则不会）
    UAV_HOVER_POWER = 200.0  # W (悬停功率)
    UAV_COMM_POWER = 2.0  # W per link (单链路通信增加的功率)
    MISSION_TIME = 1800.0  # s (任务持续时间 30分钟)
    # 初始电池电量 (J): 假设能支持悬停 40分钟
    UAV_BATTERY_CAPACITY = (UAV_HOVER_POWER + UAV_COMM_POWER * 5) * 2400.0

    # --- 6. 任务 (异构) ---
    np.random.seed(42)
    PHASE_REQS = [
        (20, 60),
        (30, 40),
        (40, 30)
    ]

    PHASES = []
    _uav_indices = np.arange(NUM_UAV)
    _usv_indices = np.arange(NUM_UAV, NUM_NODES)

    for req_usv, req_uav in PHASE_REQS:
        p_uav = np.random.choice(_uav_indices, size=req_uav, replace=False)
        p_usv = np.random.choice(_usv_indices, size=req_usv, replace=False)
        PHASES.append(np.concatenate([p_uav, p_usv]))

    BASE_FAILURE_RATE = 0.001
    PHASE_DURATION = 10.0
    STRESS_FACTOR = 1.5