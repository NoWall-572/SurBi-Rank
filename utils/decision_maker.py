# 文件路径: utils/decision_maker.py

import numpy as np

def critic_weights(F):
    """
    CRITIC (Criteria Importance Through Intercriteria Correlation) Method
    对应论文 Eq. (40)
    """
    # 1. 数据归一化 (Min-Max)
    f_min = np.min(F, axis=0)
    f_max = np.max(F, axis=0)
    denominator = f_max - f_min
    denominator[denominator == 0] = 1e-9

    X = (F - f_min) / denominator

    # 2. 计算标准差 sigma_j
    sigma = np.std(X, axis=0)

    # 3. 计算相关系数矩阵 r_jk
    corr_matrix = np.corrcoef(X.T)

    # 4. 计算信息量 C_j
    n_objs = F.shape[1]
    C = np.zeros(n_objs)

    for j in range(n_objs):
        sum_corr = np.sum(1 - corr_matrix[j, :])
        C[j] = sigma[j] * sum_corr

    # 5. 计算最终权重 w_j
    weights = C / np.sum(C)
    return weights

def select_knee_point(F, weights):
    """
    Knee Point Identification via Weighted Distance to Ideal Point
    对应论文 Eq. (41)
    """
    # 1. 归一化
    f_min = np.min(F, axis=0)
    f_max = np.max(F, axis=0)
    denominator = f_max - f_min
    denominator[denominator == 0] = 1e-9
    norm_F = (F - f_min) / denominator

    # 2. 理想点 (Min问题的理想点是全0)
    ideal_point = np.zeros(F.shape[1])

    # 3. 计算加权欧氏距离
    weighted_sq_diff = weights * ((norm_F - ideal_point) ** 2)
    distances = np.sqrt(np.sum(weighted_sq_diff, axis=1))

    # 4. 选择距离最小的点
    best_idx = np.argmin(distances)
    return best_idx