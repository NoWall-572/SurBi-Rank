import numpy as np

def critic_weights(F):
    f_min = np.min(F, axis=0)
    f_max = np.max(F, axis=0)
    denominator = f_max - f_min
    denominator[denominator == 0] = 1e-9

    X = (F - f_min) / denominator

    sigma = np.std(X, axis=0)

    corr_matrix = np.corrcoef(X.T)

    n_objs = F.shape[1]
    C = np.zeros(n_objs)

    for j in range(n_objs):
        sum_corr = np.sum(1 - corr_matrix[j, :])
        C[j] = sigma[j] * sum_corr

    weights = C / np.sum(C)
    return weights

def select_knee_point(F, weights):
    f_min = np.min(F, axis=0)
    f_max = np.max(F, axis=0)
    denominator = f_max - f_min
    denominator[denominator == 0] = 1e-9
    norm_F = (F - f_min) / denominator

    ideal_point = np.zeros(F.shape[1])

    weighted_sq_diff = weights * ((norm_F - ideal_point) ** 2)
    distances = np.sqrt(np.sum(weighted_sq_diff, axis=1))

    best_idx = np.argmin(distances)
    return best_idx
