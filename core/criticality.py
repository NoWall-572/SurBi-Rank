import numpy as np
import networkx as nx
import scipy.linalg as la
import torch
import warnings
import os

# 【修改这里】加上 core. 和 models. 前缀
from core.config import Config
from models.gcn_module import AdaptiveSurBiGCN, extract_features, normalize_adj

class SurBiRanking:
    _gcn_model = None  # 单例模式缓存模型

    @staticmethod
    def load_gcn_model():
        if SurBiRanking._gcn_model is None:
            # 动态获取项目根目录
            PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(PROJECT_ROOT, 'models', 'gcn_r_model.pth')

            if os.path.exists(model_path):
                model = AdaptiveSurBiGCN(num_features=5, hidden_dim=64)
                model.load_state_dict(torch.load(model_path))
                model.eval()
                SurBiRanking._gcn_model = model
            else:
                print(f"  [Warning] GCN 模型未找到 ({model_path})，将使用默认 r=0.5")
                return None
        return SurBiRanking._gcn_model

    @staticmethod
    def predict_adaptive_r(G):
        model = SurBiRanking.load_gcn_model()
        if model is None:
            return 0.5  # 默认值

        try:
            feat = extract_features(G)
            adj = normalize_adj(nx.to_numpy_array(G))
            with torch.no_grad():
                pred_r = model(feat, adj)
            return pred_r.item()
        except Exception as e:
            print(f"  [Error] GCN 推理失败: {e}，使用默认 r")
            return 0.5

    @staticmethod
    def natural_connectivity(adj):
        evals = la.eigvals(adj)
        evals = np.real(evals)
        n = len(adj)
        return np.log(np.sum(np.exp(evals)) / n + 1e-9)

    @staticmethod
    def compute_surbi(G, fixed_r=None):
        """
        计算 SurBi 重要性分数。
        :param G: NetworkX 图对象
        :param fixed_r: 如果提供，则强制使用该 r 值；否则使用 GCN 自适应预测。
        """
        n = len(G)
        adj = nx.to_numpy_array(G)

        # 1. 确定 r 值
        if fixed_r is not None:
            r = fixed_r
        else:
            r = SurBiRanking.predict_adaptive_r(G)
            # print(f"  [Info] 当前网络自适应 r 值: {r:.4f}")

        # 2. 计算 BI (全局) - 耗时部分
        phi_global = SurBiRanking.natural_connectivity(adj)
        bi_scores = np.zeros(n)

        # 优化：如果是超大图，这里可能需要近似，但对于 200 节点，精确计算还是可行的 (约 0.1s)
        for i in range(n):
            adj_sub = np.delete(np.delete(adj, i, axis=0), i, axis=1)
            phi_sub = SurBiRanking.natural_connectivity(adj_sub)
            bi_scores[i] = max(0, phi_global - phi_sub)

        if np.max(bi_scores) > 1e-9:
            bi_scores /= np.max(bi_scores)

        # 3. 计算 SI (局部)
        try:
            k_shell = nx.core_number(G)
            ec = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-04)
        except:
            k_shell = dict(G.degree())
            ec = nx.degree_centrality(G)

        si_scores = np.zeros(n)
        for i in range(n):
            # NetworkX 的节点 ID 必须是 0..N-1
            neighbors = list(G.neighbors(list(G.nodes())[i]))
            sum_ks = sum([k_shell[nbr] for nbr in neighbors])
            # 注意 ec 的键
            node_key = list(G.nodes())[i]
            si_scores[i] = sum_ks * ec.get(node_key, 0)

        if np.max(si_scores) > 1e-9:
            si_scores /= np.max(si_scores)

        # 4. 融合
        ni_scores = r * bi_scores + (1 - r) * si_scores
        return ni_scores