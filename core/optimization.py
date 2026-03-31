import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from core.config import Config
from scipy.spatial.distance import cdist


class GraphSmartSampling(Sampling):
    def __init__(self, swarm):
        super().__init__()
        self.swarm = swarm

    def _do(self, problem, n_samples, **kwargs):
        n_vars = problem.n_vars
        X = np.zeros((n_samples, n_vars), dtype=int)
        idx_map_rev = {v: k for k, v in enumerate(problem.idx_map)}
        safe_rel = np.where(self.swarm.reliability_matrix < 1e-9, 1e-9, self.swarm.reliability_matrix)
        dist_matrix = -np.log(safe_rel)
        G_full = nx.Graph()
        G_full.add_nodes_from(range(Config.NUM_NODES))
        rows, cols = np.where(self.swarm.reliability_matrix > 1e-9)
        for r, c in zip(rows, cols):
            if r < c: G_full.add_edge(r, c, weight=dist_matrix[r, c])
        target_n = getattr(problem, 'target_edge_count', Config.TARGET_EDGE_COUNT)
        all_edges_set = set(tuple(sorted(e)) for e in G_full.edges())
        for i in range(n_samples):
            mst_edges = list(nx.minimum_spanning_edges(G_full, weight='weight', data=False))
            final_edges = set(tuple(sorted(e)) for e in mst_edges)
            edges_needed = target_n - len(final_edges)
            if edges_needed > 0:
                available_edges = list(all_edges_set - final_edges)
                if len(available_edges) >= edges_needed:
                    indices = np.random.choice(len(available_edges), size=edges_needed, replace=False)
                    for idx in indices: final_edges.add(available_edges[idx])
            elif len(final_edges) > target_n:
                edges_list = list(final_edges)
                to_keep_idx = np.random.choice(len(edges_list), size=target_n, replace=False)
                final_edges = set(edges_list[k] for k in to_keep_idx)
            for u, v in final_edges:
                if (u, v) in idx_map_rev: X[i, idx_map_rev[(u, v)]] = 1
        return X


class BaseTopologyProblem(ElementwiseProblem):
    def calculate_natural_connectivity(self, adj):
        if len(adj) == 0: return 0.0
        try:
            evals = np.real(np.linalg.eigvals(adj))
            return np.log(np.sum(np.exp(evals)) / len(adj) + 1e-9)
        except:
            return 0.0


class StaticTopologyProblem(BaseTopologyProblem):
    def __init__(self, swarm):
        self.n_nodes = Config.NUM_NODES
        self.n_vars = int(self.n_nodes * (self.n_nodes - 1) / 2)
        super().__init__(n_var=self.n_vars, n_obj=3, n_eq_constr=1, n_ieq_constr=1, xl=0, xu=1, vtype=int)
        self.swarm = swarm
        self.idx_map = [(i, j) for i in range(self.n_nodes) for j in range(i + 1, self.n_nodes)]

    def _decode(self, x):
        adj = np.zeros((self.n_nodes, self.n_nodes))
        active_indices = np.where(x > 0.5)[0]
        rows = [self.idx_map[k][0] for k in active_indices]
        cols = [self.idx_map[k][1] for k in active_indices]
        reliabilities = self.swarm.reliability_matrix[rows, cols]
        adj[rows, cols] = reliabilities
        adj[cols, rows] = reliabilities
        return adj

    def _evaluate(self, x, out, *args, **kwargs):
        adj = self._decode(x)
        n = self.n_nodes

        out["H"] = [np.sum(x) - Config.TARGET_EDGE_COUNT]
        uav_degrees = np.sum(adj[:Config.NUM_UAV, :] > 0, axis=1)
        total_power = np.sum(Config.UAV_HOVER_POWER + Config.UAV_COMM_POWER * uav_degrees)
        g1 = (total_power * Config.MISSION_TIME) - (Config.UAV_BATTERY_CAPACITY * Config.NUM_UAV)
        out["G"] = [g1]

        try:
            evals = np.linalg.eigvalsh(np.diag(np.sum(adj, axis=1)) - adj)
            f1 = 1.0 - (evals[1] / float(n)) if len(evals) > 1 else 1.0
        except:
            f1 = 1.0

        adj_safe = np.where(adj < 1e-9, 1e-9, adj)
        dist_matrix = np.where(adj < 1e-9, 0, -np.log(adj_safe))
        d_mat = dijkstra(csgraph=csr_matrix(dist_matrix), directed=False)
        rel_mat = np.exp(-np.where(np.isinf(d_mat), 1000.0, d_mat))
        f2 = 1.0 - ((np.sum(rel_mat) - n) / (n * (n - 1)))

        scores = np.sum(adj, axis=1)
        order = np.argsort(scores)[::-1]
        ratios = [0.0, 0.2, 0.4, 0.6, 0.8]
        integral_sum = 0.0
        phi_0 = self.calculate_natural_connectivity(adj)
        for r in ratios:
            num_remove = int(n * r)
            if num_remove == 0: integral_sum += 1.0; continue
            mask = np.ones(n, dtype=bool);
            mask[order[:num_remove]] = False
            remain_idx = np.where(mask)[0]
            if len(remain_idx) > 0:
                curr_sub = adj[np.ix_(remain_idx, remain_idx)]
                integral_sum += self.calculate_natural_connectivity(curr_sub) / (phi_0 + 1e-9)
        f3 = 1.0 - (integral_sum / len(ratios))

        out["F"] = [f1, f2, f3]


class DynamicTopologyProblem(BaseTopologyProblem):
    def __init__(self, swarm, prev_adj, attacked_nodes, target_edge_count=None):
        self.n_nodes = Config.NUM_NODES
        self.n_vars = int(self.n_nodes * (self.n_nodes - 1) / 2)
        super().__init__(n_var=self.n_vars, n_obj=4, n_eq_constr=1, n_ieq_constr=0, xl=0, xu=1, vtype=int)
        self.swarm = swarm
        self.prev_adj = prev_adj
        self.attacked_nodes = set(attacked_nodes)
        self.target_edge_count = target_edge_count or Config.TARGET_EDGE_COUNT
        self.idx_map = [(i, j) for i in range(self.n_nodes) for j in range(i + 1, self.n_nodes)]
        self.dist_matrix = cdist(swarm.coords, swarm.coords)

    def _decode(self, x):
        adj = np.zeros((self.n_nodes, self.n_nodes))
        active_indices = np.where(x > 0.5)[0]
        rows = [self.idx_map[k][0] for k in active_indices]
        cols = [self.idx_map[k][1] for k in active_indices]
        reliabilities = self.swarm.reliability_matrix[rows, cols]
        final_weights = np.where(reliabilities < 0.01, 0.1, reliabilities)
        for idx in range(len(rows)):
            if rows[idx] in self.attacked_nodes or cols[idx] in self.attacked_nodes:
                final_weights[idx] = 0.0
        adj[rows, cols] = final_weights
        adj[cols, rows] = final_weights
        return adj

    def _evaluate(self, x, out, *args, **kwargs):
        adj = self._decode(x)
        n = self.n_nodes

        out["H"] = [np.sum(x) - self.target_edge_count]

        alive_nodes = [i for i in range(n) if i not in self.attacked_nodes]
        n_alive = len(alive_nodes)
        if n_alive < 2:
            out["F"] = [1.0, 1.0, 1.0, 1e5]
            return

        adj_alive = adj[np.ix_(alive_nodes, alive_nodes)]

        try:
            evals = np.linalg.eigvalsh(np.diag(np.sum(adj_alive, axis=1)) - adj_alive)
            f1 = 1.0 - (evals[1] / float(n_alive)) if len(evals) > 1 else 1.0
        except:
            f1 = 1.0

        adj_safe = np.where(adj_alive < 1e-9, 1e-9, adj_alive)
        dist_matrix = np.where(adj_alive < 1e-9, 0, -np.log(adj_safe))
        d_mat = dijkstra(csgraph=csr_matrix(dist_matrix), directed=False)
        rel_mat = np.exp(-np.where(np.isinf(d_mat), 1000.0, d_mat))
        f2 = 1.0 - ((np.sum(rel_mat) - n_alive) / (n_alive * (n_alive - 1)) if n_alive > 1 else 0)

        scores = np.sum(adj_alive, axis=1)
        order = np.argsort(scores)[::-1]
        ratios = [0.0, 0.2, 0.4, 0.6, 0.8]
        integral_sum = 0.0
        phi_0 = self.calculate_natural_connectivity(adj_alive)
        for r in ratios:
            num_remove = int(n_alive * r)
            if num_remove == 0: integral_sum += 1.0; continue
            mask = np.ones(n_alive, dtype=bool);
            mask[order[:num_remove]] = False
            remain_idx = np.where(mask)[0]
            if len(remain_idx) > 0:
                curr_sub = adj_alive[np.ix_(remain_idx, remain_idx)]
                integral_sum += self.calculate_natural_connectivity(curr_sub) / (phi_0 + 1e-9)
        f3 = 1.0 - (integral_sum / len(ratios))

        added_mask = np.triu((adj > 1e-6) & (self.prev_adj < 1e-6))
        rows, cols = np.where(added_mask)
        f4 = 0.0
        for r, c in zip(rows, cols):
            f4 += (self.dist_matrix[r, c] / (20.0 if r < Config.NUM_UAV else 5.0)) + (
                        self.dist_matrix[r, c] / (20.0 if c < Config.NUM_UAV else 5.0))

        out["F"] = [f1, f2, f3, f4]