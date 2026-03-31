import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from core.config import Config


class SwarmSystem:
    def __init__(self):
        self.coords = None
        self.node_types = []
        self.reliability_matrix = None

    def generate_topology(self):
        uav_x = np.random.uniform(0, Config.MAP_SIZE_X, Config.NUM_UAV)
        uav_y = np.random.uniform(0, Config.MAP_SIZE_Y, Config.NUM_UAV)
        uav_z = np.random.uniform(Config.Z_MIN_UAV, Config.Z_MAX_UAV, Config.NUM_UAV)
        uav_coords = np.column_stack((uav_x, uav_y, uav_z))

        usv_x = np.random.uniform(0, Config.MAP_SIZE_X, Config.NUM_USV)
        usv_y = np.random.uniform(0, Config.MAP_SIZE_Y, Config.NUM_USV)
        usv_z = np.zeros(Config.NUM_USV)
        usv_coords = np.column_stack((usv_x, usv_y, usv_z))

        self.coords = np.vstack((uav_coords, usv_coords))
        self.node_types = ['UAV'] * Config.NUM_UAV + ['USV'] * Config.NUM_USV

        self._compute_reliability_matrix()

    def _compute_reliability_matrix(self):
        n = Config.NUM_NODES
        dist_matrix = cdist(self.coords, self.coords, metric='euclidean')
        dist_safe = dist_matrix.copy()
        np.fill_diagonal(dist_safe, np.inf)

        signal_power = Config.P_TX * (dist_safe ** (-Config.ALPHA))

        jamming_power = np.zeros(n)
        if hasattr(Config, 'JAMMERS') and Config.JAMMERS:
            for jammer in Config.JAMMERS:
                j_pos = jammer['pos']
                j_p = jammer['power']
                d_j = np.linalg.norm(self.coords - j_pos, axis=1)
                d_j[d_j < 1.0] = 1.0
                jamming_power += j_p * (d_j ** (-Config.ALPHA))

        total_noise = Config.NOISE_POWER + jamming_power

        sinr_matrix = signal_power / (total_noise[:, None] + 1e-12)

        exponent = - Config.THETA_THRESHOLD / (sinr_matrix + 1e-12)
        self.reliability_matrix = np.exp(exponent)

        np.fill_diagonal(self.reliability_matrix, 0)
        self.reliability_matrix[self.reliability_matrix < 1e-6] = 0

        avg_rel = np.mean(self.reliability_matrix[self.reliability_matrix > 0])
        print(f"--- Communication Model (SINR & Jamming) ---")
        print(f"Number of jammers: {len(Config.JAMMERS)}")
        print(f"Average non-zero link reliability: {avg_rel:.4f}")
        print(f"Node with the worst interference ID: {np.argmax(jamming_power)} (Interference power: {np.max(jamming_power):.2e})")

    def get_graph_from_adjacency(self, adj_matrix):
        rows, cols = np.where(adj_matrix > 1e-6)
        G = nx.Graph()
        G.add_nodes_from(range(Config.NUM_NODES))

        pos_dict = {i: self.coords[i] for i in range(Config.NUM_NODES)}
        nx.set_node_attributes(G, pos_dict, 'pos')

        for r, c in zip(rows, cols):
            if r < c:
                G.add_edge(r, c, weight=adj_matrix[r, c])
        return G

    def calculate_reconfiguration_cost(self, new_adj, old_adj):
        added_mask = (new_adj > 1e-6) & (old_adj < 1e-6)
        rows, cols = np.where(added_mask)

        cost = 0.0
        v_uav = 20.0
        v_usv = 5.0

        for i, j in zip(rows, cols):
            if i >= j: continue

            dist = np.linalg.norm(self.coords[i] - self.coords[j])

            speed_i = v_uav if i < Config.NUM_UAV else v_usv
            speed_j = v_uav if j < Config.NUM_UAV else v_usv

            cost += (dist / speed_i) + (dist / speed_j)

        return cost