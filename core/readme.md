# Core Module: Resilient Swarm Intelligence & Topology Optimization

This folder contains the backbone logic for the **Aerial-Marine Swarm Resilient Network** project. It integrates physical environment simulation, multi-layer criticality assessment, and multi-objective optimization engines.

---

## 📂 File Structure & Overview

| File | Primary Role | Key Components |
| :--- | :--- | :--- |
| `config.py` | **Global Configuration** | Hardware specs, mission parameters, and GCN hyperparameters. |
| `network_env.py` | **Physical Simulation** | 3D/2D spatial distribution, SINR-based reliability, and Jamming models. |
| `criticality.py` | **Assessment Engine** | SurBi-Rank algorithm implementation and GCN inference logic. |
| `optimization.py` | **Optimization Framework** | Static/Dynamic topology problems and Smart Sampling for NSGA-II/III. |
| `metrics_calculator.py` | **Evaluation Tools** | Resilience Integral ($RI$), Half-Life Ratio ($HLR$), and Kendall's $\tau$ calculation. |

---

## 🛠 Key Functionalities

### 1. Physical Environment & Communication Model (`network_env.py`)
Implements a heterogeneous 3D environment for UAVs and 2D sea-surface for USVs.
*   **SINR-based Reliability**: Calculates link success probability $E_{ij}$ based on path loss ($\alpha$), transmit power, and thermal noise.
*   **Adversarial Jamming**: Simulates geographic interference from multiple jammers, affecting the local signal quality dynamically.
*   **Reconfiguration Cost**: Quantifies the kinematic energy required for nodes to adjust positions for new topology links.

### 2. SurBi-Rank Assessment (`criticality.py`)
The core algorithm for identifying critical nodes by fusing global and local perspectives:
*   **Birnbaum Importance (BI)**: Captures the global impact on Natural Connectivity when a node is removed.
*   **Surrounding Influence (SI)**: Aggregates local structural features ($K$-shell and Eigenvector Centrality).
*   **GCN Adaptive Fusion**: Uses a Graph Convolutional Network to predict the optimal trade-off weight $r$:
    $$Score = r \cdot BI + (1-r) \cdot SI$$

### 3. Optimization Engines (`optimization.py`)
Defines the search space and objectives for the evolutionary algorithms:
*   **Static Optimization**: Minimizes connectivity gap, reliability loss, and structural vulnerability.
*   **Dynamic Reconfiguration**: Introduces a 4th objective—**Kinematic Cost**—to ensure rapid recovery under physical constraints.
*   **Smart Sampling**: A graph-theory-based initialization strategy that ensures every candidate solution starts as a connected Spanning Tree (MST).

### 4. Advanced Resilience Metrics (`metrics_calculator.py`)
Beyond simple connectivity, this module evaluates the "Quality of Survivability":
*   **Resilience Integral (RI)**: The area under the Natural Connectivity degradation curve during cascading failures.
*   **Half-Life Ratio (HLR)**: The fraction of nodes that must be destroyed to reduce network performance by 50%.
*   **Kendall's Tau ($\tau$)**: Measures the alignment between predicted criticality and ground-truth cascading damage.

---

## 📐 Mathematical Foundations

The module is built upon the following metrics:

1.  **Natural Connectivity ($\Phi$)**: 
    $$\Phi(G) = \ln \left( \frac{1}{N} \sum_{i=1}^N e^{\lambda_i} \right)$$
2.  **Cascading Failure Logic**: Simulates $S \to C \to T$ (Structural $\to$ Communication $\to$ Task) layer failures where nodes with $degree \le 1$ lose task functionality.

---

## 🚀 Usage Note

These scripts are intended to be called by the experiment entry points in the `experiments/` directory. 
- Ensure `gcn_r_model.pth` is present in the `models/` folder for `criticality.py` to perform adaptive inference.
- Parameters in `config.py` should be tuned based on the specific swarm scale (e.g., `NUM_NODES = 200`).

---
