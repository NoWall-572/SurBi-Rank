## 📖 Overview
This project addresses the communication network resilience of large-scale heterogeneous unmanned systems (UAV/USV swarms) operating in adversarial environments. The proposed framework integrates GCN-based intelligence with multi-objective optimization to ensure mission continuity under targeted disruptions.

The system operates in a two-stage process:
1.  **Adaptive Criticality Assessment**: The **SurBi-Rank** algorithm fuses Birnbaum Importance ($BI$) and Surrounding Influence ($SI$). A Graph Convolutional Network (GCN) adaptively predicts the fusion weight $r$ based on the current topological state and environmental interference levels.
2.  **Robust Topology Optimization**: Using **NSGA-II/III**, the project optimizes swarm connections across structural, communication, and task layers. It handles static deployment and dynamic reconfiguration by balancing connectivity, reliability, resilience, and kinematic costs.

---

## 📂 Project Structure

The project is modularized into four primary directories:

### ⚙️ [core/](./core) (Backend Logic)
The foundational engine of the project. It encapsulates the physical environment, communication physics, and optimization problem definitions.
*   **Key Roles**: 3D/2D coordinate generation, SINR-based reliability, jamming models, and core metrics ($RI, HLR$).
*   **Documentation**: See [core/README.md](./core/README.md) for detailed mathematical and physical models.

### 🧪 [experiments/](./experiments) (Execution Entry Points)
The experimental suite for training models and validating performance.
*   **Key Roles**: GCN training, static $G_0$ generation, multi-strategy cascading failure comparison, dynamic lifecycle simulation, and ablation studies.
*   **Documentation**: See [experiments/README.md](./experiments/README.md) for script execution orders and dependency maps.

### 🧠 [models/](./models) (Neural Network Architectures)
Deep learning model definitions.
*   **File**: `gcn_module.py`
*   **Details**: Implements `AdaptiveSurBiGCN`, which aggregates nodal topological features and uses Global Mean Pooling to output the adaptive weighting scalar $r$.

### 🛠️ [utils/](./utils) (Decision Support Tools)
General-purpose mathematical tools for multi-objective decision making.
*   **File**: `decision_maker.py`
*   **Details**: Implements the **CRITIC Method** for objective weight allocation and **Knee Point Identification** to select the final optimal solution from the Pareto front.

---

## 🛠 Tech Stack
*   **Language**: Python 3.10+
*   **Deep Learning**: PyTorch
*   **Optimization**: Pymoo
*   **Graph Analysis**: NetworkX & SciPy
*   **Data Handling**: Pandas, NumPy, Matplotlib

---

## 🚀 Quick Start Guide

To reproduce the results:

1.  **Environment Setup**: Install dependencies:
    ```bash
    pip install torch pymoo networkx pandas matplotlib scipy
    ```
2.  **Model Training**: Run `experiments/exp_train_eval_gcn.py` to train the GCN.
3.  **Topology Generation**: Run `experiments/exp_static_topology.py` to generate the initial optimal network ($G_0$).
4.  **Simulation & Plotting**: Execute the scripts in `experiments/` to generate comparative data and professional figures.

---

## 📐 Mathematical Basis
The SurBi-Rank criticality score is defined as:
$$Score = r \cdot BI + (1 - r) \cdot SI$$
where $r$ is the GCN-predicted adaptive weight. Resilience is quantified using the **Resilience Integral (RI)** and **Half-Life Ratio (HLR)** based on **Natural Connectivity ($\Phi$)**.
