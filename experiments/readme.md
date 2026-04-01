# Experiments Module: Evaluation & Validation

This folder contains the execution scripts for the core experiments presented in the paper. The logic follows a pipeline from **Model Training** and **Base Topology Generation** to **Multi-dimensional Resilience Assessment**.

---

## 📂 Functional Overview & Outputs

### 1. Fundamental Preparation
| Script | Description | Output Path |
| :--- | :--- | :--- |
| `exp_train_eval_gcn.py` | Trains the GCN for adaptive weight ($r$) prediction by simulating recursive cascading failures and command node connectivity. | `models/gcn_r_model.pth` |
| `exp_static_topology.py` | Performs multi-objective optimization (NSGA-II) and selects the optimal initial topology ($G_0$) via the CRITIC method. | `data_step1_static.pkl`, `data_step2_candidates.pkl` |

### 2. Experimental Branches

#### Branch A: Multi-Strategy Static Robustness (Part 1)
| Script | Description | Output Path |
| :--- | :--- | :--- |
| `exp_part1_sim.py` | Simulates static cascading failures on $G_0$ using four strategies: SurBi-Rank, Katz, K-shell, and Monte Carlo. | `results/part1_data.csv` |
| `exp_part1_plot.py` | Generates comparative line charts for Resilience Integral ($RI$), Half-Life Ratio ($HLR$), and Ranking Accuracy ($\tau$). | `results/fig_part1_RI.png`, `results/fig_part1_HLR.png`, `results/fig_part1_Tau.png` |

#### Branch B: Dynamic Lifecycle Resilience (Part 2)
| Script | Description | Output Path |
| :--- | :--- | :--- |
| `exp_part2_sim.py` | Simulates 5 rounds of "Attack-Reconfiguration" cycles, comparing the "Baseline" (no repair) and "Dynamic" (NSGA-II repair) modes. | `results/part2_data.csv` |
| `exp_part2_plot.py` | Generates a professional composite figure containing three parallel subplots showing the resilience evolution over the lifecycle. | `results/fig_part2_lifecycle.png` |

#### Branch C: Ablation Study
| Script | Description | Output Path |
| :--- | :--- | :--- |
| `exp_ablation_study.py` | Validates the necessity of SurBi-Rank components (BI, SI, GCN) by evaluating them in an independent, automated environment. | Console output & `results/table_ablation_study.csv` |

---

## 🚀 Recommended Execution Flow

To ensure all data dependencies are met, scripts should be executed in the following order:

### Step 1: Initialize Models
Run the GCN training script first to generate the adaptive weights model.
```bash
python exp_train_eval_gcn.py
```

### Step 2: Generate Benchmarks
Run the static optimization to obtain the initial optimal swarm topology ($G_0$).
```bash
python exp_static_topology.py
```

### Step 3: Run Evaluation Branches
The following branches can be run in parallel or independently:

*   **Part 1 Branch**: Compare ranking strategies.
    ```bash
    python exp_part1_sim.py && python exp_part1_plot.py
    ```
*   **Part 2 Branch**: Validate dynamic reconfiguration performance.
    ```bash
    python exp_part2_sim.py && python exp_part2_plot.py
    ```
*   **Ablation Branch**: Conduct component analysis.
    ```bash
    python exp_ablation_study.py
    ```

---

## 📝 Technical Notes
- **Simulation vs. Plotting**: Data generation (Sim) and visualization (Plot) are decoupled to allow for visual adjustments without re-running computationally expensive simulations.
- **Dependency**: Both Branch A and Branch B require the outputs from Step 2 (`.pkl` files) to maintain environmental consistency.
- **Hardware**: High-core CPUs are recommended for `exp_part2_sim.py` and `exp_ablation_study.py` as they utilize multi-threading for optimization.


### Key Design Features:
1.  **Logical Grouping**: Uses tables to categorize scripts by "Preparation" and "Branches," making it easy for readers to understand the modular design.
2.  **Clear Data Pipeline**: Every script lists its exact output path, addressing your concern about where files are generated.
3.  **Step-by-Step Guide**: The execution flow section uses code blocks and clear headings to prevent dependency errors.
4.  **Academic Tone**: Uses standard technical terminology (e.g., *Decoupled*, *Cascading Failure*, *Pareto Front*) suitable for a research project.
