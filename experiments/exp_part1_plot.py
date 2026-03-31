import pandas as pd, matplotlib.pyplot as plt, os


def main():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'results', 'part1_data.csv'))

    metrics = {'RI': 'Residual Resilience Integral (RI)', 'HLR': 'Residual Half-Life Ratio (HLR)',
               'Tau': "Kendall's Tau (Ranking Accuracy)"}
    colors = {'SurBi-Rank': 'red', 'Katz': 'blue', 'K-shell': 'green', 'Monte Carlo': 'orange'}
    markers = {'SurBi-Rank': 's', 'Katz': 'o', 'K-shell': '^', 'Monte Carlo': '*'}

    for metric, ylabel in metrics.items():
        plt.figure(figsize=(8, 6))
        for strat in df['Strategy'].unique():
            data = df[df['Strategy'] == strat]
            plt.plot(data['Failed_Ratio'], data[metric], label=strat, color=colors[strat], marker=markers[strat],
                     linewidth=2)

        plt.title(f"Evolution of {metric} under Cascading Failures", fontsize=14)
        plt.xlabel("Proportion of Failed Nodes (%)", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PROJECT_ROOT, 'results', f'fig_part1_{metric}.png'), dpi=300)
        print(f"Saved fig_part1_{metric}.png")


if __name__ == "__main__":
    main()