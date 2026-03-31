# 文件路径: experiments/exp_part2_plot.py
import pandas as pd, matplotlib.pyplot as plt, numpy as np, os


def main():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'results', 'part2_data.csv'))

    metrics = ['RI', 'HLR', 'Tau']
    titles = ['Residual Resilience Integral (RI)', 'Residual Half-Life Ratio (HLR)', "Kendall's Tau (Accuracy)"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = {'Baseline': '#D62728', 'Dynamic': '#1F77B4'}
    markers = {'Baseline': 'o', 'Dynamic': 's'}
    linestyles = {'Baseline': '--', 'Dynamic': '-'}

    for i, m in enumerate(metrics):
        ax = axes[i]
        for strat in ['Baseline', 'Dynamic']:
            data = df[df['Strategy'] == strat]
            x_labels = ['Init']
            for r in range(1, 6): x_labels.extend([f'A{r}', f'R{r}'])
            x = np.arange(len(data))
            ax.plot(x, data[m], label=strat, color=colors[strat], marker=markers[strat], linestyle=linestyles[strat],
                    linewidth=2.5, markersize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45)
        for k in range(1, len(x), 2):
            ax.axvspan(k - 0.4, k + 0.4, color='red', alpha=0.05)
            if k + 1 < len(x): ax.axvspan(k + 0.6, k + 1.4, color='green', alpha=0.05)

        ax.set_title(titles[i], fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        if i == 1: ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'results', 'fig_part2_lifecycle.png'), dpi=300, bbox_inches='tight')
    print("Saved fig_part2_lifecycle.png")


if __name__ == "__main__":
    main()