"""Consistent figure styling for wearable biomarker papers."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def setup_style():
    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 10, 'figure.dpi': 300,
        'savefig.dpi': 300, 'savefig.bbox': 'tight'
    })

def forest_plot(names, ors, ci_low, ci_high, title='Forest Plot', filename='forest.png'):
    setup_style()
    fig, ax = plt.subplots(figsize=(8, max(4, len(names)*0.5)))
    colors = ['#d62728' if ci_low[i]>1 or ci_high[i]<1 else '#7f7f7f' for i in range(len(ors))]
    for i in range(len(names)):
        ax.plot([ci_low[i], ci_high[i]], [i, i], color=colors[i], linewidth=2)
        ax.plot(ors[i], i, 'o', color=colors[i], markersize=8)
    ax.axvline(1.0, color='black', linestyle='--', linewidth=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('Adjusted Odds Ratio per SD')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def dep_vs_anx_barplot(names, r_dep, r_anx, steiger_p, filename='dep_vs_anx.png'):
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, [abs(r) for r in r_dep], width, label='Depression', color='#1f77b4', alpha=0.8)
    ax.bar(x + width/2, [abs(r) for r in r_anx], width, label='Anxiety', color='#ff7f0e', alpha=0.8)
    for i, p in enumerate(steiger_p):
        if p < 0.05:
            ax.text(i, max(abs(r_dep[i]), abs(r_anx[i])) + 0.005,
                    '*' if p>0.01 else '**' if p>0.001 else '***', ha='center', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('|Pearson r|')
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
