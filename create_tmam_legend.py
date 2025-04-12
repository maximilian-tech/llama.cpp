import matplotlib.pyplot as plt

# Labels and corresponding colors
labels = [
    "light-ops",
    "heavy-ops",
    "machine-clear",
    "branch-mispredict",
    "memory-bound",
    "core-bound",
    "fetch-bandwidth",
    "backend-latency",
]

colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
]

# 80% opacity on each color bar
handles = [plt.Line2D([0], [0], color=color, lw=10, alpha=0.60) for color in colors]

# Optional: Nimbus Roman font
font = {'family': 'Nimbus Roman', 'size': 12}

# Legend-only figure
fig_legend = plt.figure(figsize=(5, 3))
fig_legend.legend(
    handles,
    labels,
    loc="center",
    frameon=False,
    ncol=1,
    handlelength=2.5,
    prop=font,
    title="Topdown L2 Metrics",
    title_fontproperties={'family': 'Nimbus Roman', 'size': 13},
)
fig_legend.tight_layout()
fig_legend.savefig("topdown_l2_legend.svg", format="svg", bbox_inches='tight', transparent=False)