from analysis import metric_round
from analysis.gantt_plot import GanttColor

import matplotlib.pyplot as plt
import numpy as np

def group_comparison_bar(groups, labels: list[str], colors: list[GanttColor], ignored=[], bar_width=0.4, show_improvement=False, bound=None, hatch_opacity=0.9):
    bucket_names = list(groups[0].keys())
    assert all(list(group.keys()) == bucket_names for group in groups)
    bucket_names = [x for x in bucket_names if x not in ignored]

    values = []
    for group in groups:
        values.append([group[key] for key in bucket_names])

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(bucket_names))
    for idx, value_list in enumerate(values):
        color, hatch = colors[idx]
        ypos = y_pos - bar_width / 2 + idx * bar_width
        ax.barh(ypos, value_list, bar_width, label=labels[idx], hatch=hatch,  edgecolor=(1, 1, 1, hatch_opacity), color=color)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(bucket_names)

    MINOR, MAJOR = 20, 24

    for idx, value_list in enumerate(values):
        for i, v in enumerate(value_list):
            ypos = i - bar_width / 2 + idx * (bar_width)
            text = f"{metric_round(v, fact=1)}ms"
            if show_improvement and idx != 0:
                text = f"{text} ({metric_round(values[0][i] / v, fact=1)}x)"
            ax.text(v + 10, ypos, text, va="center", fontsize=MAJOR, fontweight="bold")

    ax.invert_yaxis()
    plt.tick_params(axis="y", which="both", length=0)
    plt.tick_params(axis="both", which="major", labelsize=MAJOR)
    
    ax.set_xlabel("Latency (ms)", fontsize=MAJOR)
    ax.legend(fontsize=MINOR)
    if bound is not None:
        plt.xlim((0, bound))

    plt.tight_layout()
    return ax