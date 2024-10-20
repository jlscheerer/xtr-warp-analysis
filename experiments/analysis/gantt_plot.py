import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

#https://matplotlib.org/stable/gallery/shapes_and_collections/hatch_style_reference.html

class GanttColor:
    BLUE = ("#1f77b4", "/")
    ORANGE = ("#ff7f0e", "+")
    GREEN = ("#2ca02c", "x")
    RED = ("#d62728", "||")

    PINK = ("hotpink", "-")
    YELLOW = ("gold", ".")
    PURPLE = ("#9467bd", "o")
    MAGENTA = ("magenta", "\\")

    CYAN = ("cyan", "O")
    OLIVE = ("olive", "--")


def make_uniform_comparison_gantt_plot(execution_times, labels, colors: dict[str, GanttColor], bound=None, show_hatches=True,
                                       hatch_opacity=0.9, barlabel="$time$ms", center_align=False, center_align_pad=90, label_offset=10):
    keys = list(execution_times[0].keys())
    assert all(list(exec_times.keys()) == keys for exec_times in execution_times)
    
    times = [[] for _ in keys]
    for exec_times in execution_times:
        for idx, (key, value) in enumerate(exec_times.items()):
            assert key == keys[idx]
            times[idx].append(value)

    # Reverse labels and values to display the results descending.
    times = [[*reversed(time)] for time in times]
    labels = [*reversed(labels)]

    fig, ax = plt.subplots(figsize=(14, 2 + len(execution_times)))
    cumsum = np.zeros_like(times[0])
    for key, time_list in zip(keys, times):
        color, hatch = colors[key]
        ax.barh(labels, time_list, left=cumsum, color=color, hatch=hatch if show_hatches else None,  edgecolor=(1, 1, 1, hatch_opacity), label=key)
        cumsum += time_list
    

    MINOR, MAJOR = 20, 24

    ax.set_xlabel("Latency (ms)", fontsize=MAJOR)
    if bound is not None:
        plt.xlim((0, bound))
    plt.tick_params(axis="y", which="both", length=0)
    if center_align:
        labels = [label.get_text() for label in ax.get_yticklabels()]
        ax.set_yticklabels(labels, ha='center', multialignment='center')
        ax.tick_params(axis='y', which='major', pad=center_align_pad)
    plt.tick_params(axis="both", which="major", labelsize=MAJOR)
    
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False, useMathText=False))
    ax.ticklabel_format(style="plain", axis="x")

    if barlabel != None:
        if not isinstance(barlabel, list):
            barlabel = [barlabel] * len(labels)
        for idx, (label, total_time) in enumerate(zip(labels, cumsum)):
            content = barlabel[len(labels) - 1 - idx].replace("$time$", f"{total_time:.0f}")
            ax.text(total_time + label_offset, idx, content, va="center", ha="left", fontsize=MAJOR, fontweight="bold")

    plt.tight_layout()
    ax.legend(fontsize=MINOR)
    return ax

def group_latency_measurements(measurements, groups, ignored=[]):
    rev_map = {value: key for key, values in groups.items() for value in values}
    execution_times = {key: 0.0 for key in groups.keys()}
    for key, value in measurements.items():
        if key in ignored:
            continue
        execution_times[rev_map[key]] += value
    return execution_times