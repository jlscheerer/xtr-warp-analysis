from analysis import Engine, load_experiment_results, metric_round
from analysis.gantt_plot import group_latency_measurements
from analysis.gantt_config import *


import matplotlib.pyplot as plt

from collections import defaultdict

def _combine(xs, ys):
    return [(x[0], x[1] + y[1]) for x, y in zip(xs, ys) if x[0] == y[0]]

def run_scale_threads_by_step_analysis():
    NPROBE = 32
    experiment_results = load_experiment_results(Engine.XTR_WARP, ["warp_scale_threads.json"]).filter(
        nprobe=NPROBE
    )

    plt.figure(figsize=(10, 8))

    NUM_THREADS = sorted(list(set(experiment_results.view("num_threads"))))
    STEPS = experiment_results[0].latency.steps()

    grouped_by_threads = defaultdict(dict)
    for entry in experiment_results:
        num_threads = entry.view("num_threads")
        grouped_by_threads[num_threads] = entry.latency.avg_grouped_latency()

    for key in grouped_by_threads:
        grouped_by_threads[key] = group_latency_measurements(grouped_by_threads[key], WARP_LATENCY_GROUPS)


    grouped_by_step = defaultdict(list)
    for num_threads, measurements in grouped_by_threads.items():
        for key, value in measurements.items():
            grouped_by_step[key].append((num_threads, value))

    grouped_by_step["Decompression/Scoring$_\\text{(fused)}$"] = _combine(grouped_by_step["Decompression"], grouped_by_step["Scoring"])
    del grouped_by_step["Decompression"]
    del grouped_by_step["Scoring"]

    for step, datapoints in grouped_by_step.items():
        xs, ys = [x[0] for x in datapoints], [x[1] for x in datapoints]
        assert xs == NUM_THREADS
        plt.plot(NUM_THREADS, ys, marker='o', label=step, linestyle="--")

    MINOR, MAJOR = 20, 24
    plt.xscale('log')
    plt.yticks([20, 30, 40, 50], fontsize=MINOR)
    plt.ylabel("Latency (ms)", fontsize=MAJOR)
    plt.gca().xaxis.set_minor_locator(plt.NullLocator())

    plt.xlabel("#Threads", fontsize=MAJOR)
    plt.xticks([2**i for i in range(0, 4 + 1)], [f"$2^{i}$" for i in range(0, 4 + 1)], fontsize=MINOR)
    plt.grid()
    plt.legend(fontsize=MINOR)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f"output/scale_threads_per_step.pdf")
    plt.show()

if __name__ == "__main__":
    run_scale_threads_by_step_analysis()