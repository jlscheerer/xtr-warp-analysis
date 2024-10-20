from analysis import Engine, load_experiment_results, metric_round

import matplotlib.pyplot as plt

from collections import defaultdict

def run_scale_threads_analysis():
    experiment_results = load_experiment_results(Engine.XTR_WARP, ["warp_scale_threads.json"])
    
    NUM_THREADS = sorted(list(set(experiment_results.view("num_threads"))))

    plt.figure(figsize=(10, 8))

    grouped_by_nprobe = defaultdict(list)
    for entry in experiment_results:
        nprobe = entry.view("nprobe")
        num_threads = entry.view("num_threads")
        grouped_by_nprobe[nprobe].append((num_threads, metric_round(entry.latency.sum(), fact=1000)))
    grouped_by_nprobe = dict(grouped_by_nprobe)

    NPROBE_VALUES = [8, 16, 32]
    for nprobe in NPROBE_VALUES:
        results = grouped_by_nprobe[nprobe]
        values = sorted(results, key=lambda x: x[0])
        xs, ys = [x[0] for x in values], [x[1] for x in values]
        assert xs == NUM_THREADS
        ext = str(nprobe)
        plt.plot(NUM_THREADS, ys, label=f"$n_\\text{{probe}} = {ext}$", marker="o")

    MINOR, MAJOR = 20, 24
    plt.xscale("log")
    plt.yticks([40, 60, 80, 100, 120, 140, 160], fontsize=MINOR)
    plt.ylabel("Latency (ms)", fontsize=MAJOR)
    plt.gca().xaxis.set_minor_locator(plt.NullLocator())
    
    plt.xlabel("#Threads", fontsize=MAJOR)
    plt.tick_params(axis="x", which="major", pad=10)
    plt.xticks([2**i for i in range(0, 4 + 1)], [f"$2^{i}$" for i in range(0, 4 + 1)], fontsize=MINOR)
    plt.grid()
    plt.legend(fontsize=MINOR)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f"output/scale_threads.pdf")
    plt.show()

if __name__ == "__main__":
    run_scale_threads_analysis()