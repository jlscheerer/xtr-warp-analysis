from analysis import Engine, load_experiment_results, metric_round, DATASET_STATS

import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from scipy.optimize import curve_fit

CURVE_FIT = False

def sqrt_embeds(x, a, b, c):
    return a * np.sqrt(b * x) + c

def run_scale_datasets_analysis():
    experiment_results = load_experiment_results(Engine.XTR_WARP, ["warp_scale_datasets.json"])

    NUM_EMBEDDINGS = set()

    plt.figure(figsize=(14, 6))

    grouped_by_nprobe = defaultdict(list)
    for entry in experiment_results:
        nprobe, dataset_id = entry.view("nprobe"), entry.view("dataset_id")
        num_embeddings = DATASET_STATS[dataset_id]["embeddings"]
        latency = sum(entry.latency.plaid_style().values())
        grouped_by_nprobe[nprobe].append((num_embeddings, latency))
        NUM_EMBEDDINGS.add(num_embeddings)
    grouped_by_nprobe = dict(grouped_by_nprobe)
    NUM_EMBEDDINGS = sorted(list(NUM_EMBEDDINGS))

    for nprobe, results in grouped_by_nprobe.items():
        values = sorted(results, key=lambda x: x[0])
        xs, ys = [x[0] for x in values], [x[1] for x in values]
        assert xs == NUM_EMBEDDINGS
        ext = str(nprobe)
        plt.plot(np.log2(NUM_EMBEDDINGS), ys, label=f"$n_\\text{{probe}} = {ext}$", marker="o", linestyle="--")
        if nprobe == 32 and CURVE_FIT:
            # NOTE Almost *perfect* fit via sqrt_embeds!
            popt, pcov = curve_fit(sqrt_embeds, xs, ys)
            a, b, c = popt
            plt.plot(np.log2(NUM_EMBEDDINGS), sqrt_embeds(np.array(xs), a, b, c))

    MINOR, MAJOR = 20, 24
    plt.ylabel("Latency (ms)", fontsize=MAJOR)
    plt.yticks([60, 80, 100, 120, 140], fontsize=MINOR)
    plt.xlabel("Dataset Size (#Embeddings)", fontsize=MAJOR)
    plt.xticks([20, 22, 24, 26, 28], ["$2^{20}$", "$2^{22}$", "$2^{24}$", "$2^{26}$", "$2^{28}$"], fontsize=MINOR)
    plt.legend(fontsize=MINOR)
    plt.tick_params(axis="x", which="major", pad=10)
    plt.subplots_adjust(bottom=0.15)
    plt.grid()
    plt.savefig("output/scale_datasets.pdf")
    plt.show()

if __name__ == "__main__":
    run_scale_datasets_analysis()