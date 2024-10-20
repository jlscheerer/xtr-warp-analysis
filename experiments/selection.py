from analysis import Engine, load_experiment_results, metric_round

import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict

def run_selection_analysis():
    experiment_results = load_experiment_results(Engine.XTR_WARP, ["warp_ablation_selection.json"])
    NPROBE_VALUES = sorted(list(set(experiment_results.view("nprobe"))))

    plt.figure(figsize=(14, 6))

    grouped_by_config = defaultdict(list)
    for entry in experiment_results:
        ablation_params = entry.view("ablation_params")
        selection_fn = ablation_params["selection_fn"]
        compute_mse_via_reduce = ablation_params["compute_mse_via_reduce"]

        latency = entry.latency.plaid_style(single_run=True)
        mse_via_reduce_latency = latency["MSE via Reduction"]
        if not compute_mse_via_reduce:
            assert mse_via_reduce_latency < 0.01
        
        selection_latency = latency["top-k Precompute"]

        config = f"{selection_fn}-{compute_mse_via_reduce}"
        nprobe = entry.view("nprobe")
        total_latency = selection_latency + mse_via_reduce_latency
        grouped_by_config[config].append((nprobe, total_latency))
    grouped_by_config = dict(grouped_by_config)

    CONFIGS = ["warp_select_py-True", "warp_select_py-False", "warp_select_cpp-False"]
    PRETTY_CONFIG_NAMES = {
        "warp_select_py-True": "top-$k'$-Score-Py",
        "warp_select_py-False": "WARP$_\\text{SELECT}$-Py",
        "warp_select_cpp-False": "WARP$_\\text{SELECT}$-CPP",
    }
    for config in CONFIGS:
        datapoints = grouped_by_config[config]
        xs, ys = [x[0] for x in datapoints], [x[1] for x in datapoints]
        assert xs == NPROBE_VALUES
        print(PRETTY_CONFIG_NAMES[config], ys)
        plt.plot(NPROBE_VALUES, ys, marker='o', label=PRETTY_CONFIG_NAMES[config])

    MINOR, MAJOR = 20, 24

    plt.xticks(NPROBE_VALUES, [str(x) if x != 2 else "" for x in NPROBE_VALUES], fontsize=MINOR)
    plt.ylabel("Latency (ms)", fontsize=MAJOR)
    plt.xlabel("$n_\\text{probe}$", fontsize=MAJOR)
    plt.subplots_adjust(bottom=0.15)
    plt.grid()
    plt.legend(fontsize=MINOR)
    plt.yticks([0, 50, 100, 150, 200, 250, 300, 350], fontsize=MINOR)
    plt.savefig("output/scale_selection.pdf")
    plt.show()

if __name__ == "__main__":
    run_selection_analysis()