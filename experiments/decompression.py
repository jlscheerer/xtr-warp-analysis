from analysis import Engine, load_experiment_results, metric_round

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

from collections import defaultdict

def run_decompression_analysis():
    plt.figure(figsize=(14, 6))
    experiment_results = load_experiment_results(Engine.XTR_WARP, ["warp_ablation_decompression.json"])
    NPROBE_VALUES = sorted(list(set(experiment_results.view("nprobe"))))

    grouped_by_config = defaultdict(list)
    for entry in experiment_results:
        ablation_params = entry.view("ablation_params")
        decompression_fn = ablation_params["decompression_fn"]
        normalized_decompression = ablation_params["normalized_decompression"]
        assert not normalized_decompression or (decompression_fn == "explicit_decompress_py")
        
        config = f"{decompression_fn}-{normalized_decompression}"
        nprobe = entry.view("nprobe")
        latency = entry.latency.plaid_style(single_run=True)["Decompression"]
        grouped_by_config[config].append((nprobe, latency))
    grouped_by_config = dict(grouped_by_config)

    print(grouped_by_config)
    CONFIGS = ["explicit_decompress_py-True", "score_decompress_py-False", "score_decompress_cpp-False"]
    PRETTY_CONFIG_NAMES = {
        "explicit_decompress_py-True": "Decompress$_\\text{Explicit}$-Py",
        "score_decompress_py-False": "Decompress$_\\text{Score}$-Py",
        "score_decompress_cpp-False": "Decompress$_\\text{Score}$-CPP"
    }
    for config in CONFIGS:
        datapoints = grouped_by_config[config]
        xs, ys = [x[0] for x in datapoints], [x[1] for x in datapoints]
        assert xs == NPROBE_VALUES
        plt.plot(NPROBE_VALUES, ys, marker='o', label=PRETTY_CONFIG_NAMES[config])

        print(config, datapoints)

    MINOR, MAJOR = 20, 24

    ax = plt.gca()
    ax.yaxis.offsetText.set_fontsize(MINOR)
    plt.yticks([1000, 2000, 3000, 4000, 5000, 6000, 7000], fontsize=MINOR)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))

    plt.xticks(NPROBE_VALUES, fontsize=MINOR)
    plt.ylabel("Latency (ms)", fontsize=MAJOR)
    plt.xlabel("$n_\\text{probe}$", fontsize=MAJOR)
    plt.subplots_adjust(bottom=0.15)
    plt.grid()
    plt.legend(fontsize=MINOR)
    plt.xticks(NPROBE_VALUES, [str(x) if x != 2 else "" for x in NPROBE_VALUES], fontsize=MINOR)
    plt.savefig("output/scale_decompression.pdf")
    plt.show()


if __name__ == "__main__":
    run_decompression_analysis()