from analysis import Engine, load_experiment_results

from collections import defaultdict

import matplotlib.pyplot as plt

LABELS = {
    "matrix_aggregate_py": "Matrix Aggregate",
    "hash_aggregate_cpp": "Hash Aggregate",
    "merge_aggregate_cpp": "Merge Aggregate"
}

def run_score_aggregation_analysis():
    plt.figure(figsize=(14, 6))
    experiment_results = load_experiment_results(Engine.XTR_WARP, ["warp_ablation_score_aggregation.json"])
    NPROBE_VALUES = sorted(list(set(experiment_results.view("nprobe"))))
    grouped_by_impl = defaultdict(list)
    for result in experiment_results:
        impl = result.view("ablation_params")["aggregate_fn"]
        nprobe = result.view("nprobe")
        latency = result.latency["Build Matrix"]
        grouped_by_impl[impl].append((nprobe, latency))
    grouped_by_impl = dict(grouped_by_impl)

    for impl, datapoints in grouped_by_impl.items():
        values = sorted(datapoints, key=lambda x: x[0])
        xs, ys = [x[0] for x in values], [x[1] for x in values]
        assert xs == NPROBE_VALUES
        plt.plot(NPROBE_VALUES, ys, marker='o', label=LABELS[impl])

    MINOR, MAJOR = 20, 24
    plt.xticks(NPROBE_VALUES, [str(x) if x != 2 else "" for x in NPROBE_VALUES], fontsize=MINOR)
    
    plt.yticks([0, 200, 400, 600, 800, 1000], fontsize=MINOR)

    # plt.title("Evaluation of Score Aggregation Implementations on LoTTE Pooled (Dev)", fontsize=16)
    plt.ylabel("Latency (ms)", fontsize=MAJOR)
    plt.xlabel("$n_\\text{probe}$", fontsize=MAJOR)
    plt.grid()
    plt.legend(fontsize=MINOR)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f"output/score_aggregation.pdf")
    plt.show()

if __name__ == "__main__":
    run_score_aggregation_analysis()