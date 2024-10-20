from analysis import Engine, load_experiment_results, metric_round

import matplotlib.pyplot as plt

from collections import defaultdict

PLAID_COLORS = {10: "tab:green", 100: "tab:orange", 1000: "tab:red"}

def add_warp_results(experiment_results, nprobe_values):
    NUM_THREADS = sorted(list(set(experiment_results.view("num_threads"))))
    grouped_by_nprobe = defaultdict(list)
    for entry in experiment_results:
        nprobe = entry.view("nprobe")
        if nprobe not in nprobe_values:
            continue
        num_threads = entry.view("num_threads")
        grouped_by_nprobe[nprobe].append((num_threads, metric_round(sum(entry.latency.plaid_style().values()), fact=1)))
    grouped_by_nprobe = dict(grouped_by_nprobe)

    for nprobe, results in grouped_by_nprobe.items():
        values = sorted(results, key=lambda x: x[0])
        xs, ys = [x[0] for x in values], [x[1] for x in values]
        assert xs == NUM_THREADS
        ext = str(nprobe) if len(str(nprobe)) == 2 else f"\ {nprobe}"
        plt.plot(NUM_THREADS, ys, label=f"XTR$_\\text{{base}}/\\text{{WARP}}$, n$_\\text{{probe}} = {ext}$",
                 color="tab:blue", marker="o")

def add_plaid_results(experiment_results):
    NUM_THREADS = sorted(list(set(experiment_results.view("num_threads"))))
    # `k` corresponds to `document_top_k`
    grouped_by_k = defaultdict(list)
    for entry in experiment_results:
        k = entry.view("document_top_k")
        num_threads = entry.view("num_threads")
        grouped_by_k[k].append((num_threads, metric_round(sum(entry.latency.plaid_style().values()), fact=1)))
    grouped_by_k = dict(grouped_by_k)

    for k, results in grouped_by_k.items():
        values = sorted(results, key=lambda x: x[0])
        xs, ys = [x[0] for x in values], [x[1] for x in values]
        assert xs == NUM_THREADS

        ext = str(k)
        plt.plot(NUM_THREADS, ys, label=f"ColBERT$_\\text{{v2}}$/PLAID, k$ = {ext}$", marker="o", color=PLAID_COLORS[k], linestyle="--")

def run_scale_threads_vs_plaid_analysis():
    plt.figure(figsize=(10, 8))

    experiment_results = load_experiment_results(Engine.COLBERTv2_PLAID, ["plaid_eval_dev_sets.json"])
    add_plaid_results(experiment_results.filter(dataset="lotte.pooled.dev"))

    experiment_results = load_experiment_results(Engine.XTR_WARP, ["warp_scale_threads.json"])
    add_warp_results(experiment_results, nprobe_values=[32])


    MINOR, MAJOR = 20, 24
    plt.xscale("log")
    plt.yticks([100, 200, 300, 400, 500], fontsize=MINOR)
    plt.ylabel("Latency (ms)", fontsize=MAJOR)
    plt.gca().xaxis.set_minor_locator(plt.NullLocator())
    
    plt.xlabel("#Threads", fontsize=MAJOR)
    plt.tick_params(axis="x", which="major", pad=10)
    plt.xticks([2**i for i in range(0, 4 + 1)], [f"$2^{i}$" for i in range(0, 4 + 1)], fontsize=MINOR)
    plt.grid()
    plt.legend(fontsize=MINOR)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("output/scale_threads_vs_plaid.pdf")
    plt.show()

if __name__ == "__main__":
    run_scale_threads_vs_plaid_analysis()