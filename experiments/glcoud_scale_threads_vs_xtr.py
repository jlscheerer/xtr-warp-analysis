from analysis import Engine, load_experiment_results

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from collections import defaultdict

def get_gcloud_scann_results(dataset):
    scann_results = load_experiment_results(Engine.XTR_REFERENCE, ["xtr_eval_dev_sets_gcloud_scann.json"]).filter(
        dataset=dataset
    )

    NUM_THREADS = sorted(list(set(scann_results.view("num_threads"))))
    assert NUM_THREADS == [1, 2, 4, 8]

    grouped_by_ttk = defaultdict(list)
    for entry in scann_results:
        token_top_k = entry.view("token_top_k")
        num_threads = entry.view("num_threads")
        latency = sum(entry.latency.plaid_style().values())
        grouped_by_ttk[token_top_k].append((num_threads, latency))
    return dict(grouped_by_ttk)

def get_cloud_faiss_results(dataset):
    faiss_results = load_experiment_results(Engine.XTR_REFERENCE, ["xtr_eval_dev_sets_gcloud_faiss.json"]).filter(
        dataset=dataset
    )

    NUM_THREADS = sorted(list(set(faiss_results.view("num_threads"))))
    assert NUM_THREADS == [1, 2, 4, 8]

    grouped_by_ttk = defaultdict(list)
    for entry in faiss_results:
        token_top_k = entry.view("token_top_k")
        num_threads = entry.view("num_threads")
        latency = sum(entry.latency.plaid_style(single_run=True).values())
        grouped_by_ttk[token_top_k].append((num_threads, latency))
    return dict(grouped_by_ttk)

def get_cloud_warp_results(dataset, nprobe=32):
    warp_results = load_experiment_results(Engine.XTR_WARP, ["warp_eval_dev_sets_gcloud.json"]).filter(
        dataset=dataset, nprobe=nprobe
    )

    results = []
    for entry in warp_results:
        num_threads = entry.view("num_threads")
        latency = sum(entry.latency.plaid_style().values())
        results.append((num_threads, latency))
    return results

def gcloud_comparison(dataset, show_faiss=True):
    plt.figure(figsize=(10, 8))

    scann_results = get_gcloud_scann_results(dataset=dataset)
    faiss_results = get_cloud_faiss_results(dataset=dataset)
    merged_results = {"scann": scann_results, "faiss": faiss_results}

    NUM_THREADS = [1, 2, 4, 8]

    COLORS = ["orange", "red", "purple", "pink"]
    PRETTY_INDEX_NAME = {
        "scann": "ScaNN",
        "faiss": "FAISS"
    }
    K_MAP = {1000: "1\,000", 40000: "40\,000"}
    i = 0
    for index, index_results in merged_results.items():
        if index == "faiss" and not show_faiss:
            continue
        for token_top_k in [1_000, 40_000]:
            results = index_results[token_top_k]
            values = sorted(results, key=lambda x: x[0])
            xs, ys = [x[0] for x in values], [x[1] for x in values]
            assert xs == NUM_THREADS
            plt.plot(NUM_THREADS, ys, label=f"XTR$_\\text{{base}}$/{PRETTY_INDEX_NAME[index]}, k'$={K_MAP[token_top_k]}$", marker="o", color=COLORS[i], linestyle="--")
            i += 1 

            print(index, token_top_k, ys)

    # Add WARP Results
    NPROBE = 32
    warp_results = get_cloud_warp_results(dataset=dataset, nprobe=NPROBE)
    print(warp_results)
    values = sorted(warp_results, key=lambda x: x[0])
    xs, ys = [x[0] for x in values], [x[1] for x in values]
    assert xs == NUM_THREADS
    plt.plot(NUM_THREADS, ys, label=f"XTR$_\\text{{base}}$/WARP, n$_\\text{{nprobe}} = {NPROBE}$", marker="o", color="tab:blue") 

    print(ys)

    MINOR, MAJOR = 20, 24
    plt.xscale("log")

    if show_faiss:
        ax = plt.gca()
        ax.yaxis.offsetText.set_fontsize(MINOR)
        plt.yticks([5_000, 10_000, 15_000, 20_000], fontsize=MINOR)
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    else:
        plt.yticks([100, 200, 300, 400, 500, 600], fontsize=MINOR)

    # plt.yticks([100, 200, 300, 400, 500], fontsize=MINOR)
    plt.ylabel("Latency (ms)", fontsize=MAJOR)
    plt.gca().xaxis.set_minor_locator(plt.NullLocator())
    
    plt.xlabel("#Threads", fontsize=MAJOR)
    plt.tick_params(axis="x", which="major", pad=10)
    plt.xticks([2**i for i in range(0, 3 + 1)], [f"$2^{i}$" for i in range(0, 3 + 1)], fontsize=MINOR)
    plt.grid()
    plt.legend(fontsize=MINOR)
    plt.subplots_adjust(bottom=0.15)
    plt.legend(fontsize=MINOR)


def run_gcloud_scale_threads_vs_xtr_analysis():
    dataset = "lotte.science.dev"

    gcloud_comparison(dataset=dataset, show_faiss=False)
    plt.savefig("output/scale_gcloud_vs_xtr.pdf")
    plt.show()

    gcloud_comparison(dataset=dataset, show_faiss=True)
    plt.savefig("output/scale_gcloud_vs_xtr_faiss.pdf")
    plt.show()


if __name__ == "__main__":
    run_gcloud_scale_threads_vs_xtr_analysis()