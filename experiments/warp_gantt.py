from analysis import Engine, load_experiment_results, prettify_dataset_name
from analysis.gantt_plot import group_latency_measurements, make_uniform_comparison_gantt_plot
from analysis.gantt_config import *

import matplotlib.pyplot as plt

def remap_xtr_latency(xtr_latency):
    xtr_latency = group_latency_measurements(xtr_latency, XTR_LATENCY_GROUPS)

    # Perform remapping "Token Retrieval -> Candidate Generation", and "dummy" stages
    return group_latency_measurements(xtr_latency, {
        "Query Encoding": ["Query Encoding"],
        "Candidate Generation": ["Token Retrieval"],
        "Filtering": [],
        "Decompression": [],
        "Scoring": ["Scoring"]
    })

def make_warp_comp_gantt(dataset, show_plaid=False,
                        document_top_ks=[10, 100, 1000], token_top_k=40_000, nprobe=32, nbits=4, bound=None):
    plt.figure(figsize=(14, 6))
    latencies, labels = [], []
    K_MAP = {10: "10", 100: "100", 1000: "1\,000", 40000: "40\,000"}
    
    xtr_non_opt_results = load_experiment_results(Engine.XTR_REFERENCE, ["xtr_eval_test_unopt.json"]).filter(
        dataset=dataset, token_top_k=token_top_k, num_threads=1
    )
    assert len(xtr_non_opt_results) == 1
    xtr_non_opt_latency = xtr_non_opt_results[0].latency.plaid_style(single_run=True)
    latencies.append(remap_xtr_latency(xtr_non_opt_latency))
    labels.append(f"XTR$_\\text{{base}}$/ScaNN\n(k'$={K_MAP[token_top_k]}$, opt=False)")
    
    xtr_opt_results = load_experiment_results(Engine.XTR_REFERENCE, ["xtr_eval_test_sets.json"]).filter(
        dataset=dataset, token_top_k=token_top_k, num_threads=1
    )
    assert len(xtr_opt_results) == 1
    xtr_opt_latency = xtr_opt_results[0].latency.plaid_style()

    latencies.append(remap_xtr_latency(xtr_opt_latency))
    labels.append(f"XTR$_\\text{{base}}$/ScaNN\n(k'$={K_MAP[token_top_k]}$, opt=True)")

    if show_plaid:
        for document_top_k in reversed(document_top_ks):
            plaid_results = load_experiment_results(Engine.COLBERTv2_PLAID, ["plaid_eval_test_sets.json"]).filter(
                dataset=dataset, document_top_k=document_top_k, num_threads=1
            )
            assert len(plaid_results) == 1
            plaid_latency = plaid_results[0].latency.plaid_style()

            latencies.append(group_latency_measurements(plaid_latency, PLAID_LATENCY_GROUPS))
            labels.append(f"ColBERT$_\\text{{v2}}$/PLAID\n(k$={K_MAP[document_top_k]}$)")

    warp_results = load_experiment_results(Engine.XTR_WARP, ["xtr_warp_eval_test_sets.json"]).filter(
        dataset=dataset, nprobe=nprobe, nbits=nbits, num_threads=1
    )
    assert len(warp_results) == 1
    warp_latency = warp_results[0].latency.plaid_style()

    # Add "dummy" filtering stage
    warp_latency = group_latency_measurements(warp_latency, WARP_LATENCY_GROUPS)
    warp_latency = group_latency_measurements(warp_latency, {
        "Query Encoding": ["Query Encoding"],
        "Candidate Generation": ["Candidate Generation"],
        "Filtering": [],
        "Decompression": ["Decompression"],
        "Scoring": ["Scoring"]
    })
    latencies.append(warp_latency)
    labels.append(f"XTR$_\\text{{base}}$/WARP\n(n$_\\text{{probe}}={nprobe}$)")

    ax = make_uniform_comparison_gantt_plot(latencies, labels=labels, colors=PLAID_COLOR_MAP, bound=bound, center_align=True, center_align_pad=160)

    yticks = ax.get_yticklabels()
    yticks[0].set_weight("bold")

    if show_plaid:
        row_to_highlight = 1
        ax = plt.gca()

        y_coord = ax.get_yticks()[row_to_highlight]
        bar_height = ax.get_ylim()[1] / len(ax.get_yticks())

        ax.add_patch(plt.Rectangle((0, y_coord - bar_height/2), ax.get_xlim()[1], len(document_top_ks) * bar_height, 
                                facecolor='lightgrey', alpha=0.5, zorder=0,
                                linestyle='--', edgecolor='grey', linewidth=1))

        ax.set_axisbelow(True)

def make_warp_gantt(datasets, nprobe=32, nbits=4, bound=None, show_nprobe=False, num_threads=1):
    if not isinstance(datasets, list):
        datasets = [datasets]
    
    latencies, labels = [], []
    experiment_results = load_experiment_results(Engine.XTR_WARP, ["xtr_warp_eval_test_sets.json"])
    for dataset in datasets:
        warp_results = experiment_results.filter(
            dataset=dataset, nprobe=nprobe, nbits=nbits, num_threads=num_threads
        )
        assert len(warp_results) == 1
        warp_latency = warp_results[0].latency.plaid_style()

        warp_latency = group_latency_measurements(warp_latency, WARP_LATENCY_GROUPS)
        
        latencies.append(warp_latency)
        if show_nprobe:
            nprobe_info = f"\n(n$_\\text{{probe}}={nprobe}$"
        else: nprobe_info = ""
        labels.append(f"XTR$_\\text{{base}}$/WARP{nprobe_info}\n{prettify_dataset_name(dataset)}")

    make_uniform_comparison_gantt_plot(latencies, labels=labels, colors=WARP_COLOR_MAP, bound=bound, center_align=True, center_align_pad=160, label_offset=3)

def make_warp_gantt16(datasets, nprobe=32, nbits=4, bound=None, show_nprobe=False, num_threads=1):
    if not isinstance(datasets, list):
        datasets = [datasets]
    
    latencies, labels = [], []
    experiment_results = load_experiment_results(Engine.XTR_WARP, ["xtr_warp_eval_test_sets_16.json"])
    for dataset in datasets:
        warp_results = experiment_results.filter(
            dataset=dataset, nprobe=nprobe, nbits=nbits, num_threads=num_threads
        )
        assert len(warp_results) == 1
        warp_latency = warp_results[0].latency.plaid_style()

        warp_latency = group_latency_measurements(warp_latency, {
            "Query Encoding": ["Query Encoding"],
            "Candidate Generation": ["Candidate Generation", "top-k Precompute"],
            "Decompression/Scoring$_\\text{(fused)}$": ["Decompression", "Build Matrix"],
        })

        FUSED_WARP_COLOR_MAP = {
            "Query Encoding": GanttColor.BLUE,
            "Candidate Generation": GanttColor.ORANGE,
            "Decompression/Scoring$_\\text{(fused)}$": ("#1A8CFF", "o"),
        }
        
        latencies.append(warp_latency)
        if show_nprobe:
            nprobe_info = f"\n(n$_\\text{{probe}}={nprobe}$"
        else: nprobe_info = ""
        labels.append(f"XTR$_\\text{{base}}$/WARP{nprobe_info}\n{prettify_dataset_name(dataset)}")

    make_uniform_comparison_gantt_plot(latencies, labels=labels, colors=FUSED_WARP_COLOR_MAP, bound=bound, center_align=True, center_align_pad=160, label_offset=3)


def run_warp_gantt_analys():
    make_warp_comp_gantt(dataset="lotte.pooled.test", document_top_ks=[1000], show_plaid=True, bound=8200)
    plt.savefig("output/latency/warp/warp_comp_lotte.pooled.test.pdf")
    make_warp_comp_gantt(dataset="lotte.lifestyle.test", document_top_ks=[1000], show_plaid=True, bound=2450)
    plt.savefig("output/latency/warp/warp_comp_lotte.lifestyle.test.pdf")
    make_warp_comp_gantt(dataset="beir.nfcorpus.test", document_top_ks=[1000], show_plaid=True, bound=1100)
    plt.savefig("output/latency/warp/warp_comp_beir.nfcorpus.test.pdf")

    make_warp_gantt(datasets=["beir.nfcorpus.test", "lotte.lifestyle.test", "lotte.pooled.test"], bound=200)
    plt.savefig("output/latency/warp/warp_test_sets.pdf")

    make_warp_gantt16(datasets=["beir.nfcorpus.test", "lotte.lifestyle.test", "lotte.pooled.test"], bound=200, num_threads=16)
    plt.savefig("output/latency/warp/warp_test_sets16.pdf")

if __name__ == "__main__":
    run_warp_gantt_analys()