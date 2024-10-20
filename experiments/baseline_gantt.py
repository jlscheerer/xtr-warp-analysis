from analysis import Engine, load_experiment_results, metric_round
from analysis.gantt_plot import group_latency_measurements, make_uniform_comparison_gantt_plot
from analysis.gantt_config import *
from analysis.comparison_bar import group_comparison_bar

import matplotlib.pyplot as plt

def make_plaid_gantt(dataset, k_values, bound=None):
    plaid_results = load_experiment_results(Engine.COLBERTv2_PLAID, ["plaid_eval_dev_sets.json"])
    latencies, labels = [], []
    K_MAP = {10: "10", 100: "100", 1000: "1\,000"}
    for k in k_values:
        plaid_at_k = plaid_results.filter(dataset=dataset, num_threads=1, document_top_k=k)
        assert len(plaid_at_k) == 1
        latency_at_k = plaid_at_k[0].latency.plaid_style()
        labels.append(f"$\\text{{k}} = {K_MAP[k]}$")
        latencies.append(group_latency_measurements(latency_at_k, PLAID_LATENCY_GROUPS))

    make_uniform_comparison_gantt_plot(latencies, labels=labels, colors=PLAID_COLOR_MAP, bound=bound)

def make_xtr_gantt(dataset, token_top_k_values, bound=None):
    xtr_results = load_experiment_results(Engine.XTR_REFERENCE, ["xtr_eval_dev_sets.json"])
    latencies, labels = [], []
    K_MAP = {1000: "1\,000", 40000: "40\,000"}
    for token_top_k in token_top_k_values:
        xtr_at_k_prime = xtr_results.filter(dataset=dataset, num_threads=1, token_top_k=token_top_k)
        assert len(xtr_at_k_prime) == 1
        latency_at_k_prime = xtr_at_k_prime[0].latency.plaid_style()
        labels.append(f"$\\text{{k}}' = {K_MAP[token_top_k]}$")
        latencies.append(group_latency_measurements(latency_at_k_prime, XTR_LATENCY_GROUPS))

        print(dataset, token_top_k, group_latency_measurements(latency_at_k_prime, XTR_LATENCY_GROUPS))
    make_uniform_comparison_gantt_plot(latencies, labels=labels, colors=XTR_COLOR_MAP, bound=bound)

def make_xtr_unopt_gantt(dataset, token_top_k_values, bound):
    xtr_results = load_experiment_results(Engine.XTR_REFERENCE, ["xtr_eval_dev_sets_unopt.json"])
    latencies, labels = [], []
    K_MAP = {1000: "1\,000", 40000: "40\,000"}
    for token_top_k in token_top_k_values:
        xtr_at_k_prime = xtr_results.filter(dataset=dataset, num_threads=1, token_top_k=token_top_k)
        assert len(xtr_at_k_prime) == 1
        latency_at_k_prime = xtr_at_k_prime[0].latency.plaid_style(single_run=True)
        labels.append(f"$\\text{{k}}' = {K_MAP[token_top_k]}$")
        latencies.append(group_latency_measurements(latency_at_k_prime, XTR_LATENCY_GROUPS))
    make_uniform_comparison_gantt_plot(latencies, labels=labels, colors=XTR_COLOR_MAP, bound=bound)

def xtr_breakdown_latency(dataset, token_top_k, stage, unopt=True, return_others=False):
    xtr_results = load_experiment_results(Engine.XTR_REFERENCE, [f"xtr_eval_dev_sets{'_unopt' if unopt else ''}.json"]).filter(
        dataset=dataset, num_threads=1, token_top_k=token_top_k
    )
    assert len(xtr_results) == 1
    xtr_latencies = xtr_results[0].latency.plaid_style(single_run=unopt)
    stage_breakdown = {
        key: xtr_latencies[key] for key in XTR_LATENCY_GROUPS[stage]
    }
    if not return_others:
        return stage_breakdown
    return stage_breakdown, sum([
        value for key, value in xtr_latencies.items() if key not in XTR_LATENCY_GROUPS[stage]
    ])
    

XTR_BREAKDOWN_GROUPS = {
    # "Estimate Missing Similarity": ["Estimate Missing Similarity"], << 1ms
    "get_did2scores": ["get_did2scores"],
    "add_ems": ["add_ems"],
    "get_final_score": ["get_final_score"],
    "sort_scores": ["sort_scores"],
}

XTR_BREAKDOWN_COLOR_MAP = {
    # "Estimate Missing Similarity": GanttColor.BLUE,
    "get_did2scores": ("#B3D9FF", "/"),
    "add_ems": ("#80BFFF", "||"), # C71585
    "get_final_score": ("#4DA6FF", "x"), # 7B68EE
    "sort_scores": ("#1A8CFF", "o"),
    "": ("#808080", None)
}

XTR_BREAKDOWN_LABELS = ["opt=False", "opt=True"]

def make_xtr_opt_breadown_comparison(dataset, token_top_k, stage, bound=None):
    xtr_unopt_stage_latencies = xtr_breakdown_latency(dataset, token_top_k, stage, unopt=True)
    xtr_opt_stage_latencies = xtr_breakdown_latency(dataset, token_top_k, stage, unopt=False)

    return group_comparison_bar([
        {key: xtr_unopt_stage_latencies[key] for key in XTR_BREAKDOWN_GROUPS.keys()},
        {key: xtr_opt_stage_latencies[key] for key in XTR_BREAKDOWN_GROUPS.keys()}
    ], labels=XTR_BREAKDOWN_LABELS, colors=[("#1E88E5", "///"), ("#FFC107", "o")], show_improvement=True, bound=bound)

def make_xtr_unopt_breakdown_gantt(dataset, token_top_k, stage, bound=None):
    xtr_unopt_stage_latencies, unopt_others = xtr_breakdown_latency(dataset, token_top_k, stage, unopt=True, return_others=True)
    xtr_opt_stage_latencies, opt_others  = xtr_breakdown_latency(dataset, token_top_k, stage, unopt=False, return_others=True)

    unopt_others += xtr_unopt_stage_latencies["Estimate Missing Similarity"]
    opt_others += xtr_opt_stage_latencies["Estimate Missing Similarity"]

    unopt_total = sum(xtr_unopt_stage_latencies.values())
    opt_total = sum(xtr_opt_stage_latencies.values())

    unopt_grouped = {"": unopt_others}
    unopt_grouped.update(group_latency_measurements(xtr_unopt_stage_latencies, XTR_BREAKDOWN_GROUPS, ignored=["Estimate Missing Similarity"]))

    opt_grouped = {"": opt_others}
    opt_grouped.update(group_latency_measurements(xtr_opt_stage_latencies, XTR_BREAKDOWN_GROUPS, ignored=["Estimate Missing Similarity"]))

    latencies = [unopt_grouped, opt_grouped]
    ax = make_uniform_comparison_gantt_plot(
        latencies,
        labels=XTR_BREAKDOWN_LABELS,
        colors=XTR_BREAKDOWN_COLOR_MAP,
        bound=bound,
        barlabel=[f"{int(metric_round(unopt_total, fact=1, decimals=0))}ms", f"{int(metric_round(opt_total, fact=1))}ms ({metric_round(unopt_total / opt_total, fact=1)}x)"],
    )


def run_baseline_gantt_analysis():
    DATASETS = ["beir.nfcorpus.dev", "lotte.lifestyle.dev", "lotte.pooled.dev"]
    for dataset in DATASETS:
        make_plaid_gantt(dataset=dataset, k_values=LOTTE_K_VALUES, bound=1950)
        plt.savefig(f"output/latency/plaid_baseline_{dataset}.pdf")
        make_xtr_gantt(dataset=dataset, token_top_k_values=XTR_TOKEN_TOP_K_VALUES, bound=1950)
        plt.savefig(f"output/latency/xtr_baseline_{dataset}.pdf")

    for dataset in DATASETS:
        make_xtr_unopt_gantt(dataset=dataset, token_top_k_values=[1_000, 40_000], bound=6450)
        plt.savefig(f"output/latency/xtr_unopt_{dataset}.pdf")

    make_xtr_unopt_breakdown_gantt(dataset="lotte.pooled.dev", token_top_k=40_000, stage="Scoring", bound=6450)
    plt.savefig("output/latency/xtr_opt_improvements_lotte.pooled.dev.pdf")

    make_xtr_opt_breadown_comparison(dataset="lotte.pooled.dev", token_top_k=40_000, stage="Scoring", bound=2850)
    plt.savefig("output/latency/xtr_opt_comparison_lotte.pooled.dev.pdf")
    make_xtr_opt_breadown_comparison(dataset="lotte.lifestyle.dev", token_top_k=40_000, stage="Scoring", bound=2200)
    plt.savefig("output/latency/xtr_opt_comparison_lotte.lifestyle.dev.pdf")

if __name__ == "__main__":
    run_baseline_gantt_analysis()