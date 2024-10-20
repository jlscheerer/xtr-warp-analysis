from analysis import Engine, load_experiment_results, DATASET_STATS

import re
import numpy as np
import matplotlib.pyplot as plt

def format_nprobe(text):
    match = re.match("nprobe=([\\d]+)", text)
    assert match is not None
    value = match[1]
    return f"n$_\\text{{probe}}$ = {value}"

T_PRIME_TICKS = [0, 20_000, 40_000, 60_000, 80_000, 100_000]
NR_TICKS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

NR_TICKS2 = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]


def plot_nprobe_tprime(experiment_results, dataset, metric, nprobes):
    plt.figure(figsize=(10, 8))
    experiment_results = experiment_results.filter(dataset=dataset, nbits=4, nprobe=nprobes)

    metric_max = experiment_results.view(metric).max()

    experiment_results.plot("t_prime", lambda x: x.view(metric) / metric_max, group_by="nprobe")
    
    MINOR, MAJOR = 20, 24
    plt.ylabel(f"Normalized {metric.title()}", fontsize=MAJOR)
    plt.xlabel("$t'$", fontsize=MAJOR)
    
    legend = plt.legend(fontsize=MINOR)
    for text in legend.get_texts():
        text.set_text(format_nprobe(text.get_text()))

    plt.xticks(T_PRIME_TICKS, fontsize=MINOR)
    plt.yticks(NR_TICKS, fontsize=MINOR)
    plt.grid()

    plt.savefig(f"output/hyperparameters/nprobe_{dataset}.pdf")

def format_b(text):
    match = re.match(".*\\=(.*)", text)[1]
    match = re.match("\\((\\d+), (\\d+)\\)", match)
    nprobe, nbits = match[1], match[2]
    return f"n$_\\text{{probe}}$ = {nprobe}, $b = {nbits}$"

def plot_b_t_prime(experiment_results, dataset, metric, nprobes):
    plt.figure(figsize=(10, 8))
    experiment_results = experiment_results.filter(dataset=dataset, nbits=[2,4], nprobe=nprobes)
    metric_max = experiment_results.view(metric).max()

    experiment_results.plot("t_prime", lambda x: x.view(metric) / metric_max, group_by=lambda x: (x.view("nprobe"), x.view("nbits")))


    MINOR, MAJOR = 20, 24
    plt.ylabel(f"Normalized {metric.title()}", fontsize=MAJOR)
    plt.xlabel("$t'$", fontsize=MAJOR)
    
    legend = plt.legend(fontsize=MINOR)
    for text in legend.get_texts():
        text.set_text(format_b(text.get_text()))

    plt.xticks(T_PRIME_TICKS, fontsize=MINOR)
    plt.yticks(NR_TICKS2, fontsize=MINOR)
    plt.grid()

    plt.savefig(f"output/hyperparameters/b_{dataset}_{metric}.pdf")

def run_hyperparameter_analysis():
    experiment_results = load_experiment_results(Engine.XTR_WARP, ["warp_eval_dev_sets.json"])
    NPROBE_VALUES = [1, 2, 4, 8, 16, 32, 64]
    NPROBE_VALUES_SM = [32]
    DATASETS = ["beir.nfcorpus.dev", "beir.quora.dev", "lotte.science.dev", "lotte.pooled.dev"]

    for dataset in DATASETS:
        plot_nprobe_tprime(experiment_results, dataset, metric="recall@100", nprobes=NPROBE_VALUES)
        
    DATASETS = ["lotte.science.dev", "lotte.pooled.dev"]
    for dataset in DATASETS:    
        for k in [10, 100]:
            plot_b_t_prime(experiment_results, dataset, metric=f"recall@{k}", nprobes=NPROBE_VALUES_SM)

if __name__ == "__main__":
    run_hyperparameter_analysis()