from analysis import Engine, load_experiment_results, metric_round, PRETTY_NAMES, LOTTE_DATASETS, BEIR_DATASETS, prettify_dataset_name, _resolve_engine_directory
from analysis.table import TableRow, TableEntry, TableGroup, Table, Template, Footnote, format_model_name, compute_row_avg

import os
import json

from collections import defaultdict

def make_colbert_v2_table(collection, metric, nprobe_values=[32]):
    datasets = {"lotte": LOTTE_DATASETS, "beir": BEIR_DATASETS}[collection]["test"]

    plaid_results = load_experiment_results(Engine.COLBERTv2_PLAID, ["plaid_eval_test_sets.json"]).filter(
        collection=collection, num_threads=1
    )

    root = _resolve_engine_directory(Engine.XTR_WARP)
    with open(os.path.join(root, "cb-results.json"), "r") as file:
        warp_results = json.loads(file.read())

    plaid_k = defaultdict(dict)
    for entry in plaid_results:
        dataset = entry.view("dataset")
        k = entry.view("document_top_k")
        plaid_k[k][dataset] = entry.view(metric)
    plaid_k = dict(plaid_k)

    warp_nproobe = defaultdict(dict)
    for entry in warp_results:
        c, dataset, _ = entry["provenance"]["dataset_id"].split(".")
        if c != collection:
            continue
        nprobe = entry["provenance"]["nprobe"]
        if metric in entry["metrics"]:
            score = entry["metrics"][metric]
        else:
            m1, k = metric.split("@")
            m2 = f"{m1}_cut@{k}"
            score = entry["metrics"][m2]
        warp_nproobe[nprobe][dataset] = score

    groups = []

    K_VALUES = [10, 100, 1000]
    rows = []
    for k in K_VALUES:
        row = [f"ColBERT$_\\text{{v2}}$/PLAID (k$={k}$)"]
        for dataset in datasets:
            row.append(TableEntry(text=f"{metric_round(plaid_k[k][dataset])}"))
        row.append(compute_row_avg(row))
        rows.append(TableRow(data=row))
    groups.append(TableGroup(rows=rows, gray_text=True))

    rows = []
    for nprobe in nprobe_values:
        row = [f"ColBERT$_\\text{{v2}}$/WARP (n$_\\text{{probe}}={nprobe}$)"]
        for dataset in datasets:
            row.append(TableEntry(text=f"{metric_round(warp_nproobe[nprobe][dataset])}"))
        row.append(compute_row_avg(row))
        rows.append(TableRow(data=row, gray_background=True))
    groups.append(TableGroup(rows=rows))

    return Table(
        columns=[PRETTY_NAMES[dataset] for dataset in datasets],
        groups=groups,
    ).underline_max()

def run_colbert_v2_warp_analysis():
    table = make_colbert_v2_table(collection="beir", metric="ndcg@10", nprobe_values=[32])
    table.save("colbertv2_warp.tex")

if __name__ == "__main__":
    run_colbert_v2_warp_analysis()