from analysis import Engine, load_experiment_results, metric_round, PRETTY_NAMES, LOTTE_DATASETS, BEIR_DATASETS
from analysis.table import TableRow, TableEntry, TableGroup, Table, Template, Footnote, format_model_name, compute_row_avg

from collections import defaultdict

from tabulate import tabulate

TOKEN_TOP_K_VALUES = [1_000, 40_000]
NPROBE_VALUES = [8, 12, 16, 24, 32]

def eval_xtr(experiment_results, collection="lotte", metric="success@5"):
    grouped_by_ttk = defaultdict(dict)
    for entry in experiment_results.filter(collection=collection):
        dataset_id = entry.view("dataset_id")
        ttk = entry.view("token_top_k")
        results = entry.view(metric)
        latency = sum(entry.latency.plaid_style().values())
        grouped_by_ttk[ttk][dataset_id] = {
            metric: metric_round(results, fact=100),
            "latency": metric_round(latency, fact=1)
        }
    grouped_by_ttk = dict(grouped_by_ttk)
    return grouped_by_ttk

def display_latency_metric(result, metric):
    return f"{result[metric]:.01f} ({result['latency']:.01f})"

def print_xtr_metric_table(results, collection, datasets, metric):
    avg_metric = {token_top_k: sum([results[token_top_k][f"{collection}.{dataset}.test"][metric] for dataset in datasets]) / len(datasets)
                 for token_top_k in TOKEN_TOP_K_VALUES}
    data = [
        ([token_top_k] + [display_latency_metric(results[token_top_k][f"{collection}.{dataset}.test"], metric)
                          for dataset in datasets] + [avg_metric[token_top_k]]) for token_top_k in TOKEN_TOP_K_VALUES
    ]
    print(tabulate(data, headers=["token-top-k"] + datasets + ["Avg."], floatfmt=".01f"))

def eval_warp(experiment_results, collection, metric):
    grouped_by_nprobe = defaultdict(dict)
    for entry in experiment_results.filter(collection=collection, nprobe=NPROBE_VALUES):
        dataset_id = entry.view("dataset_id")
        nprobe = entry.view("nprobe")
        results = entry.view(metric)
        latency = sum(entry.latency.plaid_style().values())
        grouped_by_nprobe[nprobe][dataset_id] = {
            metric: metric_round(results, fact=100),
            "latency": metric_round(latency, fact=1)
        }
    grouped_by_nprobe = dict(grouped_by_nprobe)
    return grouped_by_nprobe

def print_warp_metric_table(results, collection, datasets, metric):
    print(f"XTR/WARP: {collection}, {metric}")
    avg_metric = {nprobe: sum([results[nprobe][f"{collection}.{dataset}.test"][metric] for dataset in datasets]) / len(datasets)
                    for nprobe in NPROBE_VALUES}
    data = [
        ([nprobe] + [display_latency_metric(results[nprobe][f"{collection}.{dataset}.test"], metric)
                          for dataset in datasets] + [avg_metric[nprobe]]) for nprobe in NPROBE_VALUES
    ]
    print(tabulate(data, headers=["n_probe"] + datasets + ["Avg."], floatfmt=".01f"))

def beir_recall100_table(xtr_results, warp_results, show_ref=False, show_others=True, k_prime_display=[], nprobe_display=[]):
    template = Template(filename="recall100_beir")
    datasets = BEIR_DATASETS["test"]
    xtr_eval = eval_xtr(xtr_results, collection="beir", metric="recall@100")
    warp_eval = eval_warp(warp_results, collection="beir", metric="recall@100")

    groups, hidden = [], False
    for tgroup in template.groups:
        if len(tgroup) == 0:
            group = []
            if show_ref:
                model = "XTR_base"
                row = [format_model_name(model)]
                for dataset in datasets:
                    score = template.get_score(model, dataset)
                    row.append(TableEntry(str(score)))
                row.append(compute_row_avg(row))
                group.append(TableRow(data=row, gray_text=True, gray_background=True))

            for token_top_k, scores in xtr_eval.items():
                if token_top_k not in k_prime_display:
                    continue

                row = [format_model_name(f"XTR_base/ScaNN {{\\footnotesize($k' = \\num{{{token_top_k}}}$)}}")]
                for dataset in datasets:
                    ref = scores[f"beir.{dataset}.test"]
                    score, latency = ref["recall@100"], ref["latency"]
                    row.append(TableEntry(f"{score} {{\\footnotesize({latency})}}"))
                row.append(compute_row_avg(row))
                group.append(TableRow(data=row, gray_background=True))

            for nprobe, scores in warp_eval.items():
                if nprobe not in nprobe_display:
                    continue
                row = [format_model_name(f"XTR_base/WARP {{\\footnotesize($n_\\text{{nprobe}} = {nprobe}$)}}")]
                for dataset in datasets:
                    ref = scores[f"beir.{dataset}.test"]
                    score, latency = ref["recall@100"], ref["latency"]
                    row.append(TableEntry(f"{score} {{\\footnotesize({latency})}}"))
                row.append(compute_row_avg(row))
                group.append(TableRow(data=row, gray_background=True))
            groups.append(TableGroup(group, gray_text=False).bold_max())
            hidden = True
        else:
            if not show_others:
                continue
            group = []
            for model in tgroup:
                row = [format_model_name(model)]
                for dataset in datasets:
                    score = template.get_score(model, dataset)
                    row.append(TableEntry(str(score)))
                row.append(compute_row_avg(row))
                group.append(TableRow(data=row, gray_background=False))
            groups.append(TableGroup(group, gray_text=hidden))

    return Table(
        columns=[PRETTY_NAMES[dataset] for dataset in datasets],
        groups=groups,
        caption=template.caption,
        label="table:beir_recall100"
    ).underline_max()


def beir_ndcg10_table(xtr_results, warp_results, show_ref=False, show_others=True, k_prime_display=[], nprobe_display=[]):
    template = Template(filename="ndcg10_beir")
    datasets = BEIR_DATASETS["test"]
    metric = "ndcg@10"
    xtr_eval = eval_xtr(xtr_results, collection="beir", metric=metric)
    warp_eval = eval_warp(warp_results, collection="beir", metric=metric)

    groups, hidden = [], False
    for tgroup in template.groups:
        if len(tgroup) == 0:
            group = []
            if show_ref:
                model = "XTR_base"
                row = [format_model_name(model)]
                for dataset in datasets:
                    score = template.get_score(model, dataset)
                    row.append(TableEntry(str(score)))
                row.append(compute_row_avg(row))
                group.append(TableRow(data=row, gray_text=True, gray_background=True))

            for token_top_k, scores in xtr_eval.items():
                if token_top_k not in k_prime_display:
                    continue

                row = [format_model_name(f"XTR_base/ScaNN {{\\footnotesize($k' = \\num{{{token_top_k}}}$)}}")]
                for dataset in datasets:
                    ref = scores[f"beir.{dataset}.test"]
                    score, latency = ref[metric], ref["latency"]
                    row.append(TableEntry(f"{score} {{\\footnotesize({latency})}}"))
                row.append(compute_row_avg(row))
                group.append(TableRow(data=row, gray_background=True))

            for nprobe, scores in warp_eval.items():
                if nprobe not in nprobe_display:
                    continue
                row = [format_model_name(f"XTR_base/WARP {{\\footnotesize($n_\\text{{nprobe}} = {nprobe}$)}}")]
                for dataset in datasets:
                    ref = scores[f"beir.{dataset}.test"]
                    score, latency = ref[metric], ref["latency"]
                    row.append(TableEntry(f"{score} {{\\footnotesize({latency})}}"))
                row.append(compute_row_avg(row))
                group.append(TableRow(data=row, gray_background=True))
            groups.append(TableGroup(group, gray_text=False).bold_max())
            hidden = True
        else:
            if not show_others:
                continue
            group = []
            for model in tgroup:
                model_notes = template.footnotes(model)
                if len(model_notes) != 0:
                    model_notes = "".join([f"\\{sym}" for sym in model_notes])
                    model_notes = f" $^{{{model_notes}}}$"
                else: model_notes = ""
                row = [format_model_name(f"{model}{model_notes}")]
                for dataset in datasets:
                    score = template.get_score(model, dataset)
                    row.append(TableEntry(str(score)))
                row.append(compute_row_avg(row))
                group.append(TableRow(data=row, gray_background=False))
            groups.append(TableGroup(group, gray_text=hidden))

    return Table(
        columns=[PRETTY_NAMES[dataset] for dataset in datasets],
        groups=groups,
        caption=template.caption,
        label="table:beir_ncdg10",
        footer=[Footnote(symbol=footnote["symbol"], note=text) for text, footnote in template.notes.items()]
    ).underline_max()

def lotte_success5_table(xtr_results, warp_results, show_ref=False, show_others=True, k_prime_display=[], nprobe_display=[]):
    template = Template(filename="success5_lotte")
    datasets = LOTTE_DATASETS["test"]
    metric = "success@5"
    xtr_eval = eval_xtr(xtr_results, collection="lotte", metric=metric)
    warp_eval = eval_warp(warp_results, collection="lotte", metric=metric)

    groups, hidden = [], False
    for tgroup in template.groups:
        if len(tgroup) == 0:
            group = []
            if show_ref:
                model = "XTR_base"
                row = [format_model_name(model)]
                for dataset in datasets:
                    score = template.get_score(model, dataset)
                    row.append(TableEntry(str(score)))
                row.append(compute_row_avg(row))
                group.append(TableRow(data=row, gray_text=True, gray_background=True))

            for token_top_k, scores in xtr_eval.items():
                if token_top_k not in k_prime_display:
                    continue

                row = [format_model_name(f"XTR_base/ScaNN {{\\footnotesize($k' = \\num{{{token_top_k}}}$)}}")]
                for dataset in datasets:
                    ref = scores[f"lotte.{dataset}.test"]
                    score, latency = ref[metric], ref["latency"]
                    row.append(TableEntry(f"{score} {{\\footnotesize({latency})}}"))
                row.append(compute_row_avg(row))
                group.append(TableRow(data=row, gray_background=True))

            for nprobe, scores in warp_eval.items():
                if nprobe not in nprobe_display:
                    continue
                row = [format_model_name(f"XTR_base/WARP {{\\footnotesize($n_\\text{{nprobe}} = {nprobe}$)}}")]
                for dataset in datasets:
                    ref = scores[f"lotte.{dataset}.test"]
                    score, latency = ref[metric], ref["latency"]
                    row.append(TableEntry(f"{score} {{\\footnotesize({latency})}}"))
                row.append(compute_row_avg(row))
                group.append(TableRow(data=row, gray_background=True))
            groups.append(TableGroup(group, gray_text=False).bold_max())
            hidden = True
        else:
            if not show_others:
                continue
            group = []
            for model in tgroup:
                model_notes = template.footnotes(model)
                if len(model_notes) != 0:
                    model_notes = "".join([f"\\{sym}" for sym in model_notes])
                    model_notes = f" $^{{{model_notes}}}$"
                else: model_notes = ""
                row = [format_model_name(f"{model}{model_notes}")]
                for dataset in datasets:
                    score = template.get_score(model, dataset)
                    row.append(TableEntry(str(score)))
                row.append(compute_row_avg(row))
                group.append(TableRow(data=row, gray_background=False))
            groups.append(TableGroup(group, gray_text=hidden))

    return Table(
        columns=[PRETTY_NAMES[dataset] for dataset in datasets],
        groups=groups,
        caption=template.caption,
        label="table:lotte_success5",
        footer=[Footnote(symbol=footnote["symbol"], note=text) for text, footnote in template.notes.items()]
    ).underline_max()

def run_evaluation_lotte_analysis():
    xtr_results = load_experiment_results(Engine.XTR_REFERENCE, ["xtr_eval_test_sets.json"])
    assert set(xtr_results.view("index_type")) == {"scann"}
    assert set(xtr_results.view("token_top_k")) == set(TOKEN_TOP_K_VALUES)
    assert set(xtr_results.view("num_threads")) == {1}

    warp_results = load_experiment_results(Engine.XTR_WARP, ["xtr_warp_eval_test_sets.json"])
    assert set(xtr_results.view("num_threads")) == {1}

    table = beir_ndcg10_table(xtr_results, warp_results, show_ref=False, show_others=True, k_prime_display=[40_000], nprobe_display=[32])
    table.save("beir_ndcg10.tex")

    table = beir_recall100_table(xtr_results, warp_results, show_ref=False, show_others=True, k_prime_display=[40_000], nprobe_display=[32])
    table.save("beir_recall100.tex")

    table = lotte_success5_table(xtr_results, warp_results, show_ref=False, show_others=True, k_prime_display=[40_000], nprobe_display=[32])
    table.save("lotte_success5.tex")


if __name__ == "__main__":
    run_evaluation_lotte_analysis()