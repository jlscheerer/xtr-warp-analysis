from analysis import Engine, load_experiment_results, metric_round, PRETTY_NAMES, LOTTE_DATASETS, BEIR_DATASETS, prettify_dataset_name
from analysis.table import TableRow, TableEntry, TableGroup, Table, Template, Footnote, format_model_name, compute_row_avg

def make_scann_vs_faiss_table(collection, metric, caption, label):
    scann_results = load_experiment_results(Engine.XTR_REFERENCE, ["xtr_eval_dev_sets.json"])
    faiss_results = load_experiment_results(Engine.XTR_REFERENCE, ["xtr_eval_dev_faiss.json"])

    datasets = {"lotte": LOTTE_DATASETS, "beir": BEIR_DATASETS}[collection]["dev"]

    K_MAP = {1000: "1\,000", 40000: "40\,000"}

    scann_ttk = dict()
    for token_top_k in [1_000, 40_000]:
        row = [f"XTR$_\\text{{base}}$/ScaNN (k'$={K_MAP[token_top_k]}$)"]
        for dataset in datasets:
            entry = scann_results.filter(token_top_k=token_top_k, collection=collection, dataset=dataset, num_threads=1)
            assert len(entry) == 1
            entry = entry[0]

            dataset = entry.view("dataset")
            score = entry.view(metric)
            latency = metric_round(sum(entry.latency.plaid_style().values()), fact=1)
            row.append(TableEntry(text=f"{metric_round(score)} {{\\footnotesize({latency})}}"))
        row.append(compute_row_avg(row))
        scann_ttk[token_top_k] = TableRow(data=row, gray_background=True)

    faiss_ttk = dict()
    for token_top_k in [1_000, 40_000]:
        row = [f"XTR$_\\text{{base}}$/FAISS (k'$={K_MAP[token_top_k]}$)"]
        for dataset in datasets:
            entry = faiss_results.filter(token_top_k=token_top_k, collection=collection, dataset=dataset)
            # NOTE "Hack" because of T/O operations
            assert len(entry) == 1
            entry = entry[0]
            num_threads =  entry.view("num_threads")

            dataset = entry.view("dataset")
            score = entry.view(metric)
            latency = metric_round(sum(entry.latency.plaid_style(single_run=1).values()), fact=1)
            if num_threads != 1:
                latency = "T/O"
            row.append(TableEntry(text=f"{metric_round(score)} {{\\footnotesize({latency})}}"))
        row.append(compute_row_avg(row))
        faiss_ttk[token_top_k] = TableRow(data=row)

    return Table(
        columns=[PRETTY_NAMES[dataset] for dataset in datasets],
        groups=[TableGroup(rows=[scann_ttk[1000], faiss_ttk[1_000]]).bold_max(),
                TableGroup(rows=[scann_ttk[40000], faiss_ttk[40_000]]).bold_max()],
        caption=caption,
        label=label
    )

def run_scann_vs_faiss_analysis():
    table = make_scann_vs_faiss_table(collection="lotte", metric="success@5", caption="", label="")
    table.save("scann_vs_faiss_lotte_success5.tex")

if __name__ == "__main__":
    run_scann_vs_faiss_analysis()