from analysis import Engine, load_experiment_results, metric_round, PRETTY_NAMES, LOTTE_DATASETS, BEIR_DATASETS
from analysis.table import TableRow, TableEntry, TableGroup, Table, Template, Footnote, format_model_name, compute_row_avg

def make_shift_table(collection, metric, caption, label):
    regular_results = load_experiment_results(Engine.XTR_REFERENCE, ["xtr_eval_dev_sets.json"])
    shift_results = load_experiment_results(Engine.XTR_REFERENCE, ["xtr_eval_dev_sets_shift.json"])
    
    datasets = {"lotte": LOTTE_DATASETS, "beir": BEIR_DATASETS}[collection]["dev"]

    K_MAP = {1000: "1\,000", 1125: "1\,125", 875: "875", 40000: "40\,000", 45000: "45\,000", 35000: "35\,000"}

    ttk = dict()
    for token_top_k in [1_000, 40_000]:
        row = [TableEntry(text=f"XTR$_\\text{{base}}$/ScaNN (k'$={K_MAP[token_top_k]}$)", bold=True)]
        for dataset in datasets:
            entry = regular_results.filter(token_top_k=token_top_k, collection=collection, dataset=dataset, num_threads=1)
            assert len(entry) == 1
            entry = entry[0]

            dataset = entry.view("dataset")
            score = entry.view(metric)
            row.append(TableEntry(text=f"{metric_round(score)}"))
        row.append(compute_row_avg(row))
        ttk[token_top_k] = TableRow(data=row, gray_background=True)

    for token_top_k in sorted(list(set(shift_results.view("token_top_k")))):
        row = [TableEntry(text=f"XTR$_\\text{{base}}$/ScaNN (k'$={K_MAP[token_top_k]}$)")]
        for dataset in datasets:
            entry = shift_results.filter(token_top_k=token_top_k, collection=collection, dataset=dataset, num_threads=1)
            assert len(entry) == 1
            entry = entry[0]

            dataset = entry.view("dataset")
            score = entry.view(metric)
            row.append(TableEntry(text=f"{metric_round(score)}"))
        row.append(compute_row_avg(row))
        ttk[token_top_k] = TableRow(data=row, gray_background=False)

    return Table(
        columns=[PRETTY_NAMES[dataset] for dataset in datasets],
        groups=[TableGroup(rows=[ttk[875], ttk[1_000], ttk[1125]]),
                TableGroup(rows=[ttk[35000], ttk[40_000], ttk[45000]])],
        caption=caption,
        label=label
    )

def run_xtr_k_prime_shift_analysis():
    table = make_shift_table(collection="lotte", metric="success@5", caption="None", label="A")
    table.save("shift_lotte_success5.tex")

    table = make_shift_table(collection="beir", metric="ndcg@10", caption="None", label="A")
    table.save("shift_beir_ndcg10.tex")

    table = make_shift_table(collection="beir", metric="recall@100", caption="None", label="A")
    table.save("shift_beir_recall100.tex")



if __name__ == "__main__":
    run_xtr_k_prime_shift_analysis()