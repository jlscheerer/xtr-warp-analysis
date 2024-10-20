from analysis import Engine, load_experiment_results, BEIR_DATASETS, LOTTE_DATASETS, DATASET_STATS, PRETTY_NAMES, prettify_large_num, metric_round

from collections import defaultdict

def _bytes_to_gib(bytes_value):
    return bytes_value / (1024 ** 3)

def get_warp_index_sizes(split):
    warp_experiment_results = load_experiment_results(Engine.XTR_WARP, [f"xtr_warp_index_sizes_{split}_sets.json"])
    grouped_by_collection = defaultdict(lambda: defaultdict(dict))
    for entry in warp_experiment_results:
        collection, dataset = entry.view("collection"), entry.view("dataset")
        nbits = entry.view("nbits")
        index_size = _bytes_to_gib(entry.additional_data["index_size_bytes"])
        grouped_by_collection[collection][dataset][nbits] = index_size
    return dict({x: dict(y) for x, y in grouped_by_collection.items()})

def get_xtr_index_sizes(split):
    xtr_experiment_results = load_experiment_results(Engine.XTR_REFERENCE, [f"xtr_eval_index_sizes_{split}_sets.json"])
    grouped_by_collection = defaultdict(lambda: defaultdict(dict))
    for entry in xtr_experiment_results:
        collection, dataset = entry.view("collection"), entry.view("dataset")
        index_type = entry.view("index_type")
        index_size = _bytes_to_gib(entry.additional_data["index_size_bytes"])
        grouped_by_collection[collection][dataset][index_type] = index_size
    return dict({x: dict(y) for x, y in grouped_by_collection.items()})


def construct_table_row(collection, dataset, split, xtr_index_sizes, warp_index_sizes, decimals=2, show_passages=True):
    passages = prettify_large_num(DATASET_STATS[f"{collection}.{dataset}.{split}"]["passages"], decimals=decimals)
    tokens = prettify_large_num(DATASET_STATS[f"{collection}.{dataset}.{split}"]["embeddings"], decimals=decimals)
    xtr_index_sizes = xtr_index_sizes[collection][dataset]
    warp_index_sizes = warp_index_sizes[collection][dataset]
    prf = lambda x: "{:.0{prec}f}".format(metric_round(x, decimals=decimals, fact=1), prec=decimals)
    if split == "test":
        bruteforce = f" & {prf(xtr_index_sizes['bruteforce'])}"
    else:
        bruteforce = ""
    if show_passages:
        passages = f" & {passages}"
    else:
        passages = ""
    return f"& {PRETTY_NAMES[dataset]}{passages} & {tokens}{bruteforce} & {prf(xtr_index_sizes['faiss'])} & {prf(xtr_index_sizes['scann'])} & {prf(warp_index_sizes[2])} & {prf(warp_index_sizes[4])} \\\\\n"

def group_nested_2(values):
    result = defaultdict(float)
    for key1, values1 in values.items():
        for key2, values2 in values1.items():
            for key, value in values2.items():
                result[key] += value
    return dict(result)

def construct_table_summary(split, xtr_index_sizes, warp_index_sizes, decimals=2, show_passages=True):
    total_num_passages, total_num_tokens = 0, 0
    for dataset_id, stats in DATASET_STATS.items():
        if not dataset_id.endswith(f".{split}"):
            continue
        total_num_passages += stats["passages"]
        total_num_tokens += stats["embeddings"]
    total_num_passages = prettify_large_num(total_num_passages, decimals=decimals)
    total_num_tokens = prettify_large_num(total_num_tokens, decimals=decimals)
    xtr_index_sizes = group_nested_2(xtr_index_sizes)
    warp_index_sizes = group_nested_2(warp_index_sizes)
    prf = lambda x: "{:.0{prec}f}".format(metric_round(x, decimals=decimals, fact=1), prec=decimals)
    if split == "test":
        bruteforce = f" & {prf(xtr_index_sizes['bruteforce'])}"
    else:
        bruteforce = ""
    if show_passages:
        passages = f" & {total_num_passages}"
    else:
        passages = ""
    return f"\n\\midrule\\rowcolor[gray]{{0.90}}\\multirow{{1}}{{*}}{{\\textbf{{Total}}}} & --{passages} & {total_num_tokens}{bruteforce} & {prf(xtr_index_sizes['faiss'])} & {prf(xtr_index_sizes['scann'])} & \\underline{{\\textbf{{{prf(warp_index_sizes[2])}}}}} & {prf(warp_index_sizes[4])} \\\\ \\bottomrule\n"

def make_index_table(split, show_passages=True):
    xtr_index_sizes = get_xtr_index_sizes(split=split)
    warp_index_sizes = get_warp_index_sizes(split=split)

    # Add the table header
    passage_multi = "\\multirow{2}{*}{\\# Passages} &"
    table = f"""\\begin{{table}}[h]
    \\centering
    \\resizebox{{\\columnwidth}}{{!}}{{%
    \\begin{{tabular}}{{@{{}}ll{'r' if show_passages else ''}r{'r' if split == 'test' else ''}rrrr@{{}}}}
        \\toprule
        \\multicolumn{{{3 + show_passages}}}{{c}}{{}} & \\multicolumn{{{4 + (split == "test")}}}{{c}}{{XTR}} \\\\ 
        \\multicolumn{{2}}{{c}}{{\\multirow{{2}}{{*}}{{Dataset}}}} & {passage_multi if show_passages else ""} \\multirow{{2}}{{*}}{{\\# Tokens}} & \\multicolumn{{{4 + (split == "test")}}}{{c}}{{Index Size (GiB)}} \\\\ \\cmidrule{{{4 + show_passages}-{7 + show_passages + (split == "test")}}}
        \\multicolumn{{2}}{{c}}{{}}{' &' if show_passages else ''} &{' & BruteForce' if split == 'test' else ''} & FAISS & ScaNN & WARP$_{{(b = 2)}}$ & WARP$_{{(b = 4)}}$ \\\\ \\midrule
"""

    table += f"""
        \\multirow{{{len(BEIR_DATASETS[split])}}}{{*}}{{\\textbf{{BeIR}}~\\cite{{beir}} }}
"""

    for dataset in BEIR_DATASETS[split]:
        table += construct_table_row("beir", dataset, split, xtr_index_sizes, warp_index_sizes, show_passages=show_passages)


    table += f"""
        \\midrule
        \\multirow{{{len(LOTTE_DATASETS[split])}}}{{*}}{{\\textbf{{LoTTE}}~\\cite{{colbert2}} }}
"""


    for dataset in LOTTE_DATASETS[split]:
        table += construct_table_row("lotte", dataset, split, xtr_index_sizes, warp_index_sizes, show_passages=show_passages)

    # Add the summary row
    table += construct_table_summary(split, xtr_index_sizes, warp_index_sizes, show_passages=show_passages)

    # Add the table footer
    table += f"""
        \\bottomrule
        \\end{{tabular}}
    }}
    \\caption{{List of benchmarks used for evaluation with relevant statistics.}}
    \\label{{table:index_sizes_{split}}}
\\end{{table}}
    """
    return table

def write_index_table(split, show_passages):
    table = make_index_table(split=split, show_passages=show_passages)
    with open(f"output/index_sizes_{split}.tex", "w") as file:
        file.write(table)

def run_index_sizes_analysis():
    write_index_table(split="test", show_passages=False)
    write_index_table(split="dev", show_passages=False)
    

if __name__ == "__main__":
    run_index_sizes_analysis()