from analysis import Engine, load_experiment_results, metric_round

BEIR_DATASTETS, LOTTE_DATASETS = ["nfcorpus", "fiqa"], ["technology", "pooled"]
EVALUATED_RUNTIMES = [
    "TORCHSCRIPT",
    "ONNX.NONE", "ONNX.PREPROCESS", "ONNX.QUANTIZED_QATTENTION", "ONNX.DYN_QUANTIZED_QINT8",
    "OPENVINO"
]

def evaluate_runtime_metrics(experiment_results, runtime):
    eval = experiment_results.filter(
        runtime=runtime, num_threads=1
    )
    beir = list(eval.filter(dataset=BEIR_DATASTETS, document_top_k=100).view(
        "dataset_id", "ndcg@10", "recall@100"
    ))
    lotte_s5 = {key: value for (key, value) in eval.filter(dataset=LOTTE_DATASETS, document_top_k=100).view(
        "dataset_id", "success@5"
    )}
    lotte_s1000 = {key: value for (key, value) in eval.filter(dataset=LOTTE_DATASETS, document_top_k=1000).view(
        "dataset_id", "success@1000"
    )}
    lotte = [(key, lotte_s5[key], lotte_s1000[key]) for key in lotte_s5.keys()]
    return {key: (metric_round(m1), metric_round(m2)) for key, m1, m2 in beir + lotte}

def evaluate_runtime_latency(experiment_results, runtime):
    NUM_THREADS = sorted(list(set(experiment_results.view("num_threads"))))
    results = []
    for num_threads in NUM_THREADS:
        eval = experiment_results.filter(
            runtime=runtime, num_threads=num_threads
        )
        num_iterations = 0
        total_time_s = 0
        for entry in eval:
            # assert len(entry.latency.tracker) == 1
            num_iterations += entry.latency.tracker[0]["num_iterations"]
            total_time_s += min(*[tracker["time_per_step"]["Query Encoding"] for tracker in entry.latency.tracker])
        average_time_ms = (total_time_s / num_iterations) * 1000
        results.append((num_threads, metric_round(average_time_ms, fact=1)))
    return results

def run_runtime_analysis():
    experiment_results = load_experiment_results(Engine.XTR_WARP, ["warp_eval_runtimes.json"])
    evaluated_runtimes = set(
        experiment_results.view("runtime")
    ) - {None}

    print("===[Metrics]===")
    print("baseline", evaluate_runtime_metrics(experiment_results, runtime=[None]))
    for runtime in EVALUATED_RUNTIMES:
        print(runtime, evaluate_runtime_metrics(experiment_results, runtime=runtime))

    print("===[Latency]===")
    print("baseline", evaluate_runtime_latency(experiment_results, runtime=[None]))
    for runtime in EVALUATED_RUNTIMES:
        print(runtime, evaluate_runtime_latency(experiment_results, runtime=runtime))

if __name__ == "__main__":
    run_runtime_analysis()