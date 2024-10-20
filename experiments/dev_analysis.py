from analysis import Engine, load_experiment_results, metric_round, PRETTY_NAMES, LOTTE_DATASETS, BEIR_DATASETS

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class WrappedLambda:
    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

    def __call__(self, *args, **kwds):
        return self.fn(*args, **kwds)

    def __repr__(self):
        return self.name

def plot_nbits_analysis(warp, reference):
    nprobe_nbits = WrappedLambda(name="aa", fn=lambda x: f"{x.provenance.nprobe}-{x.provenance.nbits}")
    warp.filter(dataset="lotte.pooled.dev", nprobe=[16, 32], nbits=[2, 4]).plot(
        "t_prime", "success@5", group_by=nprobe_nbits
    )
    plt.show()

def get_ref_score(reference, dataset, metric, token_top_k):
    ref = reference.filter(dataset=dataset, token_top_k=token_top_k, num_threads=1)
    assert len(ref) == 1
    ref = ref[0]
    return ref.view(metric)

def plot_tprime_analysis(warp, reference, dataset, metric):
    plt.figure(figsize=(12, 8))
    
    MINOR, MAJOR = 20, 24

    ref_score = get_ref_score(reference, dataset, metric, token_top_k=40_000)
    plt.text(-3_500, ref_score + 0.004, "XTR$_\\text{base}$/ScaNN (k'$=40$k)", color="grey", fontsize=12)
    plt.axhline(y=ref_score, color="grey", linestyle="--", zorder=-1)

    ref_score = get_ref_score(reference, dataset, metric, token_top_k=1_000)
    plt.axhline(y=ref_score, color="grey", linestyle="--", zorder=-1)
    plt.text(-3_500, ref_score + 0.004, "XTR$_\\text{base}$/ScaNN (k'$=1$k)", color="grey", fontsize=12)

    warp.filter(dataset=dataset, nbits=4, nprobe=[1, 2, 4, 8, 16, 32, 64]).plot("t_prime", metric, group_by="nprobe")

    plt.ylabel(metric, fontsize=MAJOR)
    plt.xlabel("t'", fontsize=MAJOR)

    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    xticks = [0, 20_000, 40_000, 60_000, 80_000, 100_000]
    plt.xticks(xticks, [f"{x // 1000}k" for x in xticks], fontsize=MINOR)

    plt.tick_params(axis="both", which="major", labelsize=MINOR)
    plt.grid()

    # Add an x-label
    plt.axvline(x=70_000, color="tab:red", linestyle="--", zorder=1)
    plt.text(70_000 - 2_050, plt.gca().get_ylim()[0] + 0.01, "Vertical Line", color="tab:red", fontsize=12, rotation=90)

    plt.legend(fontsize=MINOR)
    plt.show()

def run_dev_analysis():
    warp = load_experiment_results(Engine.XTR_WARP, ["warp_eval_dev_sets.json"])
    reference = load_experiment_results(Engine.XTR_REFERENCE, ["xtr_eval_dev_sets.json"])

    # plot_nbits_analysis(warp, reference)
    # plot_tprime_analysis(warp, reference, dataset="lotte.pooled.dev", metric="success@100")
    plot_tprime_analysis(warp, reference, dataset="lotte.pooled.dev", metric="success@1000")

if __name__ == "__main__":
    run_dev_analysis()