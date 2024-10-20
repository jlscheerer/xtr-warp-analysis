import os
import math
import json

from enum import Enum, auto
from typing import Literal, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict
from itertools import product
from dotenv import load_dotenv

import matplotlib.pyplot as plt
import numpy as np

load_dotenv()

# Ordered by #Passages.
BEIR_DATASETS = {
    "dev": ["nfcorpus", "fiqa", "quora"],
    "test": ["nfcorpus", "scifact", "scidocs", "fiqa", "webis-touche2020", "quora"]
}

LOTTE_DATASETS = {
    "dev": ["recreation", "lifestyle", "writing", "science", "technology", "pooled"],
    "test": ["lifestyle", "recreation", "writing", "technology", "science", "pooled"]
}

def _load_dataset_stats(files):
    dataset_stats = dict()
    for filename in files:
        with open(filename, "r") as file:
            stats = json.loads(file.read())
            for key, value in stats.items():
                dataset_stats[key] = value
    return dataset_stats

DATASET_STATS = _load_dataset_stats([
    os.path.join(os.environ["STATISTICS_DIRECTORY"], f"stats.{split}.json")
    for split in ["dev", "test"]
])

METRICS = ["recall", "ndcg", "map", "success"]
STATISTICS = ["passages", "avg_doclen", "embeddings", "median_size", "embeddings", "centroids"]
PROVENANCES = [
    "nprobe", "t_prime", "nbits", "split", "bound", "collection", "dataset",
    "document_top_k", "runtime", "num_threads", "type", "parameters", "fused_ext",
    "ablation_params", "index_type", "optimized", "token_top_k"
]

# Source: https://stackoverflow.com/questions/33019698/how-to-properly-round-up-half-float-numbers
def regular_round(n, decimals=0):
    expoN = n * 10 ** decimals
    if abs(expoN) - abs(math.floor(expoN)) < 0.5:
        return math.floor(expoN) / 10 ** decimals
    return math.ceil(expoN) / 10 ** decimals

def metric_round(value, fact=100, decimals=1):
    return regular_round(value * fact, decimals=decimals)

def prettify_large_num(value, decimals=1):
    if value < 1_000:
        return str(value)
    if value < 1_000_000:
        return "{:.0{prec}f}K".format(value / 1_000, prec=decimals)
    if value < 1_000_000_000:
        return "{:.0{prec}f}M".format(value / 1_000_000, prec=decimals)
    if value < 1_000_000_000_000:
        return "{:.0{prec}f}B".format(value / 1_000_000_000, prec=decimals)
    else: raise AssertionError


def prettify_dataset_name(dataset_id):
    pretty_names = {
        "nfcorpus" : "NFCorpus", "scifact": "SciFact", "scidocs": "SCIDOCS",
        "quora": "Quora", "fiqa": "FiQA-2018", "webis-touche2020": "Touche-2020",
    }
    collection, dataset, split = dataset_id.split(".")
    split_suffix = "" if split == "test" else " (Dev)"
    if dataset in pretty_names:
        dataset = pretty_names[dataset]
    else:
        dataset = dataset.title()
    return f"{dataset}{split_suffix}"

@dataclass
class View:
    @staticmethod
    def make_views(*views):
        if len(views) == 0:
            return []
        result = []
        for view in views:
            if isinstance(view, View):
                return view
                result.append(view)
            if view in STATISTICS:
                result.append(StatisticsView(view))
            elif view in (PROVENANCES + ["dataset_id"]):
                result.append(ProvenanceView(view))
            elif isinstance(view, str) and "@" in view:
                result.append(MetricView(view))
            else:
                result.append(CallableView(view))
        return result
    
@dataclass
class ProvenanceView(View):
    key: str

    def __init__(self, key):
        assert key in (PROVENANCES + ["dataset_id"])
        self.key = key

    def __repr__(self):
        return self.key

@dataclass
class MetricView(View):
    metric: str
    k: int

    def __init__(self, metric, k=None):
        if "@" in metric:
            assert k is None
            metric, k = metric.split("@")
            k = int(k)
        assert k is not None and metric in METRICS
        
        self.metric = metric
        self.k = k

    def __repr__(self):
        return f"{self.metric}@{self.k}"

@dataclass
class StatisticsView(View):
    statistic: str

    def __init__(self, statistic):
        assert statistic in STATISTICS
        self.statistic = statistic

    def __repr__(self):
        return self.statistic

@dataclass
class CallableView(View):
    view: None

    def __init__(self, view):
        self.view = view

    def extract(self, measurement):
        return self.view(measurement)

@dataclass
class Dataset:
    collection: Literal["beir", "lotte"] = None
    dataset: str = None
    split: Literal["dev", "test"] = None

    def id(self):
        return f"{self.collection}.{self.dataset}.{self.split}"

@dataclass
class Provenance:
    dataset: Dataset
    nbits: Literal[2, 4] = None
    document_top_k: int = None
    runtime: Optional[str] = None
    nprobe: int = None
    t_prime: int = None
    bound: int = None
    num_threads: int = None
    type: str = None
    parameters: dict = None
    fused_ext: bool = None
    ablation_params: dict = None

    # xtr-eval
    index_type: str = None
    optimized: bool = None
    token_top_k: int = None

    def __init__(self, dataset, document_top_k, num_threads, parameters,
                 t_prime=None, bound=None, runtime=None, nbits=None, nprobe=None, fused_ext=True, ablation_params=None, # xtr-warp
                 index_type=None, optimized=None, token_top_k=None, # xtr-eval
                 type=None, collection=None, split=None):
        if isinstance(dataset, Dataset) or (isinstance(dataset, list) and (len(dataset) == 0 or isinstance(dataset[0], Dataset))):
            assert collection is None and split is None
            self.dataset = dataset
        else:
            self.dataset = Dataset(collection=collection, dataset=dataset, split=split)
        self.nbits = nbits
        self.nprobe = nprobe
        self.document_top_k = document_top_k
        self.runtime = runtime
        self.t_prime = t_prime
        self.bound = bound
        self.num_threads = num_threads
        self.type = type
        self.parameters = parameters
        self.fused_ext = fused_ext
        self.ablation_params = ablation_params

        self.index_type = index_type
        self.optimized = optimized
        self.token_top_k = token_top_k
    
    def __getitem__(self, key):
        if key in ["collection", "dataset", "split"]:
            return getattr(self.dataset, key)
        return getattr(self, key)

    def dataset_id(self):
        return self.dataset.id()

    def view(self, view):
        if view.key == "dataset_id":
            return self.dataset_id()
        return self[view.key]
    
    def __repr__(self):
        data = asdict(self)
        keys = [x for x in data.keys() if x not in ["collection", "dataset", "split"]]
        dict_repr = {"dataset": self.dataset_id()}
        dict_repr.update({
            key: data[key] for key in keys
        })
        str_repr = ", ".join([f"{key}={value}" for key, value in dict_repr.items()])
        return f"({str_repr})"

@dataclass
class LatencyEvaluation:
    tracker: list

    def __init__(self, tracker):
        self.tracker = tracker

    def __getitem__(self, key):
        """
        Returns the execution time of a step in milliseconds.
        """
        assert len(self.tracker) == 1
        return self.step(key)
        
    def step(self, key, mode="min"):
        if len(self.tracker) == 1:
            return 1000 * (self.tracker[0]["time_per_step"][key] / self.tracker[0]["num_iterations"])
        mode_fn = {
            "min": min,
            "max": max,
            "avg": lambda x: sum(x) / len(x)
        }[mode]
        aggregate = mode_fn([tracker["time_per_step"][key] for tracker in self.tracker])
        return 1000 * aggregate / self.tracker[0]["num_iterations"]

    def steps(self):
        return list(self.tracker[0]["steps"])

    def avg_latency(self):
        return sum([tracker["iteration_time"] for tracker in self.tracker]) / sum([tracker["num_iterations"] for tracker in self.tracker])

    def sum(self):
        return sum([tracker["iteration_time"] for tracker in self.tracker]) / sum([tracker["num_iterations"] for tracker in self.tracker])

    def plaid_style(self, single_run=False):
        if not single_run:
            # As specified in the PLAID paper, take the minimum AVERAGE latency over three runs
            assert len(self.tracker) == 3
        num_iterations = self.tracker[0]["num_iterations"]
        keys = list(self.tracker[0]["time_per_step"].keys())
        
        assert all(tracker["num_iterations"] == num_iterations for tracker in self.tracker)
        assert all(list(tracker["time_per_step"].keys()) == keys for tracker in self.tracker)

        def latency_for_key(key):
            if single_run:
                return 1000 * self.tracker[0]["time_per_step"][key] / num_iterations
            return 1000 * (min(*[tracker["time_per_step"][key] for tracker in self.tracker]) / num_iterations)

        return {
            key: latency_for_key(key)
            for key in keys
        }

    def avg_grouped_latency(self, single_run=False):
        if not single_run:
            assert len(self.tracker) == 3
        num_iterations = self.tracker[0]["num_iterations"]
        keys = list(self.tracker[0]["time_per_step"].keys())
        
        assert all(tracker["num_iterations"] == num_iterations for tracker in self.tracker)
        assert all(list(tracker["time_per_step"].keys()) == keys for tracker in self.tracker)

        def latency_for_key(key):
            if single_run:
                return 1000 * self.tracker[0]["time_per_step"][key] / num_iterations
            return 1000 * (sum([tracker["time_per_step"][key] for tracker in self.tracker]) / len(self.tracker) / num_iterations)

        return {
            key: latency_for_key(key)
            for key in keys
        }

@dataclass
class Statistics:
    centroids: int
    embeddings: int
    median_size: int
    passages: int
    avg_doclen: int

    def __init__(self, dataset, centroids, embeddings, median_size):
        self.centroids = centroids
        self.embeddings = embeddings
        self.median_size = median_size
        stats = DATASET_STATS[dataset]
        self.passages = stats["passages"]
        self.avg_doclen = stats["avg_doclen"]

    def view(self, view):
        return getattr(self, view.statistic)

@dataclass
class Metrics:
    data: dict = None

    def __init__(self, data):
        metrics = defaultdict(dict)
        for key, value in data.items():
            metric, k = key.split("@")
            metric, k = metric.replace("_cut", "").lower(), int(k)
            metrics[metric][k] = value
        self.data = dict(metrics)

    def __repr__(self):
        def repr_values(values):
            return ", ".join([
                f"@{key}: {value:.2f}" for key, value in values.items()
            ])

        data = ", ".join([
            f"{key}=[{repr_values(values)}]" for key, values in self.data.items()
        ])
        return f"({data})"

    def __getitem__(self, key):
        metric, k = key.split("@")
        return self.data[metric][int(k)]
    
    def view(self, view):
        if view.metric not in self.data:
            return None
        if view.k not in self.data[view.metric]:
            return None
        return self.data[view.metric][view.k]

@dataclass
class Matcher(Provenance):
    def __init__(self, dataset=None, nbits=None, nprobe=None, document_top_k=None, t_prime=None, bound=None, runtime=None, 
                 num_threads=None, parameters=None, fused_ext=None, type=None, collection=None, split=None, ablation_params=None,
                 index_type=None, optimized=None, token_top_k=None):
        super().__init__(dataset=dataset, nbits=nbits, nprobe=nprobe, document_top_k=document_top_k, t_prime=t_prime, bound=bound, runtime=runtime, 
                         num_threads=num_threads, parameters=parameters, fused_ext=fused_ext, type=type, collection=collection, split=split, ablation_params=ablation_params,
                         index_type=index_type, optimized=optimized, token_top_k=token_top_k)        
    
    def matches(self, provenance):
        def match_dataset(dataset):
            for key in ["collection", "dataset", "split"]:
                value = getattr(dataset, key)
                if value is not None and value != provenance[key]:
                    return False
            return True
            
        if isinstance(self.dataset, list):
            if all(not match_dataset(x) for x in self.dataset):
                return False
        else:
            if not match_dataset(self.dataset):
                return False

        for key in ["nbits", "nprobe", "t_prime", "bound", "document_top_k", "runtime", "num_threads",
                    "type", "parameters", "fused_ext", "ablation_params", "index_type", "optimized", "token_top_k"]:
            value = self[key]
            if value is None:
                continue
            if isinstance(value, list):
                if provenance[key] not in value:
                    return False
            elif provenance[key] != value:
                return False
        return True

    @staticmethod
    def build(collection=None, dataset=None, nbits=None, nprobe=None, document_top_k=None, t_prime=None, bound=None, runtime=None, 
              num_threads=None, parameters=None, fused_ext=None, type=None, split=None, ablation_params=None,
              index_type=None, optimized=None, token_top_k=None):
        def upcast(item, required=False):
            if item is None:
                return [[None]] if required else [[]]
            if isinstance(item, list):
                if len(item) == 1 and isinstance(item[0], list) and len(item[0]) == 1 and item[0][0] is None:
                    return item
                return [[x] for x in item]
            return [[item]]

        is_none = (dataset is None)
        if not isinstance(dataset, list):
            dataset = [dataset]
            
        if len(dataset) != 0:
            dataset_components = [x.split(".") for x in dataset] if not is_none else [[None]]
            has_collection = all([x[0] in ["lotte", "beir"] for x in dataset_components])
            has_split = all([x[-1] in ["dev", "test"] for x in dataset_components])
            assert min(len(x) for x in dataset_components) == max(len(x) for x in dataset_components)
            components = min(len(x) for x in dataset_components)
            collection, split = upcast(collection, required=not has_collection), upcast(split, required=not has_split)
            dataset_components = [
                c + d + s for c, d, s in product(collection, dataset_components, split)
            ]
            dataset = [
                Dataset(collection=x[0], dataset=x[1], split=x[2]) for x in dataset_components
            ]
            
        return Matcher(dataset=dataset, nbits=nbits, nprobe=nprobe, t_prime=t_prime, document_top_k=document_top_k, bound=bound,
                       runtime=runtime, num_threads=num_threads, parameters=parameters, fused_ext=fused_ext, type=type, ablation_params=ablation_params,
                       index_type=index_type, optimized=optimized, token_top_k=token_top_k)

@dataclass
class Measurement:
    provenance: Provenance = None
    metrics: Metrics = None
    statistics: Statistics = None
    latency: LatencyEvaluation = None
    additional_data: dict = None

    def __init__(self, content):
        if isinstance(content, dict):
            self.provenance = Provenance(**content["provenance"])
            if "metrics" in content:
                self.metrics = Metrics(data=content["metrics"])
            if "statistics" in content:
                self.statistics = Statistics(dataset=self.provenance.dataset_id(), **content["statistics"])
            if "tracker" in content:
                self.latency = LatencyEvaluation(tracker=content["tracker"])
            self.additional_data = dict()
            for key in content:
                if key in ["provenance", "metrics", "statistics", "tracker"]:
                    continue
                self.additional_data[key] = content[key]
            if len(self.additional_data) == 0:
                self.additional_data = None
        else:
            self.provenance = content.provenance
            self.metrics = content.metrics
            self.statistics = content.statistics
            self.latency = content.latency

    def view(self, *args, partial=False):
        views = View.make_views(*args)
        results = []
        for view in views:
            if isinstance(view, StatisticsView):
                result = self.statistics.view(view)
            elif isinstance(view, MetricView):
                result = self.metrics.view(view)
            elif isinstance(view, ProvenanceView):
                result = self.provenance.view(view)
            else:
                assert isinstance(view, CallableView)
                result = view.extract(self)
            
            if result is None and not partial:
                return None
            results.append(result)
        if len(results) == 1:
            return results[0]
        return tuple(results)
    
    def __repr__(self):
        return f"Measurement(Provenance={self.provenance}, Metrics={self.metrics})"

@dataclass
class ViewedCollection:
    views: list[View]
    data: list[tuple]

    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        return f"{self.data}"

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return self.data.__iter__()

    def _partition2(self):
        assert len(self.views) == 2
        xys = sorted(self.data, key=lambda x: x[0])
        xs = [x[0] for x in xys]
        ys = [x[1] for x in xys]
        return xs, ys

    def filter_fn(self, fn):
        return ViewedCollection(self.views, [x for x in self.data if fn(x)])

    def filter(self, fn):
        return self.filter_fn(fn)

    def map_fn(self, fn):
        return ViewedCollection(self.views, [fn(x) for x in self.data])

    def map(self, fn):
        return self.map_fn(fn)
    
    def agg(self, fn):
        assert len(self.views) == 1
        return fn(self.data)

    def max(self):
        return self.agg(fn=max)

    def project(self, *args):
        new_views = [self.views[x] for x in args]
        new_data = []
        for entry in self.data:
            new_data.append(tuple([entry[x] for x in args]))
            if len(args) == 1:
                new_data[-1] = new_data[-1][0]
        return ViewedCollection(new_views, new_data)
    
    def plot(self, title=None):
        plt.plot(*self._partition2(), marker="o")

        if title is not None:
            plt.title(f"{title}")
        
        plt.xlabel(self.views[0])
        plt.ylabel(self.views[1])

    def scatter(self, title=None, colors=None):
        xs, ys = self._partition2()
        plt.scatter(xs, ys, marker="o", c=colors)
        
        if title is not None:
            plt.title(f"{title}")
        
        plt.xlabel(self.views[0])
        plt.ylabel(self.views[1])

        return np.linspace(min(xs), max(xs), 100)

@dataclass
class Collection:
    data: list

    def __init__(self, data: list):
        if isinstance(data, Collection):
            self.data = data
        else:
            self.data = [Measurement(x) for x in data]
    
    def filter(self, **kwargs):
        matcher = Matcher.build(**kwargs)
        return self.filter_fn(lambda x: matcher.matches(x.provenance))

    def filter_fn(self, fn):
        filtered = []
        for result in self.data:
            if fn(result):
                filtered.append(result)
        return Collection(data=filtered)
    
    def group_by(self, *views):
        split = defaultdict(list)
        for result in self.data:
            key = result.view(*views)
            split[key].append(result)
        return GroupedCollection(key=View.make_views(*views), groups=split)

    def view(self, *views, partial=False, verbose=False):
        viewed, dropped = [], 0
        for result in self.data:
            view = result.view(*views)
            if view is None:
                dropped += 1
            else: viewed.append(view)
        if dropped != 0 and verbose:
            print(f"#> [WARNING]: Dropped {dropped} results")
        return ViewedCollection(views=View.make_views(*views), data=viewed)

    def plot(self, *views, group_by=None, **kwargs):
        if group_by is not None:
            self.group_by(group_by).view(*views).plot(**kwargs)
            return
        self.view(*views).plot(**kwargs)
    
    def __iter__(self):
        return self.data.__iter__()
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __repr__(self):
        return f"{self.data}"

    def __len__(self):
        return len(self.data)

@dataclass
class GroupedViewCollection:
    key: list[View]
    groups: dict

    def __init__(self, key, groups):
        self.key = key
        self.groups = groups

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, key):
        return self.groups[key]
    
    def __iter__(self):
        return self.groups.items().__iter__()

    def agg(self, *args, **kwargs):
        return self._lift(lambda _, value: value.agg(*args, **kwargs))

    def max(self):
        return self.agg(fn=max)
    
    def _lift(self, fn):
        return GroupedViewCollection(key=self.key, groups={x: fn(x, y) for x, y in self.groups.items()})
    
    def plot(self, title=None):
        assert len(self.key) == 1
        assert len(self.groups) != 0
        
        for key, value in self.groups.items():
            plt.plot(*value._partition2(), marker="o", label=f"{self.key[0]}={key}")

        if title is not None:
            plt.title(f"{title}")

        views = next(iter(self.groups.values())).views        
        plt.xlabel(views[0])
        plt.ylabel(views[1])
        
        plt.legend()
    
    def __repr__(self):
        return f"GroupedViewCollection({self.groups})"
        
@dataclass
class GroupedCollection:
    key: list[View]
    groups: dict

    def __init__(self, key, groups):
        self.key = key
        self.groups = {
            key: Collection(value) for key, value in groups.items()
        }

    def filter(self, *args, **kwargs):
        return self._lift(lambda _, value: value.filter(*args, **kwargs))

    def filter_fn(self, *args, **kwargs):
        return self._lift(lambda _, value: value.filter_fn(*args, **kwargs))
    
    def view(self, *args, **kwargs):
        return self._lift(lambda _, value: value.view(*args, **kwargs), viewed=True)

    def plot(self, *views, **kwargs):
        self.view(*views).plot(**kwargs)
    
    def _lift(self, fn, viewed=False):
        groups = {
            key: fn(key, value) for key, value in self.groups.items()
        }
        parent = GroupedCollection
        if viewed:
            parent = GroupedViewCollection
        return parent(key=self.key, groups={x: y for x, y in groups.items() if len(y) != 0})
    
    def __getitem__(self, key):
        return self.groups[key]
    
    def __iter__(self):
        return self.groups.items().__iter__()

    def __repr__(self):
        return f"GroupedCollection({self.groups})"

class Engine(Enum):
    XTR_WARP = auto()
    XTR_REFERENCE = auto()
    COLBERTv2_PLAID = auto()

def _resolve_engine_directory(engine: Engine):
    if engine == Engine.XTR_WARP:
        return os.environ["XTR_WARP_RESULTS_DIRECTORY"]
    elif engine == Engine.XTR_REFERENCE:
        return os.environ["XTR_REFERENCE_RESULTS_DIRECTORY"]
    elif engine == Engine.COLBERTv2_PLAID:
        return os.environ["COLBERT_REFERENCE_RESULTS_DIRECTORY"]
    else: raise AssertionError

def load_experiment_results(engine: Engine, filenames: list[str]):
    provenances = []
    experiment_results = []
    for filename in filenames:
        root = _resolve_engine_directory(engine)
        with open(os.path.join(root, filename), "r") as file:
            data = json.loads(file.read())
        for result in data:
            provenance = result["provenance"]
            if provenance in provenances:
                continue
            provenances.append(provenance)
            experiment_results.append(result)
    return Collection(data=experiment_results)


PRETTY_NAMES = {
    "nfcorpus": "NFCorpus",
    "scifact": "SciFact",
    "scidocs": "SCIDOCS",
    "fiqa": "FiQA-2018",
    "webis-touche2020": "Touché-2020",
    "quora": "Quora",
    "lifestyle": "Lifestyle",
    "writing": "Writing",
    "recreation": "Recreation",
    "technology": "Technology",
    "science": "Science",
    "pooled": "Pooled",
}

def prettify_dataset_name(dataset_id):
    collection, dataset, split = dataset_id.split(".")
    collection = {
        "lotte": "LoTTE", "beir": "BEIR"
    }[collection]
    return f"{collection} {PRETTY_NAMES[dataset]} ({split.title()})"