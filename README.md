# XTR/WARP Analysis Scripts

> This repository provides scripts to generate the plots, tables, and figures presented in the thesis.

## Installation

It is strongly recommended to create a [conda environment](https://docs.anaconda.com/anaconda/install/linux/#installation) using the commands below. We include the corresponding environment file (`environment.yml`).

```sh
conda env create -f environment.yml
conda activate xtr-warp-analysis
```

### Environment Setup
To generate the desired artifacts, define the following variables in a `.env` file at the root of the repository:

```sh
STATISTICS_DIRECTORY="statistics/"
XTR_WARP_RESULTS_DIRECTORY="..."
XTR_REFERENCE_RESULTS_DIRECTORY="..."
COLBERT_REFERENCE_RESULTS_DIRECTORY="..."
```

- `XTR_WARP_RESULTS_DIRECTORY`: Path to the experiment results of [XTR/WARP](https://github.com/jlscheerer/xtr-warp)
- `XTR_REFERENCE_RESULTS_DIRECTORY`: Path to the experiment results of [xtr-eval](https://github.com/jlscheerer/xtr-eval)
- `COLBERT_REFERENCE_RESULTS_DIRECTORY`: Path to the experiment results of [colbert-eval](https://github.com/jlscheerer/colbert-eval)

### Usage
Run the following command to generate plots and tables from your experiment results:
```sh
python experiments/EXPERIMENT_NAME
````
**Note**: Replace `EXPERIMENT_NAME` with the name of your experiment.

**Example:**
```sh
python experiments/warp_gantt.py
````

The generated plots and tables will be saved in the `output/` directory.


