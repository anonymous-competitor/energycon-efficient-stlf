# Efficient electricity consumption forecasting using hybrid LSTM and CNN with transfer learning

<p align="center">
  <img src="img/IEEE-Region-8-Logo.png" width="180" alt="IEEE R8">
</p>

<p align="center"><strong>Implementation of the IEEE ENERGYCON 2026 Student Paper</strong></p>

## Abstract

Achieving efficient and accurate forecasting of household electricity consumption is challenging due to volatile usage patterns and resource-intensive training, yet it is essential for grid scheduling, operational planning and integration of renewable energy. We propose a transfer learning method using a CNN–LSTM universal model trained on a subset of households and adapted through fine-tuning, combined with efficient hyper-parameter optimization. Experiments on daily consumption data show a 68–85% reduction in training time compared to existing conventional training method, with statistically lower average short-term prediction error and marginal long-term deviation, confirming the effectiveness of the proposed method for efficient, adaptable consumption forecasting.

## Methodology

In short our methodology consists of the following steps:

1. Train universal model on train subset of all consumer time series data
   - Optimize HPO by using innovative early-stopping criteria
   - Pick the one universal model based on performance on validation subset
2. Fine-tune universal model to all consumer time series data
3. Report the results

## Table of contents

1. [Abstract](#abstract)
2. [Methodology](#methodology)
3. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
4. [Usage Examples](#usage-examples)
5. [Documentation](#documentation)
   - [Folder Structure](#folder-structure)
   - [Experiment Configuration](#experiment-configuration)
   - [Experiment Results](#experiment-results)

## Getting Started

### Prerequisites

- GPU recommended (CPU supported)
- Python 3.10.5
- CUDA 576.83
- PyTorch 2.5.1

### Installation

To install all required packages run:

```
pip install requirements.txt
```

This project doesn't use package manager, since it was created for experimental purposes. In the future package mismatch might cause issues.

## Usage examples

Code entry point is `main.py` which works as a console application. It supports following options:

```
python main.py -d
```

- `-d` - data preprocessing. This will preprocess raw data from `data/raw` folder and save the modified data to `data/proccessed`. Preprocessing is explained in the paper section <b>III. Methodology A. Data preprocessing</b>.

```
python main.py -e
```

- `-e` - run experiment. This flag will execute experiment based on experiment settings from `experiment.json` by running conventional training for LSTM, CNN-LSTM then universal model training and finally fine-tuning.

```
python main.py -r
```

- `-r` - obtain results. This flag will create 3 aggregated tables (training time, performance and statistical analysis) in the `results` folder. These tables are presented in <b>IV. Experiment</b> section.

## Documentation

### Folder Structure

- `data` - contains raw and processed data
- `img` - images for readme
- `logs` - loss, performance metrics, trial details and evaluation data are saved here during the training.
- `models` - trained models (.pt), because of size this folder remains local
- `results` - aggregated tables for training time, performance and statistic test
- `src` - source code for the entire project
  - `analysis` - code for results aggregation and statistic analysis
  - `data` - code for data processing
  - `logging` - code for logging loss, performance metrics, etc. during training
  - `models` - code for model architecture definitions
  - `notebooks` - contains notebooks for visualisation of data preprocessing and results
  - `training` - code for conventional and proposed training approaches.

### Experiment Configuration

To change experiment configuration, you can modify `experiment.json`. Parameters are presented in the following table:

| Parameter      | Description                                                          |
| :------------- | :------------------------------------------------------------------- |
| `reps`         | Number of experiment repetitions.                                    |
| `seeds`        | List of different seeds for each repetition and `len(seeds) == reps` |
| `epochs`       | Number of training epochs                                            |
| `min_epochs`   | Minimal number of training epochs before termination                 |
| `patience`     | Number of epochs to wait for improvement of results in fine-tuning   |
| `trials`       | Number of different hyper-parameter combinations                     |
| `device`       | CUDA or CPU                                                          |
| `lookback`     | Number of past load values before prediction                         |
| `horizon`      | Array of forecasting horizons to be experimented on                  |
| `search_space` | Hyper-parameters for model tuning                                    |
