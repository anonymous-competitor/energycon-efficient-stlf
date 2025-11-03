import src.data.preprocess as dp
import src.analysis.results as res
import src.training.train as tr
import src.training.config as conf
import src.logging.log as log

from argparse import ArgumentParser
import os, shutil, random
import numpy as np
import torch


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment():
    # experiment config can be modified in experiment.json
    experiment = conf.EXPERIMENT
    log.create_logs_files()

    # make folder
    os.makedirs(tr.MODELS, exist_ok=True)

    # empty models folder before starting a new run
    shutil.rmtree(tr.MODELS, ignore_errors=True)
    os.makedirs(tr.MODELS, exist_ok=True)

    # for each experiment set seed from experiment.json
    reps = int(experiment["reps"])
    seeds = list(experiment.get("seeds", []))

    for i in range(reps):
        seed = int(seeds[i])
        set_seed(seed)

        tr.conventional_training("LSTM", i)
        tr.conventional_training("CNNLSTM", i)
        tr.universal_model_training(i)
        tr.fine_tuning(i)


# Check folder results
def get_results():
    metric_names = ["RMSE", "MAPE"]

    res.get_training_time()
    res.get_performance_metrics(metric_names)
    res.get_stats_method_wise_comparison(metric_names)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--data", action="store_true", help="Run data preprocessing"
    )
    parser.add_argument("-e", "--experiment", action="store_true", help="Run training")
    parser.add_argument("-r", "--results", action="store_true", help="Run training")
    args = parser.parse_args()

    ran = False
    if args.data:
        dp.process_raw_data()
        ran = True
    if args.experiment:
        run_experiment()
        ran = True
    if args.results:
        get_results()
        ran = True
    if not ran:
        parser.error("Select at least one: -d/--data or -t/--train -r/--results")
