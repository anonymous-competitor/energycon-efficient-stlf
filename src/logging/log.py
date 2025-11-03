import os
import csv
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOSS_LOG = os.path.join(LOG_DIR, "loss.csv")
METRIC_LOG = os.path.join(LOG_DIR, "metrics.csv")
DETAILS_LOG = os.path.join(LOG_DIR, "details.csv")
PRED_DATA = os.path.join(LOG_DIR, "data.csv")


DETAILS_COLUMNS = [
    "model",
    "type",
    "source",
    "experiment",
    "trial",
    "dataset",
    "horizon",
    "lookback",
    "early_stopping",
    "device",
    "batch_size",
    "learning_rate",
    "optimizer",
    "dense_size",
    "dropout_fc",
    "dropout_lstm",
    "lstm_hidden_size",
    "lstm_layers",
    "cnn_channels",
    "dropout_cnn",
    "kernel_size",
]

METRIC_COLUMNS = [
    "model",
    "horizon",
    "experiment",
    "type",
    "source",
    "epoch",
    "inference",
    "MAE",
    "MSE",
    "RMSE",
    "MAPE",
    "R2",
    "MDA",
    "Spearman",
    "MAE_IQR",
    "MSE_IQR",
    "RMSE_IQR",
    "MAPE_IQR",
    "R2_IQR",
    "MDA_IQR",
]


LOSS_COLUMNS = [
    "model",
    "epoch",
    "train_loss",
    "val_loss",
    "start_epoch_time",
    "end_epoch_time",
    "train_inference",
    "val_inference",
]

PRED_DATA_COLUMNS = ["model", "y_true", "y_pred"]


def create_logs_files():
    os.makedirs(LOG_DIR, exist_ok=True)

    with open(LOSS_LOG, "w", newline="") as f:
        csv.writer(f).writerow(LOSS_COLUMNS)

    with open(METRIC_LOG, "w", newline="") as f:
        csv.writer(f).writerow(METRIC_COLUMNS)

    with open(DETAILS_LOG, "w", newline="") as f:
        csv.writer(f).writerow(DETAILS_COLUMNS)

    with open(PRED_DATA, "w", newline="") as f:
        csv.writer(f).writerow(PRED_DATA_COLUMNS)


def log_entry(path, columns, entry: dict):
    df = pd.DataFrame([entry], columns=columns)
    df.to_csv(path, mode="a", index=False, header=not os.path.exists(path))


def get_model_config(model_name):
    details = pd.read_csv(DETAILS_LOG)
    row = details.loc[details["model"] == model_name]
    if row.empty:
        raise ValueError(f"No config for model={model_name}")
    return row.iloc[0].to_dict()
