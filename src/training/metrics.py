import torch
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)
from scipy.stats import spearmanr


@torch.no_grad()
def calculate_metrics(scaler, y_true, y_pred):
    # Metric calculation per y_true and y_pred on SINGLE consumer
    y_true = scaler.inverse_transform(y_true)
    y_pred = scaler.inverse_transform(y_pred)

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    mda = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff))
    rho, _ = spearmanr(y_true, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "MDA": mda,
        "Spearman": rho,
    }


@torch.no_grad()
def aggregate_metrics_per_consumer(targets, preds, ids, scalers):
    # Metric calculation per y_true and y_pred aggregated over ALL consumers

    targets, preds, ids = targets.numpy(), preds.numpy(), ids.numpy()
    # compute per-consumer metrics
    uids = np.unique(ids)
    per_consumer = []
    for cid in uids:
        m = ids == cid
        per_consumer.append(
            calculate_metrics(
                scalers[cid],
                targets[m],
                preds[m],
            )
        )

    # aggregate across consumers
    keys = list(per_consumer[0].keys())
    out = {}
    for k in keys:
        vals = [pc[k] for pc in per_consumer]
        arr = np.array(vals, dtype=float)
        # Spearman via Fisher z mean
        if k.lower().startswith("spearman"):
            z = np.arctanh(np.clip(arr, -0.999999, 0.999999))
            agg = float(np.tanh(z.mean()))
        else:
            agg = float(np.median(arr))
            disp = float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25))
        out[k] = agg
        out[f"{k}_IQR"] = disp
    return out


def calculate_aunl(losses):
    # Ensure inputs are numpy arrays
    losses = np.array(losses)

    n = len(losses)
    if n <= 1:
        return 1.0

    # AUNL for loss
    if np.all(losses == losses[0]):
        aunl = 1.0
    else:
        losses_min, losses_max = np.min(losses), np.max(losses)
        losses_scaled = (losses - losses_min) / (losses_max - losses_min)
        h = 1 / (n - 1)
        aunl = np.sum((losses_scaled[:-1] + losses_scaled[1:]) / 2) * h

    return aunl
