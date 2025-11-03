import src.logging.log as log
import torch
import torch.nn as nn
import pandas as pd
import numpy as np


# Class to implement AUNL algorithm
class ScoreTracker:
    def __init__(self):
        self.aunl = float("inf")
        self.metric = float("inf")

    # Returns (aunl_improved, metric_improved)
    def update_aunl(self, aunl):
        if aunl < self.aunl:
            self.aunl = aunl

    def update_metric(self, metric):
        if metric < self.metric:
            self.metric = metric


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    # Model training per epoch
    model.train()
    total_loss = 0

    all_preds, all_targets, all_ids = [], [], []

    for i, batch in enumerate(train_loader):
        x, y, ids = batch
        x, y = x.to(device), y.to(device)
        y_pred = model(x)

        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        all_preds.append(y_pred.detach().cpu())
        all_targets.append(y.detach().cpu())
        all_ids.append(ids.detach().cpu())

        print(
            f"🟦 [Train] Batch {i + 1}/{len(train_loader)} - Loss: {loss.item():.4f}",
            end="\r",
        )

    out_ids = torch.cat(all_ids)
    print()  # For newline after loop
    return (
        total_loss / len(train_loader.dataset),
        torch.cat(all_preds),
        torch.cat(all_targets),
        out_ids,
    )


def predict(model, data_loader, criterion, device):
    # Predict and return unscaled values
    model.eval()
    total_loss = 0
    all_preds, all_targets, all_ids = [], [], []

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            x, y, id = batch
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            loss = criterion(y_pred, y)

            total_loss += loss.item() * y.size(0)
            all_preds.append(y_pred.cpu())
            all_targets.append(y.cpu())
            all_ids.append(id.cpu())

            print(
                f"🟨 [Eval ] Batch {i + 1}/{len(data_loader)} - Loss: {loss.item():.4f}",
                end="\r",
            )

    y_pred_flat = torch.cat(all_preds, dim=0)
    y_true_flat = torch.cat(all_targets, dim=0)
    out_ids = torch.cat(all_ids, dim=0)

    print()
    return (total_loss / len(data_loader.dataset), y_pred_flat, y_true_flat, out_ids)


def select_universal_model(horizon, rep):
    metrics_df = pd.read_csv(log.METRIC_LOG)
    # pick models only under correct horizon and experiment based on validation data
    f = metrics_df[
        metrics_df["model"].str.contains("universal", na=False)
        & (metrics_df["type"] == "val")
        & (metrics_df["experiment"] == rep)
        & (metrics_df["horizon"] == horizon)
    ]

    idx = f.groupby("model", dropna=False)["epoch"].idxmax()
    metrics_df = f.loc[idx].copy()

    def norm_minmax(s, asc=True):
        s = pd.Series(s)  # ensure Series
        a, b = s.min(), s.max()
        if not np.isfinite(a) or not np.isfinite(b):
            z = pd.Series(np.nan, index=s.index)
        elif b > a:
            z = (s - a) / (b - a)
        else:
            z = pd.Series(0.0, index=s.index)  # constant column -> all zeros
        return z if asc else (1.0 - z)

    def col(name, default=0.0):
        return (
            metrics_df[name]
            if name in metrics_df.columns
            else pd.Series(default, index=metrics_df.index)
        )

    Z = pd.DataFrame(
        {
            "MAE": norm_minmax(col("MAE"), asc=True),
            "MAE_IQR": norm_minmax(col("MAE_IQR"), asc=True),
            "Spearman": norm_minmax(col("Spearman"), asc=False),
            "MDA": norm_minmax(col("MDA"), asc=False),
            "R2": norm_minmax(col("R2"), asc=False),
        },
        index=metrics_df.index,
    )

    # Model picking score criteria:
    metrics_df["score"] = (
        1.00 * Z["MAE"]  # lowest MAE
        + 1.00 * Z["MAE_IQR"]  # lowest MAE variability
        + 0.75 * Z["Spearman"]  # trend and directional correctness
        + 0.50 * Z["MDA"]  # direction accuracy
        + 0.50 * Z["R2"]  # variance fidelity
    )

    best_row = metrics_df.sort_values(by="score", ascending=True).iloc[0]
    return best_row


def freeze_all_except_lstm(model):
    # freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # unfreeze all LSTM layers
    for m in model.modules():
        if isinstance(m, nn.LSTM):
            for p in m.parameters():
                p.requires_grad = True

    return model
