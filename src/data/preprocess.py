import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
files = os.listdir(DATA_RAW)

SPLIT_RATIO = (0.6, 0.1, 0.3)


def process_raw_data():
    print("Preprocessing data: it will take around 15s")

    for file in files:
        df = pd.read_csv(f"{DATA_RAW}/{file}", sep=";")
        df = df[["ts", "vrednost"]]

        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values("ts")

        # Fill out missing dates
        df_new = pd.DataFrame()
        df_new["ts"] = pd.date_range(
            start=df["ts"].iloc[0],
            end=df["ts"].iloc[-1],
            freq="15min",
            inclusive="both",
        )
        df_new = pd.merge(df_new, df, on="ts", how="outer")

        # Format decimal numbers
        df_new["vrednost"] = df_new["vrednost"].str.replace(",", ".").astype(float)

        df = df_new.copy()
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

        # Prepare columns for data filling
        df["month"] = df["ts"].dt.month
        df["dow"] = df["ts"].dt.dayofweek
        df["hour"] = df["ts"].dt.hour
        df["minute"] = df["ts"].dt.minute
        keys = ["month", "dow", "hour", "minute"]

        # Select most frequent value for month, day of the week, hour and minute and save it to column
        modes = (
            df.dropna(subset=["vrednost"])
            .groupby(keys, observed=True)["vrednost"]
            .agg(lambda s: s.value_counts().idxmax())
            .rename("value_mode")
            .reset_index()
        )
        out = df.merge(modes, on=keys, how="left")

        # Fill out missing values with most frequent value for month, day of the week, hour and minute
        out["vrednost"] = out["vrednost"].fillna(out["value_mode"])
        df = out.drop(columns=keys + ["value_mode"])

        # Aggregate to daily level
        daily = df.resample("D", on="ts")["vrednost"].sum().reset_index()

        # Round and rename columns
        daily["vrednost"] = daily["vrednost"].round(4)
        daily.rename(columns={"ts": "datetime", "vrednost": "load"}, inplace=True)
        daily["consumer_id"] = file.split(".")[0]

        daily.to_csv(f"{DATA_PROCESSED}/{file}", index=False)


def build_sequence(data, lookback, horizon):
    # Create sequence pairs
    X, y = [], []
    for i in range(lookback, len(data) - horizon + 1, horizon):
        x_seq = data[i - lookback : i]
        y_seq = data[i : i + horizon]
        X.append(x_seq)
        y.append(y_seq)

    return np.array(X), np.array(y).squeeze(-1)


def process_convential_training_data(
    data,
    lookback,
    horizon,
    batch_size,
):
    df = data[["load"]]

    # SPLIT DATA
    total_len = len(df)
    train_end = int(total_len * SPLIT_RATIO[0])
    val_end = train_end + int(total_len * SPLIT_RATIO[1])

    train_split = df.iloc[:train_end].copy()
    val_split = df.iloc[train_end:val_end].copy()
    test_split = df.iloc[val_end:].copy()

    # SCALE DATA
    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(train_split)
    val_scaled = scaler.transform(val_split)
    test_scaled = scaler.transform(test_split)

    # BUILD SEQUENCES
    X_train, y_train = build_sequence(train_scaled, lookback, horizon)
    X_val, y_val = build_sequence(val_scaled, lookback, horizon)
    X_test, y_test = build_sequence(test_scaled, lookback, horizon)

    # CREATE DATA LOADERS
    X_train = torch.tensor(X_train, dtype=torch.float32)  # (N, L, 1)
    y_train = torch.tensor(y_train, dtype=torch.float32)  # (N, H)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    id_train = torch.full((X_train.shape[0],), 0, dtype=torch.long)
    id_val = torch.full((X_val.shape[0],), 0, dtype=torch.long)
    id_test = torch.full((X_test.shape[0],), 0, dtype=torch.long)

    loaders = (
        DataLoader(
            TensorDataset(X_train, y_train, id_train),
            batch_size=batch_size,
            shuffle=False,
        ),
        DataLoader(
            TensorDataset(X_val, y_val, id_val), batch_size=batch_size, shuffle=False
        ),
        DataLoader(
            TensorDataset(X_test, y_test, id_test), batch_size=batch_size, shuffle=False
        ),
    )

    input_shape = X_train.shape[1:]

    return loaders, scaler, input_shape


def process_universal_training_data(lookback, horizon, batch_size):
    files = [f"{DATA_PROCESSED}/{f}" for f in os.listdir(DATA_PROCESSED)]
    dfs = [pd.read_csv(p) for p in files]

    # Select only consumers with above median lenghts (longest time series)
    lengths = np.array([len(df) for df in dfs])
    median_length = int(np.median(lengths))
    keep_idx = np.where(lengths >= median_length)[0].tolist()

    # Stack all consumers
    df = pd.concat(
        [
            d.assign(consumer_id=i)[["load", "consumer_id"]]
            for i, d in enumerate(dfs)
            if i in keep_idx
        ],
        ignore_index=True,
    )

    train_parts, val_parts, test_parts, scalers = [], [], [], {}

    for cid, df_c in df.groupby("consumer_id"):
        # DEFINE SPLITS
        n = len(df_c)
        tr = int(n * SPLIT_RATIO[0])
        va = tr + int(n * SPLIT_RATIO[1])

        # DEFINE SCALER
        scaler = MinMaxScaler().fit(df_c[["load"]].iloc[:tr])
        scalers[cid] = scaler

        # SPLIT DATA
        segments = {
            "train": df_c.iloc[:tr],
            "val": df_c.iloc[tr:va],
            "test": df_c.iloc[va:],
        }

        for split_name, seg in segments.items():
            # SCALE INDIVIDUAL CONSUMERS
            scaled = scaler.transform(seg[["load"]])

            # BUILD SEQUENCES
            X, y = build_sequence(scaled, lookback, horizon)
            ids = np.full(len(X), cid, dtype=np.int64)

            if split_name == "train":
                train_parts.append((X, y, ids))
            elif split_name == "val":
                val_parts.append((X, y, ids))
            else:
                test_parts.append((X, y, ids))

    def _stack(parts):
        X = torch.tensor(np.vstack([p[0] for p in parts]), dtype=torch.float32)
        y = torch.tensor(np.vstack([p[1] for p in parts]), dtype=torch.float32)
        i = torch.tensor(np.concatenate([p[2] for p in parts]), dtype=torch.long)
        return TensorDataset(X, y, i)

    ds_train = _stack(train_parts)
    ds_val = _stack(val_parts)
    ds_test = _stack(test_parts)

    loaders = (
        DataLoader(ds_train, batch_size=batch_size, shuffle=False),
        DataLoader(ds_val, batch_size=batch_size, shuffle=False),
        DataLoader(ds_test, batch_size=batch_size, shuffle=False),
    )
    sample_X = ds_train.tensors[0]
    input_shape = (sample_X.shape[1], sample_X.shape[2])

    return loaders, scalers, input_shape
