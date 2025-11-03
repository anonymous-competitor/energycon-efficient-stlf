import src.data.preprocess as data_process
import src.training.config as config
import src.training.utils as util
import src.training.metrics as m
import src.logging.log as log

import torch.nn as nn
import pandas as pd
import datetime
import time
import torch
import json
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODELS = os.path.join(PROJECT_ROOT, "models")


def conventional_training(model_type, rep):
    files = [
        f"{data_process.DATA_PROCESSED}/{f}"
        for f in os.listdir(data_process.DATA_PROCESSED)
    ]
    dfs = [pd.read_csv(p) for p in files]

    experiment = config.EXPERIMENT
    epochs = experiment["epochs"]
    device = experiment["device"]

    # For each consumer and for each horizon do N trials by using random search
    for df in dfs:
        for horizon in experiment["horizon"]:
            for i in range(experiment["trials"]):
                date_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                model_name = f"{model_type}_t{i}_h{horizon}_id{date_str}.pt"
                p = config.get_configuration()  # Parameter selection

                # Log parameter combination along with other meta-data
                log.log_entry(
                    log.DETAILS_LOG,
                    log.DETAILS_COLUMNS,
                    {
                        "model": model_name,
                        "type": model_type,
                        "source": None,
                        "experiment": rep,
                        "trial": i,
                        "dataset": df["consumer_id"].iloc[0],
                        "horizon": horizon,
                        "lookback": experiment["lookback"],
                        "early_stopping": False,
                        "device": experiment["device"],
                        **p,
                    },
                )

                # DATA PREPARATION
                data_loader, scaler, input_shape = (
                    data_process.process_convential_training_data(
                        df,
                        experiment["lookback"],
                        horizon,
                        p["batch_size"],
                    )
                )

                # MODEL INITIALIZATION
                model, optimizer, criterion = config.get_model(
                    model_type, horizon, device, p, input_shape
                )

                (train_loader, val_loader, test_loader) = data_loader
                train_losses, val_losses = [], []

                # For each epoch train model and log loss along metrics
                for epoch in range(epochs):

                    # TRAINING
                    strain = time.time()
                    train_loss, train_preds, train_targets, _ = util.train_one_epoch(
                        model, train_loader, criterion, optimizer, device
                    )
                    etrain = time.time()

                    # VALIDATION
                    svl = time.time()
                    val_loss, val_preds, val_targets, _ = util.predict(
                        model, val_loader, criterion, device
                    )
                    evl = time.time()

                    train_losses.append(train_loss)
                    val_losses.append(val_loss)

                    print(
                        f"📈 Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                    )

                    # === Logging LOSS ===
                    log.log_entry(
                        log.LOSS_LOG,
                        log.LOSS_COLUMNS,
                        {
                            "model": model_name,
                            "epoch": epoch + 1,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "start_epoch_time": datetime.datetime.fromtimestamp(
                                strain
                            ).strftime("%Y-%m-%d %H:%M:%S"),
                            "end_epoch_time": datetime.datetime.fromtimestamp(
                                etrain
                            ).strftime("%Y-%m-%d %H:%M:%S"),
                            "train_inference": round(etrain - strain, 3),
                            "val_inference": round(evl - svl, 3),
                        },
                    )

                    # === Logging TRAIN METRICS ===
                    metrics = m.calculate_metrics(
                        scaler,
                        train_targets.numpy(),
                        train_preds.numpy(),
                    )
                    log.log_entry(
                        log.METRIC_LOG,
                        log.METRIC_COLUMNS,
                        {
                            "epoch": epoch + 1,
                            "model": model_name,
                            "source": None,
                            "horizon": horizon,
                            "experiment": rep,
                            "type": "train",
                            "inference": round(etrain - strain, 3),
                            **metrics,
                        },
                    )

                    # === Logging VAL METRICS ===
                    val_metrics = m.calculate_metrics(
                        scaler,
                        val_targets.numpy(),
                        val_preds.numpy(),
                    )
                    log.log_entry(
                        log.METRIC_LOG,
                        log.METRIC_COLUMNS,
                        {
                            "epoch": epoch + 1,
                            "model": model_name,
                            "source": None,
                            "horizon": horizon,
                            "experiment": rep,
                            "type": "val",
                            "inference": round(evl - svl, 3),
                            **val_metrics,
                        },
                    )

                # TESTING
                sts = time.time()
                _, test_preds, test_targets, _ = util.predict(
                    model, test_loader, criterion, device
                )
                ets = time.time()

                # === Logging TEST METRICS ===
                test_metrics = m.calculate_metrics(
                    scaler, test_targets.numpy(), test_preds.numpy()
                )
                log.log_entry(
                    log.METRIC_LOG,
                    log.METRIC_COLUMNS,
                    {
                        "epoch": epoch + 1,
                        "model": model_name,
                        "source": None,
                        "horizon": horizon,
                        "experiment": rep,
                        "type": "test",
                        "inference": round(ets - sts, 3),
                        **test_metrics,
                    },
                )

                # === Logging data used in test evaluation ===
                log.log_entry(
                    log.PRED_DATA,
                    log.PRED_DATA_COLUMNS,
                    {
                        "model": model_name,
                        "y_true": json.dumps(
                            scaler.inverse_transform(test_targets.numpy()).tolist(),
                            sort_keys=True,
                        ),
                        "y_pred": json.dumps(
                            scaler.inverse_transform(test_preds.numpy()).tolist(),
                            sort_keys=True,
                        ),
                    },
                )

                # Saving model
                torch.save(model, os.path.join(MODELS, model_name))
                print(f"💾 Saved model {model_name}")


def universal_model_training(rep):
    tracker = util.ScoreTracker()
    tracking_metric = "MSE"
    experiment = config.EXPERIMENT

    epochs = experiment["epochs"]
    device = experiment["device"]
    min_epochs = experiment["min_epochs"]

    # For each horizon do N trials over all consumers
    for horizon in experiment["horizon"]:
        for i in range(experiment["trials"]):
            date_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            model_name = f"universal_t{i}_h{horizon}_id{date_str}.pt"
            p = config.get_configuration()  # parameter configuration

            # Log parameter combination along with other meta-data
            log.log_entry(
                log.DETAILS_LOG,
                log.DETAILS_COLUMNS,
                {
                    "model": model_name,
                    "type": "universal",
                    "source": None,
                    "experiment": rep,
                    "trial": i,
                    "dataset": None,
                    "horizon": horizon,
                    "lookback": experiment["lookback"],
                    "early_stopping": False,
                    "device": experiment["device"],
                    **p,
                },
            )

            # DATA PREPARATION - all consumers
            loaders, scalers, input_shape = (
                data_process.process_universal_training_data(
                    experiment["lookback"],
                    horizon,
                    p["batch_size"],
                )
            )

            # MODEL INITIALIZATION
            model, optimizer, criterion = config.get_model(
                "CNNLSTM", horizon, device, p, input_shape
            )
            (train_loader, val_loader, _) = loaders
            train_losses, val_losses = [], []

            # For each epoch train model and log loss along metrics, terminate based on AUNL
            for epoch in range(epochs):

                # TRAINING
                strain = time.time()
                train_loss, train_preds, train_targets, train_ids = (
                    util.train_one_epoch(
                        model, train_loader, criterion, optimizer, device
                    )
                )
                etrain = time.time()

                # VALIDATION
                svl = time.time()
                val_loss, val_preds, val_targets, val_ids = util.predict(
                    model, val_loader, criterion, device
                )
                evl = time.time()

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(
                    f"📈 Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

                # === Logging LOSS ===
                log.log_entry(
                    log.LOSS_LOG,
                    log.LOSS_COLUMNS,
                    {
                        "model": model_name,
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "start_epoch_time": datetime.datetime.fromtimestamp(
                            strain
                        ).strftime("%Y-%m-%d %H:%M:%S"),
                        "end_epoch_time": datetime.datetime.fromtimestamp(
                            etrain
                        ).strftime("%Y-%m-%d %H:%M:%S"),
                        "train_inference": round(etrain - strain, 3),
                        "val_inference": round(evl - svl, 3),
                    },
                )

                # === Logging TRAIN METRICS ===
                metrics = m.aggregate_metrics_per_consumer(
                    train_targets,
                    train_preds,
                    train_ids,
                    scalers,
                )
                log.log_entry(
                    log.METRIC_LOG,
                    log.METRIC_COLUMNS,
                    {
                        "epoch": epoch + 1,
                        "model": model_name,
                        "source": None,
                        "horizon": horizon,
                        "experiment": rep,
                        "type": "train",
                        "inference": round(etrain - strain, 3),
                        **metrics,
                    },
                )

                # === Logging VAL METRICS ===
                val_metrics = m.aggregate_metrics_per_consumer(
                    val_targets,
                    val_preds,
                    val_ids,
                    scalers,
                )
                log.log_entry(
                    log.METRIC_LOG,
                    log.METRIC_COLUMNS,
                    {
                        "epoch": epoch + 1,
                        "model": model_name,
                        "source": None,
                        "horizon": horizon,
                        "experiment": rep,
                        "type": "val",
                        "inference": round(evl - svl, 3),
                        **val_metrics,
                    },
                )

                # === Early stopping based on AUNL ===
                metric = val_metrics[tracking_metric]
                aunl = m.calculate_aunl(val_losses)

                if epoch > min_epochs:
                    if aunl > tracker.aunl:
                        print(f"⚠️ No improvement in AUNL ({aunl}/{tracker.aunl})")
                        print("🛑 Early stopping.")
                        break

                if metric < tracker.metric:
                    tracker.update_aunl(aunl)
                    tracker.update_metric(metric)

            # Saving model
            torch.save(model, os.path.join(MODELS, model_name))
            print(f"💾 Saved model {model_name}")


def fine_tuning(rep):
    files = [
        f"{data_process.DATA_PROCESSED}/{f}"
        for f in os.listdir(data_process.DATA_PROCESSED)
    ]
    dfs = [pd.read_csv(p) for p in files]

    learning_rate = 0.0001

    # For each horizon fine-tune universal model to all conusmers
    experiment = config.EXPERIMENT
    for df in dfs:
        for horizon in experiment["horizon"]:
            # Pick the best universal model based on metric score from validation metrics for given horizon and experiment
            best_row = util.select_universal_model(horizon, rep)
            p = log.get_model_config(
                best_row["model"]
            )  # get chosen model configuration

            date_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            model_name = (
                f"fine_tuned_{df['consumer_id'].iloc[0]}_h{horizon}_id{date_str}.pt"
            )

            # Log parameter combination along with other meta-data
            log.log_entry(
                log.DETAILS_LOG,
                log.DETAILS_COLUMNS,
                {
                    **p,
                    "model": model_name,
                    "type": "fine_tuned",
                    "source": best_row["model"],
                    "experiment": rep,
                    "trial": None,
                    "dataset": df["consumer_id"].iloc[0],
                    "horizon": horizon,
                    "lookback": experiment["lookback"],
                    "early_stopping": True,
                    "device": experiment["device"],
                },
            )

            # DATA PREPARATION - single consumer
            data_loader, scaler, _ = data_process.process_convential_training_data(
                df,
                experiment["lookback"],
                horizon,
                p["batch_size"],
            )

            # MODEL LOADING AND COMPONENT FREEZING
            model = torch.load(f"{MODELS}/{best_row['model']}")
            model = util.freeze_all_except_lstm(model)
            optimizer = config.get_optimizer(
                p["optimizer"], model.parameters(), learning_rate
            )
            criterion = nn.L1Loss()

            epochs = experiment["epochs"]
            device = experiment["device"]
            min_epochs = experiment["min_epochs"]
            experiment_patience = experiment["patience"]
            (train_loader, val_loader, test_loader) = data_loader
            train_losses, val_losses = [], []

            best_val_loss = float("inf")
            patience = 0
            # For each epoch train model and log loss along metrics
            for epoch in range(epochs):

                # TRAINING
                strain = time.time()
                train_loss, train_preds, train_targets, _ = util.train_one_epoch(
                    model, train_loader, criterion, optimizer, device
                )
                etrain = time.time()

                # VALIDATION
                svl = time.time()
                val_loss, val_preds, val_targets, _ = util.predict(
                    model, val_loader, criterion, device
                )
                evl = time.time()

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(
                    f"📈 Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

                # === Logging LOSS ===
                log.log_entry(
                    log.LOSS_LOG,
                    log.LOSS_COLUMNS,
                    {
                        "model": model_name,
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "start_epoch_time": datetime.datetime.fromtimestamp(
                            strain
                        ).strftime("%Y-%m-%d %H:%M:%S"),
                        "end_epoch_time": datetime.datetime.fromtimestamp(
                            etrain
                        ).strftime("%Y-%m-%d %H:%M:%S"),
                        "train_inference": round(etrain - strain, 3),
                        "val_inference": round(evl - svl, 3),
                    },
                )

                # === Logging TRAIN METRICS ===
                metrics = m.calculate_metrics(
                    scaler,
                    train_targets.numpy(),
                    train_preds.numpy(),
                )
                log.log_entry(
                    log.METRIC_LOG,
                    log.METRIC_COLUMNS,
                    {
                        "epoch": epoch + 1,
                        "model": model_name,
                        "source": None,
                        "horizon": horizon,
                        "experiment": rep,
                        "type": "train",
                        "inference": round(etrain - strain, 3),
                        **metrics,
                    },
                )

                # === Logging VAL METRICS ===
                val_metrics = m.calculate_metrics(
                    scaler,
                    val_targets.numpy(),
                    val_preds.numpy(),
                )
                log.log_entry(
                    log.METRIC_LOG,
                    log.METRIC_COLUMNS,
                    {
                        "epoch": epoch + 1,
                        "model": model_name,
                        "source": None,
                        "horizon": horizon,
                        "experiment": rep,
                        "type": "val",
                        "inference": round(evl - svl, 3),
                        **val_metrics,
                    },
                )

                # === Early stopping based on convential early stopping ===
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                    torch.save(
                        model, os.path.join(MODELS, model_name)
                    )  # SAVE ONLY GOOD MODELS
                    print("✅ Model improved and was saved.")
                else:
                    patience += 1
                    print(
                        f"⏳ No improvement. Patience: {patience}/{experiment_patience}"
                    )
                    if epoch > min_epochs:
                        if patience >= experiment_patience:
                            print("🛑 Early stopping triggered.")
                            break

            # TESTING
            sts = time.time()
            _, test_preds, test_targets, _ = util.predict(
                model, test_loader, criterion, device
            )
            ets = time.time()

            # === Logging TEST METRICS ===
            test_metrics = m.calculate_metrics(
                scaler,
                test_targets.numpy(),
                test_preds.numpy(),
            )
            log.log_entry(
                log.METRIC_LOG,
                log.METRIC_COLUMNS,
                {
                    "epoch": epoch + 1,
                    "model": model_name,
                    "source": None,
                    "horizon": horizon,
                    "experiment": rep,
                    "type": "test",
                    "inference": round(ets - sts, 3),
                    **test_metrics,
                },
            )

            # === Logging data used in test evaluation ===
            log.log_entry(
                log.PRED_DATA,
                log.PRED_DATA_COLUMNS,
                {
                    "model": model_name,
                    "y_true": json.dumps(
                        scaler.inverse_transform(test_targets.numpy()).tolist(),
                        sort_keys=True,
                    ),
                    "y_pred": json.dumps(
                        scaler.inverse_transform(test_preds.numpy()).tolist(),
                        sort_keys=True,
                    ),
                },
            )
