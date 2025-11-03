import os
import json
import random
import torch
from src.models.simple_lstm import SimpleLSTMModel
from src.models.cnn_lstm import CNNLSTMModel
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
EXPERIMENT_JSON = os.path.join(PROJECT_ROOT, "experiment.json")


with open(EXPERIMENT_JSON, "r") as f:
    EXPERIMENT = json.load(f)


def get_configuration():
    # Randomly select parameters from search space given in experiment.json
    p = {}
    space = EXPERIMENT["search_space"]

    p["cnn_channels"] = random.choices(
        space["cnn_channel"], k=int(random.choice(space["cnn_size"]))
    )
    p["kernel_size"] = int(random.choice(space["kernel_size"]))
    p["lstm_hidden_size"] = int(random.choice(space["neurons"]))
    p["lstm_layers"] = int(random.choice(space["lstm_layers"]))
    p["dense_size"] = int(random.choice(space["neurons"]))
    p["batch_size"] = int(random.choice(space["batch_size"]))
    p["dropout_cnn"] = float(random.choice(space["dropout"]))
    p["dropout_lstm"] = float(random.choice(space["dropout"]))
    p["dropout_fc"] = float(random.choice(space["dropout"]))
    p["learning_rate"] = float(random.choice(space["learning_rate"]))
    p["optimizer"] = random.choice(space["optimizer"])

    return p


def get_optimizer(name, model_params, lr=1e-3, **kwargs):
    # Returnes initialized optimizer object
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(model_params, lr=lr, **kwargs)
    elif name == "adamw":
        return torch.optim.AdamW(model_params, lr=lr, **kwargs)
    elif name == "adagrad":
        return torch.optim.Adagrad(model_params, lr=lr, **kwargs)
    elif name == "nadam":
        return torch.optim.NAdam(model_params, lr=lr, **kwargs)
    else:
        raise ValueError(f"❌ Unknown optimizer: '{name}'")


def get_model(model_name, horizon, device, p, input_shape):
    # Returns initialized model based on required architecture
    model = (
        SimpleLSTMModel(
            input_shape=input_shape,
            output_size=int(horizon),
            lstm_hidden_size=p["lstm_hidden_size"],
            lstm_layers=p["lstm_layers"],
            dense_size=p["dense_size"],
            dropout_lstm=p["dropout_lstm"],
            dropout_fc=p["dropout_fc"],
        )
        if model_name == "LSTM"
        else CNNLSTMModel(
            input_shape=input_shape,
            output_size=int(horizon),
            conv_channels=p["cnn_channels"],
            kernel_size=p["kernel_size"],
            lstm_hidden_size=p["lstm_hidden_size"],
            lstm_layers=p["lstm_layers"],
            dense_size=p["dense_size"],
            dropout_cnn=p["dropout_cnn"],
            dropout_fc=p["dropout_fc"],
        )
    )
    model.to(device)

    optimizer = get_optimizer(p["optimizer"], model.parameters(), p["learning_rate"])
    criterion = nn.L1Loss()

    return model, optimizer, criterion
