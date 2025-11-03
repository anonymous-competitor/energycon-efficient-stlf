import torch.nn as nn
import torch.nn.functional as F


class CNNLSTMModel(nn.Module):
    def __init__(
        self,
        input_shape,
        output_size,
        conv_channels,
        kernel_size,
        lstm_hidden_size,
        lstm_layers,
        dense_size,
        dropout_cnn,
        dropout_fc,
    ):
        super(CNNLSTMModel, self).__init__()
        time_steps, num_features = input_shape

        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_channels = num_features
        for out_channels in conv_channels:
            self.convs.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            if time_steps >= 4:
                self.pools.append(nn.MaxPool1d(kernel_size=2))
                time_steps //= 2
            in_channels = out_channels

        self.dropout1 = nn.Dropout(dropout_cnn)

        self.lstm_stack = nn.ModuleList()
        for i in range(lstm_layers):
            in_size = in_channels if i == 0 else lstm_hidden_size
            self.lstm_stack.append(
                nn.LSTM(
                    input_size=in_size, hidden_size=lstm_hidden_size, batch_first=True
                )
            )

        self.dropout2 = nn.Dropout(dropout_fc)
        self.fc1 = nn.Linear(lstm_hidden_size, dense_size)
        self.fc2 = nn.Linear(dense_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, features, time)

        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x))
            if i < len(self.pools):
                x = self.pools[i](x)

        x = self.dropout1(x)
        x = x.permute(0, 2, 1)  # (batch, time, features) for LSTM

        for lstm in self.lstm_stack:
            x, _ = lstm(x)

        x = x[:, -1, :]  # last timestep
        x = self.dropout2(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
