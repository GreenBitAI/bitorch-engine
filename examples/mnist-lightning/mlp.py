
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, hidden_features=1024, num_layers=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert 2 <= num_layers <= 3, "MLP currently only supports 2 or 3 layers."
        self.num_layers = num_layers
        self.flatten = nn.Flatten()

        self.first_linear = nn.Linear(28*28, hidden_features)
        self.act1 = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(num_features=hidden_features)

        self.linear1 = nn.Linear(hidden_features, hidden_features)
        self.act2 = nn.PReLU()
        self.bn2 = nn.BatchNorm1d(num_features=hidden_features)

        if num_layers > 2:
            self.linear2 = nn.Linear(hidden_features, hidden_features)
            self.act3 = nn.PReLU()
            self.bn3 = nn.BatchNorm1d(num_features=hidden_features)

        self.last_linear = nn.Linear(hidden_features, 10)

    def forward(self, x):
        x = self.flatten(x)

        x = self.first_linear(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.linear1(x)
        x = self.bn2(x)
        x = self.act2(x)

        if self.num_layers > 2:
            x = self.linear2(x)
            x = self.bn3(x)
            x = self.act3(x)

        return self.last_linear(x)


class SequentialMLP(nn.Module):
    def __init__(self, hidden_features = 2048, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten = nn.Flatten()
        self.first_linear = nn.Linear(28*28, hidden_features)
        self.body = nn.Sequential(
            nn.BatchNorm1d(num_features=hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.BatchNorm1d(num_features=hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.BatchNorm1d(num_features=hidden_features),
            nn.ReLU(),
        )
        self.last_linear = nn.Linear(hidden_features, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.first_linear(x)
        x = self.body(x)
        return self.last_linear(x)
