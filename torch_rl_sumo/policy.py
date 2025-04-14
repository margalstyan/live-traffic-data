import torch
import torch.nn as nn

class RoutePolicy(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.fc_mean = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(1))  # Learnable std

    def forward(self, x):
        mean = self.fc_mean(x)
        std = torch.exp(self.log_std)
        return mean.squeeze(), std.squeeze()
