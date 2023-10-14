import torch
import torch.nn as nn


class EnvFactorsEncoder(nn.Module):

    def __init__(self, original_dim, adjustment_dim):
        super().__init__()
        self.original_dim = original_dim
        self.adjustment_dim = adjustment_dim

        self.env_encoder =  nn.Sequential(
          nn.Linear(self.original_dim, 100),
          nn.ReLU(),
          nn.Linear(100, 100),
          nn.ReLU(),
          nn.Linear(100, self.adjustment_dim)
        )

    def forward(self, env_factors):
        return self.env_encoder(env_factors)


class AdaptiveEncoder(nn.Module):

    def __init__(self, state_dim, action_dim, adjustment_dim, num_prev_pairs=3):
        super().__init__()
        self.state_dim = state_dim
        self.actions_dim = action_dim
        self.num_prev_paris = num_prev_pairs
        self.adjustment_dim = adjustment_dim

        self.adaptive_model =  nn.Sequential(
          nn.Linear(num_prev_pairs * (self.state_dim + self.actions_dim), 100),
          nn.ReLU(),
          nn.Linear(100, 100),
          nn.ReLU(),
          nn.Linear(100, self.adjustment_dim)
        )

    def forward(self):
        pass

