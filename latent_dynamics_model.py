import torch
import torch.nn as nn

from vae import StateEncoder, StateDecoder

class LatentDynamicsModel(nn.Module):
    """
    Model the dynamics in latent space via residual learning z_{t+1} = z_{t} + f(z_{t},a_{t})
    Use StateEncoder and StateDecoder encoding-decoding the state into latent space.
    where
        z_{t}  = encoder(x_{t})
        z_{t+1} = z_{t} + latent_dynamics_model(z_{t}, a_{t})
        x_{t+1} = decoder(z_{t+1})

    Latent dynamics model must be a Linear 3-layer network with 100 units in each layer and ReLU activations.
    The input to the latent_dynamics_model must be the latent states and actions concatentated along the last dimension.
    """

    def __init__(self, latent_dim, action_dim, num_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.num_channels = num_channels

        self.latent_dynamics_model =  nn.Sequential(
          nn.Linear(self.latent_dim + self.action_dim, 100),
          nn.ReLU(),
          nn.Linear(100, 100),
          nn.ReLU(),
          nn.Linear(100, self.latent_dim)
        )
        
        self.encoder = StateEncoder(latent_dim, num_channels = num_channels)
        self.decoder = StateDecoder(latent_dim, num_channels = num_channels)


        # ---

    def forward(self, state, action):
        """
        Compute next_state resultant of applying the provided action to provided state
        :param state: torch tensor of shape (..., num_channels, 32, 32)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_state: torch tensor of shape (..., num_channels, 32, 32)
        """
        state = self.encoder(state)
        inpt = torch.cat((state, action), dim=-1)
        next_state = self.latent_dynamics_model(inpt) + state
        next_state = self.decoder(next_state)

        return next_state

    def encode(self, state):
        """
        Encode a state into the latent space
        :param state: torch tensor of shape (..., num_channels, 32, 32)
        :return: latent_state: torch tensor of shape (..., latent_dim)
        """
        latent_state = self.encoder(state)
        return latent_state

    def decode(self, latent_state):
        """
        Decode a latent state into the original space.
        :param latent_state: torch tensor of shape (..., latent_dim)
        :return: state: torch tensor of shape (..., num_channels, 32, 32)
        """
        state = self.decoder(latent_state)
        return state

    def latent_dynamics(self, latent_state, action):
        """
        Compute the dynamics in latent space
        z_{t+1} = z_{t} + latent_dynamics_model(z_{t}, a_{t})
        :param latent_state: torch tensor of shape (..., latent_dim)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_latent_state: torch tensor of shape (..., latent_dim)
        """
        inpt = torch.cat((latent_state, action), dim=-1)
        next_latent_state = self.latent_dynamics_model(inpt) + latent_state
        return next_latent_state