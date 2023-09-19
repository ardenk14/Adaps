import torch
import torch.nn as nn
import torch.nn.functional as F

class VAELoss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta  # Weight of the KL divergence term

    def forward(self, x_hat, x, mu, logvar):
        """
        Compute the VAE loss.
        vae_loss = MSE(x, x_hat) + beta * KL(N(\mu, \sigma), N(0, 1))
        where KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param x: <torch.tensor> ground truth tensor of shape (batch_size, state_size)
        :param x_hat: <torch.tensor> reconstructed tensor of shape (batch_size, state_size)
        :param mu: <torch.tensor> of shape (batch_size, state_size)
        :param logvar: <torch.tensor> of shape (batch_size, state_size)
        :return: <torch.tensor> scalar loss
        """
        mse_loss = F.mse_loss(x_hat.float(), x.float(), reduction='mean')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        loss = mse_loss + self.beta * kl_divergence
        return loss


class MultiStepLoss(nn.Module):
    def __init__(self, state_loss_fn, latent_loss_fn, alpha=0.1):
        super().__init__()
        self.state_loss = state_loss_fn
        self.latent_loss = latent_loss_fn
        self.alpha = alpha

    def forward(self, model, states, actions):
        """
        Compute the multi-step loss resultant of multi-querying the model from (state, action) and comparing the predictions with targets.
        Here the loss is computed based on 3 terms:
         - reconstruction loss: enforces good encoding and reconstruction of the states (no dynamics).
         - latent loss: enforces dynamics on latent space by matching the state encodings with the dynamics on latent space.
         - prediction loss: enforces reconstruction of the predicted states by matching the predictions with the targets.

         :param model: <nn.Module> model to be trained.
         :param states: <torch.tensor> tensor of shape (batch_size, traj_size + 1, state_size)
         :param actions: <torch.tensor> tensor of shape (batch_size, traj_size, action_size)
        """
        # compute reconstruction loss -- compares the encoded-decoded states with the original states
        rec_loss = 0.
        latent_values = model.encode(states)
        decoded_state = model.decode(latent_values)
        rec_loss += self.state_loss(decoded_state, states)

        # propagate dynamics on latent space as well as reconstructed states
        pred_latent_values = []
        pred_states = []
        prev_z = latent_values[:, 0, :]  # get initial latent value
        prev_state = states[:, 0, :]  # get initial state value
        for t in range(actions.shape[1]):
            next_z = None
            next_state = None
            next_z = model.latent_dynamics(prev_z, actions[:, t])
            pred_latent_values.append(next_z)
            next_state = model(prev_state, actions[:, t]) #model.decode(next_z)
            pred_states.append(next_state)

            prev_z = next_z
            prev_state = next_state
        pred_states = torch.stack(pred_states, dim=1)
        pred_latent_values = torch.stack(pred_latent_values, dim=1)
        # compute prediction loss -- compares predicted state values with the given states
        pred_loss = self.state_loss(pred_states, states[:, 1:])# / actions.shape[1]

        # compute latent loss -- compares predicted latent values with the encoded latent values for states
        lat_loss = self.latent_loss(pred_latent_values, model.encode(states[:, :-1]))# / actions.shape[1]

        multi_step_loss = rec_loss + pred_loss + self.alpha * lat_loss

        return multi_step_loss