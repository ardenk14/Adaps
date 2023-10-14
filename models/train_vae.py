import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from adaptive_part import EnvFactorsEncoder
from process_data import NormalizationTransform, process_stereo_data
from vae import StateVAE
from loss_functions import VAELoss

# Train the VAE

# TODO: Update these and use them
LR = 0.001
NUM_EPOCHS = 2000
BETA = 0.001
LATENT_DIM = 10
NUM_CHANNELS = 3#1
NUM_STEPS = 1
LATENT_DIM = 16

# TODO: Write the data loader for data you collect
train_loader, val_loader, norm_constants = process_stereo_data("../data_scene_flow/training/")
norm_tr = NormalizationTransform(norm_constants)

env_model = EnvFactorsEncoder(30, 5)
model = StateVAE(50)
loss_func = VAELoss(beta=BETA)
optimizer = optim.Adam(model.parameters(), lr=LR)
pbar = tqdm(range(NUM_EPOCHS))
train_losses = []
for epoch_i in pbar:
    train_loss_i = 0.
    # --- Your code here
    cnt = 0
    for batch_i, data in enumerate(train_loader):
      optimizer.zero_grad()

      #states = data['states']
      #targets = data['actions']
      img = data['img']
      env_vars = data['env_vars']
      #R_img = data['R_img']

      #reconstructed_states, mu, log_var, latent_state = vae_model(states)
      env = env_model(env_vars)
      reconstructed_states, mu, log_var, latent_state = model(img, env)

      loss = loss_func(reconstructed_states, img, mu, log_var)

      loss.backward()
      optimizer.step()

      train_loss_i += loss.item()
      cnt += 1

    train_loss_i /= cnt


    # ---
    train_loss_i += loss.item()
    pbar.set_description(f'Latent dim {LATENT_DIM} - Loss: {train_loss_i:.4f}')
    train_losses.append(train_loss_i)

losses = train_losses
#vaes = vae_model
# Evaluate:
#vae_model.eval()
#states_rec, mu, log_var, latent_state = vae_model(states)


# plot train loss and test loss:
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 3))
axes = [axes]
axes[0].plot(losses, label=f'latent_dim: {LATENT_DIM}')
axes[0].grid()
axes[0].legend()
axes[0].set_title('Train Loss')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Train Loss')
axes[0].set_yscale('log')
plt.show()


# ---

# save model:
torch.save(model.state_dict(), 'AdaptiveVAE.pt')
torch.save(env_model.state_dict(), 'EnvFactorsModel.pt')