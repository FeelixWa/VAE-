import torch as tch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils import spectral_norm
import os
import sys
import signal

from models.discriminator import DConv
from utils.tools import visualize_samples

def vae_gan_trainer(params):
    """
    Train a Variational Autoencoder (VAE) and a Generative Adversarial Network (GAN) discriminator.

    # Note: A pretrained VAE model is required to use this function. Ensure that the 'pretrained_vae_path' parameter is provided.


    Parameters:
    - params (dict): Dictionary containing the training parameters and model configurations.
      Keys include:
        - 'device': Device to run the training on (e.g., 'cuda' or 'cpu').
        - 'batch_size': Number of samples per batch.
        - 'lr_vae': Learning rate for the VAE optimizer.
        - 'lr_disc': Learning rate for the discriminator optimizer.
        - 'n_epoch': Number of epochs for training.
        - 'epoch_hurdle': Epoch at which certain parameters are updated.
        - 'model_name': Name of the model for saving.
        - 'dataset': Dataset to use for training.
        - 'iter_disc': Number of iterations to train the discriminator.
        - 'iter_disc_spe': Number of iterations for special discriminator training.
        - 'gamma': Weight for the discriminator loss in the VAE loss.
        - 'n_epochs_vae_only': Number of epochs to train VAE only before training the GAN discriminator.
        - 'disc_training_only': Flag to indicate if only the discriminator should be trained.
        - 'be': Beta parameter for KL divergence term in VAE loss.
        - 'pretrained_vae_path': Path to a pretrained VAE model. This VAE should have been trained on a similar dataset or task to ensure effective performance.
        - 'pretrained_disc_path': Path to a pretrained GAN discriminator. This discriminator should have been trained to distinguish between real and generated samples, enhancing the VAE training process.
    """

    # Extract parameters from the dictionary
    device = params['device']
    batch_size = params['batch_size']
    lr_vae = params['lr_vae']
    lr_disc = params['lr_disc']
    n_epoch = params['n_epoch']
    epoch_hurdle = params['epoch_hurdle']
    model_name = params['model_name']
    dataset = params['dataset']
    iter_disc = params['iter_disc']
    iter_disc_spe = params['iter_disc_spe']
    n_epochs_vae_only = params.get('n_epochs_vae_only', 0)
    disc_training_only = params.get('disc_training_only', False)
    be = params.get('be', 1e-6)
    gamma = params.get('gamma', 1e-2)
    sigmoid = tch.nn.Sigmoid()

    print('Gamma parameter:', gamma)

    # Initialize DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load pretrained VAE model
    if 'pretrained_vae_path' in params:
        vae_net = tch.load(params['pretrained_vae_path']).to(device)
    else:
        raise ValueError('Pre-trained VAE model path must be provided.')

    # Initialize or load the Discriminator
    if 'pretrained_disc_path' in params:
        discriminator_net = tch.load(params['pretrained_disc_path']).to(device)
    else:
        discriminator_net = DConv().to(device)

    # Apply Spectral Normalization to the discriminator's layers
    apply_spectral_norm(discriminator_net)

    # Initialize optimizers
    optimizer_disc = optim.Adam(discriminator_net.parameters(), lr=lr_disc)
    optimizer_vae = optim.Adam(vae_net.parameters(), lr=lr_vae)

    def signal_handler(sig, frame):
        """
        Handle interrupts by saving the model and exiting gracefully.
        """
        folder_path = '/home/fwatine/python/TEXTURE/saved_models'
        response = input("Do you want to save the model before exiting? (y/n): ")
        if response.lower() == 'y':
            tch.save(vae_net, os.path.join(folder_path, model_name + '_interrupted.pth'))
            tch.save(vae_net.state_dict(), os.path.join(folder_path, model_name + '_dict_interrupted.pth'))
            print('Model saved. Exiting...')
        else:
            print('Exiting without saving the model...')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Determine whether VAE training should occur
    vae_training = not disc_training_only

    # Training loop
    for epoch in range(n_epoch):

        # Initialize metrics for the epoch
        proba_real, proba_fake = 0, 0
        loss_disc_epoch = 0
        loss_vae_epoch = 0.0
        loss_recon_epoch = 0.0
        loss_recon_d_epoch = 0.0
        nb_batches, nb_it_disc = 0, 0

        # Update beta parameter after hurdle epoch
        be_current = 0 if epoch <= epoch_hurdle else be

        # Adjust discriminator iterations based on epoch
        current_iter_disc = iter_disc_spe if (epoch == 0 or epoch % 25 == 0) else iter_disc

        discriminator_training = (epoch >= n_epochs_vae_only)

        for im in data_loader:
            x_real = im.to(device)

            ### Discriminator Training ###
            if discriminator_training:
                for _ in range(current_iter_disc):
                    optimizer_disc.zero_grad()

                    # Real samples
                    out_real = discriminator_net(x_real)[0]
                    loss_real = -out_real.mean()
                    loss_real.backward()

                    # Fake samples
                    x_fake, _, _, _ = vae_net(x_real)
                    out_fake = discriminator_net(x_fake.detach())[0]
                    loss_fake = out_fake.mean()
                    loss_fake.backward()

                    # Update discriminator
                    optimizer_disc.step()

                    loss_disc_epoch += loss_fake.item() + loss_real.item()
                    proba_real += sigmoid(out_real).mean().item()
                    proba_fake += sigmoid(out_fake).mean().item()
                    nb_it_disc += 1

            ### VAE Training ###
            if vae_training:
                optimizer_vae.zero_grad()
                x_fake, mu, log_var, _ = vae_net(x_real)

                # Discriminator loss
                out_fake = discriminator_net(x_fake)[0]
                disc_loss = -out_fake.mean()

                # Reconstruction loss
                recon_loss = F.mse_loss(x_real, x_fake, reduction='mean') if not disc_training_only else 0

                # KL divergence loss
                std = tch.exp(0.5 * log_var)
                kl_loss = tch.mean(tch.sum(mu**2 + std**2 - log_var, dim=1))

                # Total VAE loss
                loss_vae = recon_loss + kl_loss * be_current + disc_loss * gamma
                loss_vae.backward()
                optimizer_vae.step()

                # Accumulate losses
                loss_vae_epoch += loss_vae.item()
                loss_recon_epoch += recon_loss
                loss_recon_d_epoch += disc_loss.item() * gamma
                nb_batches += 1

        # Compute average losses for the epoch
        avg_loss_vae_epoch = loss_vae_epoch / max(nb_batches, 1)
        avg_loss_recon_epoch = loss_recon_epoch / max(nb_batches, 1)
        avg_loss_recon_d_epoch = loss_recon_d_epoch / max(nb_batches, 1)
        avg_loss_disc_epoch = loss_disc_epoch / max(nb_it_disc, 1)

        # Print and log statistics
        print(f'Epoch {epoch + 1}/{n_epoch}')
        print(f'  Loss Discriminator: {avg_loss_disc_epoch:.4f}')
        print(f'  Loss VAE: {avg_loss_vae_epoch:.4f}')
        print(f'    of which Recon: {avg_loss_recon_epoch:.4f}')
        print(f'    of which Recon Disc: {avg_loss_recon_d_epoch:.4f}')
        print(f'  Probability Real: {proba_real / max(nb_it_disc, 1):.4f}')
        print(f'  Probability Fake: {proba_fake / max(nb_it_disc, 1):.4f}')

        # Visualize samples every 5 epochs
        if epoch % 5 == 0:
            visualize_samples(x_real, x_fake, epoch)

        # Save model checkpoints every 10 epochs or at the last epoch
        if (epoch + 1) % 10 == 0 or epoch == n_epoch - 1:
            folder_path = '/home/fwatine/python/TEXTURE/saved_models'
            tch.save(vae_net, os.path.join(folder_path, model_name + '.pth'))
            tch.save(vae_net.state_dict(), os.path.join(folder_path, model_name + '_dict.pth'))
            tch.save(discriminator_net, os.path.join(folder_path, model_name + '_discriminator' + '.pth'))
            tch.save(discriminator_net.state_dict(), os.path.join(folder_path, model_name + '_discriminator_dict.pth'))

def apply_spectral_norm(module):
    """
    Apply spectral normalization to convolutional and linear layers.

    Parameters:
    - module (nn.Module): The neural network module to which spectral normalization should be applied.
    """
    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            try:
                spectral_norm(layer)
            except RuntimeError:
                # Spectral norm is already applied, so we do nothing
                continue