import torch as tch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import sys
import signal

from models.encoder import EncoderConv
from models.decoder import DecoderConv
from models.vae import vae_flow_loss, VaeFlow
from utils.tools import imshow

def vae_pf_trainer(params):
    """
    Train a Variational Autoencoder (VAE) with flow-based architecture.

    Parameters:
    - params (dict): Dictionary containing training parameters and model configurations.
      Keys include:
        - 'device': Device for training (e.g., 'cuda' or 'cpu').
        - 'batch_size': Number of samples per batch.
        - 'latent_dim': Dimension of the latent space.
        - 'lr': Learning rate for the optimizer.
        - 'n_epochs': Number of epochs to train.
        - 'epoch_hurdle': Epoch at which beta parameter is updated.
        - 'be': Beta parameter for KL component in VAE loss.
        - 'ce': Parameter for the log-det jacobian loss component in VAE loss.
        - 'model_name': Name for saving the model.
        - 'dataset': Dataset used for training.
        - 'milestones': Epochs to reduce learning rate.
        - 'nb_conv_init': Number of initial convolutional layers.
        - 'nb_conv_fin': Number of final convolutional layers.
        - 'type': Type of model architecture ('conv', 'conv2', 'fcn', 'conv_dil', 'conv_dil2', 'conv_sparse_kernel').
        - 'K': Number of flow components.
        - 'pretrained_vae_path': Optional path to a pre-trained model.
    """

    # Extract parameters from the dictionary
    device = params['device']
    batch_size = params['batch_size']
    latent_dim = params['latent_dim']
    lr = params['lr']
    n_epochs = params['n_epochs']
    epoch_hurdle = params['epoch_hurdle']
    be_param = params['be']
    ce_param = params['ce']
    model_name = params['model_name']
    dataset = params['dataset']
    milestones = params['milestones']
    nb_conv_init = params['nb_conv_init']
    nb_conv_fin = params['nb_conv_fin']
    model_type = params['type']
    K = params.get('K', 10)

    # Initialize DataLoader 
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    if 'pretrained_vae_path' in params:
        # Load pre-trained model if path is provided
        model = tch.load(params['pretrained_vae_path']).to(device)
    else:
        # Define encoder and decoder based on the model type
        if model_type == 'conv':
            encoder = EncoderConv(latent_dim, nb_conv_init=nb_conv_init).to(device)
            decoder = DecoderConv(latent_dim, nb_conv_fin=nb_conv_fin).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create VAE model with encoder and decoder
        model = VaeFlow(encoder, decoder, device, K)

        # Print the architecture of the encoder
        print("Encoder Architecture:")
        for name, layer in encoder.named_modules():
            print(f"Layer: {name}, Type: {layer}")

        # Print the architecture of the decoder
        print("Decoder Architecture:")
        for name, layer in decoder.named_modules():
            print(f"Layer: {name}, Type: {layer}")

    # Initialize optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    model.train()

    def signal_handler(sig, frame):
        """
        Handle interrupt signals by saving the model and exiting.
        """
        folder_path = '/home/fwatine/python/TEXTURE/saved_models'
        response = input("Do you want to save the model before exiting? (y/n): ")
        if response.lower() == 'y':
            # Save the model and its state dict
            tch.save(model, os.path.join(folder_path, model_name + '_interrupted.pth'))
            tch.save(model.state_dict(), os.path.join(folder_path, model_name + '_dict_interrupted.pth'))
            print('Model saved. Exiting...')
        else:
            print('Exiting without saving the model...')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)


    # Initialize variables for tracking the best model
    best_loss = float('inf')
    best_model = None
    best_model_dict = None
    be = 0
    ce = ce_param

    # Initialize learning rate scheduler
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # Training loop
    for epoch in range(n_epochs):
        # Update beta parameter after reaching the specified epoch
        if epoch == epoch_hurdle:
            be = be_param

        # Initialize metrics for the epoch
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_ldj_loss = 0.0
        num_batches = 0

        # Iterate through the data loader
        for x in data_loader:
            x = x.to(device)

            # Forward pass
            x_hat, mu, log_var, SLDJ, z = model(x)
            
            # Compute the loss and other metrics
            loss_val, recon_loss, kl_loss, ldj_loss = vae_flow_loss(x, x_hat, mu, log_var, be, ce, SLDJ)

            # Backpropagation
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Accumulate losses
            epoch_loss += loss_val.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_ldj_loss += ldj_loss.item()
            num_batches += 1

        # Compute average loss for the epoch
        avg_loss = epoch_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_ldj_loss = epoch_ldj_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches

        # Print average losses
        print(f'Loss at epoch {epoch}: {avg_loss:.4f}')
        print(f'of which recon_loss: {avg_recon_loss:.4f}')
        print(f'of which ldj_loss: {avg_ldj_loss:.4f}')
        print(f'of which kl_loss: {avg_kl_loss:.4f}')

        # Step the learning rate scheduler
        scheduler.step()

        # Visualize samples
        if epoch % 5 == 0:
            imshow(x, x_hat)

        # Save model checkpoints
        if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
            folder_path = '/home/fwatine/python/TEXTURE/saved_models'
            tch.save(best_model, os.path.join(folder_path, model_name + '.pth'))
            tch.save(best_model.state_dict(), os.path.join(folder_path, model_name + '_dict.pth'))

    return avg_loss
