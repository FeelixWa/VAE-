import torch as tch
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import signal
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from ..models.encoder import EncoderConv
from ..models.decoder import DecoderConv
from ..models.vae import VaePyramid
from ..utils.steerable_pyramid import SteerablePyramid
from ..utils.tools import visualize_samples, visualize_samples_pyramid, normalize

def vae_pyramid_trainer(params):
    """
    Train a Variational Autoencoder (VAE) with a pyramid structure for multi-scale analysis.

    Parameters:
    - params (dict): Dictionary containing the training parameters and model configurations.
      Keys include:
        - 'device': Device to run the training on (e.g., 'cuda' or 'cpu').
        - 'latent_dim': Dimension of the latent space.
        - 'n_epochs': Number of epochs for training.
        - 'model_name': Name of the model for saving.
        - 'dataset': Dataset to use for training.
        - 'lr': Learning rate for the optimizer.
        - 'be': Beta parameter for KL component in VAE loss.
        - 'n_scale': Number of scales for the pyramid.
        - 'colors': Number of color channels.
        - 'size': Size of the input images.
        - 'nb_conv_init': Number of initial convolutional layers in encoder.
        - 'nb_conv_fin': Number of final convolutional layers in decoder.
        - 'fourier': Boolean flag for Fourier transform.
        - 'type': Type of model architecture (e.g., 'conv', 'conv_sparse_kernel').
        - 'batch_size': Batch size for data loading.
        - 'alpha': Weight for the image reconstruction loss during early epochs.
        - 'scale_weight': List of weights for each scale in the pyramid.
        - 'pretrained_vae_path': Optional path to load a pre-trained VAE model.
    """
    
    # Extract parameters
    device = params['device']
    latent_dim = params['latent_dim']
    n_epochs = params['n_epochs']
    model_name = params['model_name']
    dataset = params['dataset']
    lr = params.get('lr', 1e-4)
    be = params.get('be', 1e-4)
    n_scale = params.get('n_scale', 3)
    size = params.get('size', 256)
    nb_conv_init = params['nb_conv_init']
    nb_conv_fin = params['nb_conv_fin']
    model_type = params.get('type', 'conv')
    batch_size = params['batch_size']
    alpha = params['alpha']
    scale_weight = params.get('scale_weight', [1] * (n_scale + 2))

    # Initialize Steerable Pyramid
    steerable_pyramid = SteerablePyramid(
        n_i=size, n_j=size, n_scale=n_scale, n_ori=1,
        real=True, up_sampled=True, fourier=False, device=device
    ).to(device)

    # Initialize DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    if 'pretrained_vae_path' in params:
        # Load pre-trained VAE model if path is provided
        vae_net = tch.load(params['pretrained_vae_path']).to(device)
    else:
        # Define encoder and decoder based on model type
        if model_type == 'conv':
            encoder = EncoderConv(latent_dim, nb_conv_init=nb_conv_init, n_scale=(n_scale + 2)).to(device)
            decoder = DecoderConv(latent_dim, nb_conv_fin=nb_conv_fin, n_scale=(n_scale + 2)).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Create VAE model with pyramid
        model = VaePyramid(encoder, decoder, steerable_pyramid, device).to(device)

        # Print the architecture of the encoder
        print("Encoder Architecture:")
        for name, layer in encoder.named_modules():
            print(f"Layer: {name}, Type: {layer}")

        # Print the architecture of the decoder
        print("Decoder Architecture:")
        for name, layer in decoder.named_modules():
            print(f"Layer: {name}, Type: {layer}")

    # Initialize optimizer
    optimizer_vae = optim.Adam(model.parameters(), betas=(0.5, 0.999), lr=lr, weight_decay=0.0)
    model.train()

    def signal_handler(sig, frame):
        """
        Handle interrupts by saving the model and exiting gracefully.
        """
        folder_path = '/home/fwatine/python/TEXTURE/saved_models'
        response = input("Do you want to save the model before exiting? (y/n): ")
        if response.lower() == 'y':
            tch.save(model, os.path.join(folder_path, model_name + '_interrupted.pth'))
            tch.save(model.state_dict(), os.path.join(folder_path, model_name + '_dict_interrupted.pth'))
            print('Model saved. Exiting...')
        else:
            print('Exiting without saving the model...')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Training loop
    for epoch in range(n_epochs):

        # Initialize metrics for the epoch
        epoch_loss = 0.0
        epoch_reconstructed_image_loss = 0.0
        epoch_kl_loss = 0.0
        num_batches = 0
        epoch_pyramid_loss = {i: 0 for i in range(n_scale + 2)}

        for x_real in data_loader:
            x_real = x_real.to(device)

            optimizer_vae.zero_grad()

            # Forward pass
            pyramid_output, pyramid_decoded, reconstructed_x, mu, log_var, _ = model(x_real)
            
            # Reconstruction loss
            reconstructed_x = reconstructed_x.real
            reconstructed_image_loss = F.mse_loss(x_real, reconstructed_x, reduction='mean')
            recon_loss = alpha * reconstructed_image_loss
            epoch_reconstructed_image_loss += recon_loss

            # Pyramid loss
            pyramid_loss = {i: 0 for i in range(n_scale + 2)}
            for i in range(n_scale + 2):
                pyramid_loss[i] = F.mse_loss(pyramid_output[:, :, i, :, :], pyramid_decoded[:, :, i, :, :], reduction='mean')
                epoch_pyramid_loss[i] += pyramid_loss[i] * scale_weight[i]
                recon_loss += pyramid_loss[i] * scale_weight[i]

            # KL loss
            std = tch.exp(0.5 * log_var)
            kl_loss = tch.mean(tch.sum(mu**2 + std**2 - log_var, dim=1))

            # Total loss
            loss_vae = recon_loss + kl_loss * be
            
            # Backpropagation
            loss_vae.backward()
            optimizer_vae.step()

            # Accumulate losses
            epoch_loss += loss_vae.item()
            epoch_reconstructed_image_loss += reconstructed_image_loss.item()
            epoch_kl_loss += kl_loss.item()
            num_batches += 1

        # Compute average loss for the epoch
        avg_loss = epoch_loss / num_batches
        avg_reconstructed_image_loss = epoch_reconstructed_image_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches
        avg_pyramid_loss = [epoch_pyramid_loss[i] / num_batches for i in range(n_scale + 2)]

        print(f'Loss at epoch {epoch}: {avg_loss:.4f}')
        print(f'of which recon_loss: {avg_reconstructed_image_loss:.4f}')
        print(f'of which kl_loss: {avg_kl_loss:.4f}')
        for i in range(n_scale + 2):
            print(f"of which pyramid loss_scale {i}: {avg_pyramid_loss[i]}")


        # Visualize samples
        if epoch % 5 == 0:
            visualize_samples(x_real, normalize(reconstructed_x), epoch)
            visualize_samples_pyramid(x_real, normalize(reconstructed_x), normalize(pyramid_output), normalize(pyramid_decoded), epoch)

        # Save model checkpoints
        if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
            folder_path = '/home/fwatine/python/TEXTURE/saved_models'
            tch.save(model, os.path.join(folder_path, model_name + '.pth'))
            tch.save(model.state_dict(), os.path.join(folder_path, model_name + '_dict.pth'))

    return avg_loss
