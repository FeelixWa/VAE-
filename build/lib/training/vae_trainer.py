import torch as tch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import signal
from torch.optim.lr_scheduler import MultiStepLR

from ..models.encoder import EncoderConv, EncoderConvSparseKernel
from ..models.decoder import DecoderConv
from ..models.vae import vae_loss, Vae
from ..utils.tools import imshow

def vae_trainer(params):
    """
    Train a Variational Autoencoder (VAE) with various possible architectures.

    Parameters:
    - params (dict): Dictionary containing the training parameters and model configurations.
      Keys include:
        - 'device': Device to run the training on (e.g., 'cuda' or 'cpu').
        - 'batch_size': Batch size for data loading.
        - 'latent_dim': Dimension of the latent space.
        - 'lr': Learning rate for the optimizer.
        - 'n_epochs': Number of epochs for training.
        - 'epoch_hurdle': Epoch number to change certain parameters (e.g., regularization).
        - 'be': Beta parameter for KL component in VAE loss.
        - 'model_name': Name of the model for saving.
        - 'dataset': Dataset to use for training.
        - 'milestones': Learning rate scheduler milestones.
        - 'nb_conv_init': Number of initial convolutional layers in encoder
        - 'nb_conv_fin': Number of final convolutional layers in decoder
        - 'type': Type of model architecture (e.g., 'conv', 'fcn').
    """
    # Extract parameters
    device = params['device']
    batch_size = params['batch_size']
    latent_dim = params['latent_dim']
    lr = params['lr']
    n_epochs = params['n_epochs']
    epoch_hurdle = params['epoch_hurdle']
    be_param = params['be']
    model_name = params['model_name']
    dataset = params['dataset']
    milestones = params['milestones']
    nb_conv_init = params['nb_conv_init']
    nb_conv_fin = params['nb_conv_fin']
    model_type = params['type']

    # Initialize DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    if 'pretrained_vae_path' in params:
        # Load pre-trained model if specified
        model = tch.load(params['pretrained_vae_path']).to(device)
    else:
        # Define model based on type
        if model_type == 'conv':
            encoder = EncoderConv(latent_dim, nb_conv_init=nb_conv_init).to(device)
            decoder = DecoderConv(latent_dim, nb_conv_fin=nb_conv_fin).to(device)
        elif model_type == 'conv_sparse_kernel':
            encoder = EncoderConvSparseKernel(latent_dim, nb_conv_init=nb_conv_init).to(device)
            decoder = DecoderConv(latent_dim, nb_conv_fin=nb_conv_fin).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create VAE model
        model = Vae(encoder, decoder, device)
        print("Encoder Architecture:")
        for name, layer in encoder.named_modules():
            print(f"Layer: {name}, Type: {layer}")
        print("Decoder Architecture:")
        for name, layer in decoder.named_modules():
            print(f"Layer: {name}, Type: {layer}")

    # Initialize optimizer and scheduler
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
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

    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    best_loss = float('inf')
    best_model = None

    # Training loop
    for epoch in range(n_epochs):
        
        # Initialize metrics for the epoch
        if epoch == epoch_hurdle:
            be = be_param
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        num_batches = 0

        for x in data_loader:
            x = x.to(device)

            # Forward pass
            x_hat, mu, log_var, z = model(x)

            # Reconstruction and KL loss
            loss_val, recon_loss, kl = vae_loss(x, x_hat, mu, log_var, be)
            
            
            # Backpropagation
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Accumulate losses
            epoch_loss += loss_val.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl.item()
            num_batches += 1

        # Compute average loss for the epoch
        avg_loss = epoch_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches

        print(f'Loss at epoch {epoch}: {avg_loss:.4f}')
        print(f'of which recon_loss: {avg_recon_loss:.4f}')
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
