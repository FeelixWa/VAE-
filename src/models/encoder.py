import torch as tch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EncoderConv(nn.Module):
    """
    Encoder with convolutional layers.

    Args:
        latent_dim (int): Dimension of the latent space.
        activation (str): Activation function ('relu' or 'sigm').
        channels (int): Number of input channels.
        nb_conv_init (int): Initial number of convolutional filters.
        n_scale (int): Number of scales in the input (FOR PLANAR FLOW)

    Attributes:
        conv1, conv2, conv3, conv4, conv5, conv6 (nn.Module): Convolutional layers.
        bn1, bn2, bn3, bn4, bn5, bn6 (nn.Module): Batch normalization layers.
        flatten (nn.Module): Flatten layer.
        fc1, fc_mean, fc_log_var (nn.Module): Fully connected layers.
        activation (function): Activation function.
    """
    
    def __init__(self, latent_dim, activation='relu', channels=3, nb_conv_init=128, n_scale=1):
        super(EncoderConv, self).__init__()

        self.nb_conv = nb_conv_init
        self.activation = F.sigmoid if activation == 'sigm' else F.relu

        # Define convolutional layers
        self.conv1 = nn.Conv2d(channels * n_scale, self.nb_conv, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.nb_conv)
        self.conv2 = nn.Conv2d(self.nb_conv, self.nb_conv * 2, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(self.nb_conv * 2)
        self.conv3 = nn.Conv2d(self.nb_conv * 2, self.nb_conv * 4, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(self.nb_conv * 4)
        self.conv4 = nn.Conv2d(self.nb_conv * 4, self.nb_conv * 8, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(self.nb_conv * 8)
        self.conv5 = nn.Conv2d(self.nb_conv * 8, self.nb_conv * 16, 3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(self.nb_conv * 16)
        self.conv6 = nn.Conv2d(self.nb_conv * 16, self.nb_conv * 32, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(self.nb_conv * 32)

        # Flatten the output from the final convolutional layer
        flattened_size = 8 * 8 * self.nb_conv * 32
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flattened_size, self.nb_conv * 32)
        self.fc_mean = nn.Linear(self.nb_conv * 32, latent_dim)
        self.fc_log_var = nn.Linear(self.nb_conv * 32, latent_dim)

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Mean and log variance of the latent space.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.activation(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.activation(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation(x)
        
        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        
        return mean, log_var


class EncoderConvSparseKernel(nn.Module):
    """
    Encoder with sparse convolutional kernels.

    Args:
        latent_dim (int): Dimension of the latent space.
        activation (str): Activation function ('relu' or 'sigm').
        channels (int): Number of input channels.
        nb_conv_init (int): Initial number of convolutional filters.

    Attributes:
        conv1_sparse, conv2_sparse, conv3_sparse, conv4_sparse, conv5_sparse, conv6_sparse (nn.Module): Sparse convolutional layers.
        bn1, bn2, bn3, bn4, bn5, bn6 (nn.Module): Batch normalization layers.
        flatten (nn.Module): Flatten layer.
        fc1, fc_mean, fc_log_var (nn.Module): Fully connected layers.
        activation (function): Activation function.
    """
    
    def __init__(self, latent_dim, activation='relu', channels=3, nb_conv_init=128):
        super(EncoderConvSparseKernel, self).__init__()

        self.nb_conv = nb_conv_init
        self.activation = F.sigmoid if activation == 'sigm' else F.relu

        # Define sparse convolutional layers
        self.conv1_sparse = nn.Conv2d(channels, self.nb_conv, 11, padding=5)
        self.init_sparse_kernel(self.conv1_sparse)
        self.bn1 = nn.BatchNorm2d(self.nb_conv)

        self.conv2_sparse = nn.Conv2d(self.nb_conv, self.nb_conv * 2, 10, stride=2, padding=4)
        self.init_sparse_kernel(self.conv2_sparse)
        self.bn2 = nn.BatchNorm2d(self.nb_conv * 2)

        self.conv3_sparse = nn.Conv2d(self.nb_conv * 2, self.nb_conv * 4, 10, stride=2, padding=4)
        self.init_sparse_kernel(self.conv3_sparse)
        self.bn3 = nn.BatchNorm2d(self.nb_conv * 4)

        self.conv4_sparse = nn.Conv2d(self.nb_conv * 4, self.nb_conv * 8, 10, stride=2, padding=4)
        self.init_sparse_kernel(self.conv4_sparse)
        self.bn4 = nn.BatchNorm2d(self.nb_conv * 8)

        self.conv5_sparse = nn.Conv2d(self.nb_conv * 8, self.nb_conv * 16, 10, stride=2, padding=4)
        self.init_sparse_kernel(self.conv5_sparse)
        self.bn5 = nn.BatchNorm2d(self.nb_conv * 16)

        self.conv6_sparse = nn.Conv2d(self.nb_conv * 16, self.nb_conv * 32, 10, stride=2, padding=4)
        self.init_sparse_kernel(self.conv6_sparse)
        self.bn6 = nn.BatchNorm2d(self.nb_conv * 32)

        # Flatten the output from the final convolutional layer
        flattened_size = 8 * 8 * self.nb_conv * 32
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flattened_size, self.nb_conv * 32)
        self.fc_mean = nn.Linear(self.nb_conv * 32, latent_dim)
        self.fc_log_var = nn.Linear(self.nb_conv * 32, latent_dim)

        # Ensure zero weights remain zero
        self._freeze_zero_weights()

    def init_sparse_kernel(self, conv_layer):
        """
        Initialize the sparse kernel for a convolutional layer.

        Args:
            conv_layer (nn.Conv2d): Convolutional layer to initialize.
        """
        kernel_random_sparsity_per_kernel(conv_layer.weight) # you can use kernel_random_curve as well

    def _freeze_zero_weights(self):
        """
        Freeze the weights that are zero to ensure they do not update during training.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                with tch.no_grad():
                    zero_mask = (module.weight == 0).float().to('cuda')
                    module.weight.register_hook(lambda grad, mask=zero_mask: grad * (1 - mask))
                    module.weight.requires_grad = True

    def forward(self, x):
        """
        Forward pass through the sparse encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Mean and log variance of the latent space.
        """
        x = self.conv1_sparse(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2_sparse(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.conv3_sparse(x)
        x = self.bn3(x)
        x = self.activation(x)

        x = self.conv4_sparse(x)
        x = self.bn4(x)
        x = self.activation(x)

        x = self.conv5_sparse(x)
        x = self.bn5(x)
        x = self.activation(x)

        x = self.conv6_sparse(x)
        x = self.bn6(x)
        x = self.activation(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation(x)
        
        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        
        return mean, log_var


def kernel_random_sparsity_per_kernel(weight, sparsity=0.1):
    """
    Initialize the weight tensor with random values but maintain a specified sparsity per kernel.

    Args:
        weight (torch.Tensor): The weight tensor to initialize.
        sparsity (float): The proportion of non-zero elements per kernel (e.g., 0.1 for 10% sparsity).
    """
    with tch.no_grad():
        out_channels, in_channels, h, w = weight.shape
        kernel_size = h * w
        num_nonzero_per_kernel = int(kernel_size * sparsity)

        for out_channel in range(out_channels):
            for in_channel in range(in_channels):
                kernel = tch.zeros((h, w), device='cuda')
                indices = tch.randperm(kernel_size, device='cuda')[:num_nonzero_per_kernel]
                kernel.view(-1)[indices] = 1.0
                weight[out_channel, in_channel] = kernel

def kernel_random_curve(weight):
    """
    Initialize the weight tensor with a curve pattern.

    This function sets the weights of the convolutional kernels to follow a 
    curve pattern generated by a polynomial function. The curve is normalized 
    and scaled to fit within the kernel dimensions.

    Args:
        weight (torch.Tensor): The weight tensor of shape (out_channels, in_channels, height, width).
    """
    with tch.no_grad():
        # Ensure the weight tensor is zeroed out before initialization
        weight.zero_()
        out_channels, in_channels, h, w = weight.shape

        for out_channel in range(out_channels):
            # Generate x values for the curve
            x = np.linspace(-int(w / 2), int(w / 2), w)
            k = np.random.randint(2, 5)  # Number of polynomial coefficients
            coeffs = np.random.randn(k)  # Random polynomial coefficients
            y = np.polyval(coeffs, x)  # Evaluate polynomial function
            
            # Normalize and scale the curve to fit within the kernel
            y = (y - y.min()) / (y.max() - y.min()) * (h - 1)
            y = np.round(y).astype(int)  # Convert to integer indices

            # Convert numpy arrays to PyTorch tensors
            y = tch.tensor(y, dtype=tch.long, device=weight.device)

            # Set weights along the curve for each input channel
            for i in range(w):
                for in_channel in range(in_channels):
                    weight[out_channel, in_channel, y[i], i] = 1.0
