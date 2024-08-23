import torch as tch
import torch.nn as nn
import torch.nn.functional as F

class DecoderConv(nn.Module):
    """
    Decoder with transposed convolutional layers.

    Args:
        latent_dim (int): Dimension of the latent space.
        activation (str): Activation function ('relu' or 'sigm').
        channels (int): Number of output channels.
        nb_conv_fin (int): Number of convolutional filters in the final layer.
        n_scale (int): Number of scales in the input (FOR PLANAR FLOW).

    Attributes:
        fc1, fc2 (nn.Module): Fully connected layers to reshape the latent vector.
        deconv1, deconv2, deconv3, deconv4, deconv5, deconv6 (nn.Module): Transposed convolutional layers.
        bn1, bn2, bn3, bn4, bn5 (nn.Module): Batch normalization layers.
        activation (function): Activation function

    """
    
    def __init__(self, latent_dim, activation='relu', channels=3, nb_conv_fin=128, n_scale=1):
        super(DecoderConv, self).__init__()
        
        # Set initial parameters
        self.latent_dim = latent_dim
        self.nb_conv = nb_conv_fin
        self.activation = F.sigmoid if activation == 'sigm' else F.relu

        # Fully connected layers to reshape the latent vector
        self.fc1 = nn.Linear(latent_dim, self.nb_conv * 32)
        self.fc2 = nn.Linear(self.nb_conv * 32, 8 * 8 * self.nb_conv * 32)

        # Define transposed convolutional layers
        self.deconv1 = nn.ConvTranspose2d(self.nb_conv * 32, self.nb_conv * 16, 3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(self.nb_conv * 16)
        
        self.deconv2 = nn.ConvTranspose2d(self.nb_conv * 16, self.nb_conv * 8, 3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(self.nb_conv * 8)
        
        self.deconv3 = nn.ConvTranspose2d(self.nb_conv * 8, self.nb_conv * 4, 3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(self.nb_conv * 4)
        
        self.deconv4 = nn.ConvTranspose2d(self.nb_conv * 4, self.nb_conv * 2, 3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(self.nb_conv * 2)
        
        self.deconv5 = nn.ConvTranspose2d(self.nb_conv * 2, self.nb_conv, 3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(self.nb_conv)

        self.deconv6 = nn.ConvTranspose2d(self.nb_conv, channels * n_scale, 3, stride=1, padding=1, output_padding=0)

    def forward(self, x):
        """
        Forward pass through the decoder.

        Args:
            x (torch.Tensor): Input tensor (latent vector).

        Returns:
            torch.Tensor: Reconstructed image.
        """
        # Reshape latent vector and apply fully connected layers
        x = self.fc1(x)
        x = self.activation(x)
        
        x = self.fc2(x)
        x = self.activation(x)
        
        # Reshape to prepare for transposed convolutions
        x = x.view(-1, self.nb_conv * 32, 8, 8)
        
        # Apply transposed convolutions with batch normalization and activation
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        
        x = self.deconv4(x)
        x = self.bn4(x)
        x = self.activation(x)

        x = self.deconv5(x)
        x = self.bn5(x)
        x = self.activation(x)
        
        # Final transposed convolution to return to original image size
        x = self.deconv6(x)
        
        # Use sigmoid activation to constrain output to [0, 1] range
        x = F.sigmoid(x)
        
        return x
