import torch as tch
import torch.nn as nn
import torch.nn.functional as F

class Vae(nn.Module):
    """
    Variational Autoencoder (VAE) model.

    Args:
        Encoder (nn.Module): Encoder network.
        Decoder (nn.Module): Decoder network.
        device (torch.device): Device to run the model on.
    """
    def __init__(self, Encoder, Decoder, device):
        super(Vae, self).__init__()
        self.encoder = Encoder
        self.decoder = Decoder
        self.device = device

    def forward(self, x):
        mu, log_var = self.encoder(x)
        std = tch.exp(0.5 * log_var)
        z = tch.randn_like(std, device=self.device) * std + mu
        x_hat = self.decoder(z)
        return x_hat, mu, log_var, z

# Define VAEFlow which extends VAE with Planar Flow
class VaeFlow(nn.Module):
    """
    Variational Autoencoder (VAE) with normalizing flows.

    Args:
        Encoder (nn.Module): Encoder network.
        Decoder (nn.Module): Decoder network.
        device (torch.device): Device to run the model on.
        K (int): Number of flow steps.
    """
    def __init__(self, Encoder, Decoder, device, K):
        super(VaeFlow, self).__init__()
        self.encoder = Encoder
        self.decoder = Decoder
        self.device = device
        self.K = K

    def forward(self, x):
        SLDJ = tch.zeros(x.shape[0], 1).to(self.device)
        mu, log_var = self.encoder(x)
        std = tch.exp(0.5 * log_var)
        z_0 = tch.randn_like(std, device=self.device) * std + mu
        dim = z_0.shape[1]
        z = z_0
        for _ in range(self.K):
            planar_flow = PlanarFlow(dim)
            z, LDJ = planar_flow(z)
            SLDJ += LDJ
        x_hat = self.decoder(z)
        return x_hat, mu, log_var, SLDJ, z
    

class PlanarFlow(nn.Module):
    """
    Planar Flow layer for normalizing flows.

    Args:
        dim (int): Dimensionality of the input.
    """
    def __init__(self, dim):
        super(PlanarFlow, self).__init__()
        self.w = nn.Parameter(tch.randn(1, dim) * 0.01)
        self.b = nn.Parameter(tch.randn(1, 1) * 0.01)
        self.u = nn.Parameter(tch.randn(1, dim) * 0.01)

    def h(self, x):
        return tch.tanh(x)

    def h_prime(self, x):
        return 1 - tch.tanh(x) ** 2

    def forward(self, z):
        w = self.w.to(z.device)
        b = self.b.to(z.device)
        u = self.u.to(z.device)

        affine = z @ w.t() + b
        z_next = u * self.h(affine) + z
        psi = self.h_prime(affine) * w
        LDJ = -tch.log(tch.abs(psi @ u.t() + 1) + 1e-8)
        return z_next, LDJ


class VAEPyramid(nn.Module):
    """
    Variational Autoencoder (VAE) with steerable pyramid.

    Args:
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        steerable_pyramid (nn.Module): Steerable pyramid module.
        device (torch.device): Device to run the model on.
    """
    def __init__(self, encoder, decoder, steerable_pyramid, device):
        super(VAEPyramid, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.steerable_pyramid = steerable_pyramid
        self.device = device

    def forward(self, x):
        pyramid_output = self.steerable_pyramid(x).real
        b, c, s, h, w = pyramid_output.shape
        pyramid_output_redim = pyramid_output.view(b, -1, h, w)
        mu, log_var = self.encoder(pyramid_output_redim)
        std = tch.exp(0.5 * log_var)
        z = tch.randn_like(std, device=self.device) * std + mu

        # Decode
        pyramid_decoded = self.decoder(z)
        pyramid_decoded = pyramid_decoded.view(b, c, s, h, w)

        # Reconstruct image from steerable pyramid
        reconstructed_x = self.steerable_pyramid.recompose(pyramid_decoded)
        return pyramid_output, pyramid_decoded, reconstructed_x, mu, log_var, z


def vae_loss(x, x_hat, mu, log_var, be, loss_type='mse'):
    """
    Compute the loss for a VAE model.

    Args:
        x (torch.Tensor): Input tensor.
        x_hat (torch.Tensor): Reconstructed tensor.
        mu (torch.Tensor): Mean of the latent space.
        log_var (torch.Tensor): Log variance of the latent space.
        be (float): Weight for KL divergence term.
        loss_type (str): Loss function type ('mse' or 'bce').

    Returns:
        tuple: Total loss, reconstruction loss, and KL divergence loss.
    """
    recon_loss_fn = F.binary_cross_entropy if loss_type == 'bce' else F.mse_loss
    recon_loss = recon_loss_fn(x, x_hat, reduction='mean')
    std = tch.exp(0.5 * log_var)
    kl_loss = tch.mean(tch.sum(mu**2 + std**2 - log_var, dim=1))
    return recon_loss + be * kl_loss, recon_loss, kl_loss

def vae_flow_loss(x, x_hat, mu, log_var, be, ce, SLDJ, loss_type='mse'):
    """
    Compute the loss for a VAE with normalizing flows.

    Args:
        x (torch.Tensor): Input tensor.
        x_hat (torch.Tensor): Reconstructed tensor.
        mu (torch.Tensor): Mean of the latent space.
        log_var (torch.Tensor): Log variance of the latent space.
        be (float): Weight for KL divergence term.
        ce (float): Weight for log determinant of Jacobian term.
        SLDJ (torch.Tensor): Sum of log determinants of Jacobians.
        loss_type (str): Loss function type ('mse' or 'bce').

    Returns:
        tuple: Total loss, reconstruction loss, KL divergence loss, and log determinant of Jacobian loss.
    """
    recon_loss_fn = F.binary_cross_entropy if loss_type == 'bce' else F.mse_loss
    recon_loss = recon_loss_fn(x, x_hat, reduction='mean')
    std = tch.exp(0.5 * log_var)
    kl_loss = tch.mean(tch.sum(mu**2 + std**2 - log_var, dim=1))
    ldj_loss = tch.mean(-SLDJ)
    return recon_loss + be * kl_loss + ce * ldj_loss, recon_loss, kl_loss, ldj_loss
