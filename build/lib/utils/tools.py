import numpy as np
import matplotlib.pyplot as plt

def imshow(img1, img2, n_im=4):
    """
    Display a set of images from img1 and img2 in a grid.

    Args:
        img1 (Tensor): A tensor of images (batch_size, channels, height, width).
        img2 (Tensor): Another tensor of images with the same shape as img1.
        n_im (int): Number of images to display from each tensor.
    
    Returns:
        figure (matplotlib.figure.Figure): The created figure object.
    """
    # Convert tensors to numpy arrays and detach them from the computation graph
    npimg1 = img1.detach().cpu().numpy()
    npimg2 = img2.detach().cpu().numpy()

    # Select a subset of images (up to n_im)
    npimg1 = npimg1[:n_im]
    npimg2 = npimg2[:n_im]

    # Create subplots with 2 rows and n_im columns
    figure, axes = plt.subplots(2, n_im, figsize=(n_im * 3, 6))

    # Display images from img1 and img2
    for i in range(n_im):
        axes[0, i].imshow(np.transpose(npimg1[i], (1, 2, 0)))  # Transpose to (H, W, C) for RGB image
        axes[0, i].axis('off')  # Turn off axis

        axes[1, i].imshow(np.transpose(npimg2[i], (1, 2, 0)))  # Transpose to (H, W, C) for RGB image
        axes[1, i].axis('off')  # Turn off axis

    plt.tight_layout()  # Adjust layout

    return figure

def visualize_samples(real_samples, fake_samples, epoch):
    """
    Visualize a set of real and fake samples.

    Args:
        real_samples (Tensor): A tensor of real images (batch_size, channels, height, width).
        fake_samples (Tensor): A tensor of fake images with the same shape as real_samples.
        epoch (int): The epoch number to display in the title.
    """
    fig, axes = plt.subplots(2, 6, figsize=(16, 4))
    for i in range(min(6, real_samples.shape[0])):
        if real_samples.shape[1] == 1:  # Grayscale images
            axes[0, i].imshow(real_samples[i, 0].cpu().detach().numpy(), cmap='gray')
            axes[1, i].imshow(fake_samples[i, 0].cpu().detach().numpy(), cmap='gray')
        else:  # Color images
            axes[0, i].imshow(real_samples[i].cpu().detach().permute(1, 2, 0).numpy())
            axes[1, i].imshow(fake_samples[i].cpu().detach().permute(1, 2, 0).numpy())
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    plt.suptitle(f"Original vs. Fake Samples - Epoch {epoch}")
    plt.show()

def visualize_samples_pyramid(real_samples, fake_samples, real_pyramid, fake_pyramid, epoch):
    """
    Visualize real and fake samples with their corresponding pyramids.

    Args:
        real_samples (Tensor): A tensor of real images (batch_size, channels, height, width).
        fake_samples (Tensor): A tensor of fake images with the same shape as real_samples.
        real_pyramid (Tensor): A tensor of real image pyramids.
        fake_pyramid (Tensor): A tensor of fake image pyramids.
        epoch (int): The epoch number to display in the title.
    """
    n = real_pyramid.shape[2]
    fig, axes = plt.subplots(2, n + 1, figsize=(16, 4))
    if real_pyramid.shape[1] == 1:
        axes[0, 0].imshow(real_samples[0, 0].cpu().detach().numpy(), cmap='gray')
        axes[1, 0].imshow(fake_samples[0, 0].cpu().detach().numpy(), cmap='gray')
    else:
        axes[0, 0].imshow(real_samples[0].cpu().detach().permute(1, 2, 0).numpy())
        axes[1, 0].imshow(fake_samples[0].cpu().detach().permute(1, 2, 0).numpy())

    for i in range(n):
        if real_pyramid.shape[1] == 1:  # Grayscale images
            axes[0, i + 1].imshow(real_pyramid[0, :, i].cpu().detach().numpy(), cmap='gray')
            axes[1, i + 1].imshow(fake_pyramid[0, :, i].cpu().detach().numpy(), cmap='gray')
        else:  # Color images
            axes[0, i + 1].imshow(real_pyramid[0, :, i].cpu().detach().permute(1, 2, 0).numpy())
            axes[1, i + 1].imshow(fake_pyramid[0, :, i].cpu().detach().permute(1, 2, 0).numpy())
        axes[0, i + 1].axis('off')
        axes[1, i + 1].axis('off')
    plt.suptitle(f"Original vs. Fake Pyramid - Epoch {epoch}")
    plt.show()

def normalize(tensor,max=1.0):
    return ((tensor-tensor.min())) / (tensor.max()-tensor.min())*max
