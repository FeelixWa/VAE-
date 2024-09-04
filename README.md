# VAE Variations

This repository contains variations of Variational Autoencoders (VAEs) developed during my internship at MAP5, an applied mathematics research lab.

## Overview

- **Project Name**: VAE Variations
- **Context**: Internship at MAP5 (Applied Mathematics Research Lab)
- **Description**: This project includes several advanced VAE architectures and techniques, each exploring different aspects of Variational Autoencoders.

## Variants Included

1. **VAE with Planar Flow**
   - **Description**: This variant integrates planar normalizing flows into the VAE architecture to improve the flexibility of the latent space distribution, allowing the model to capture more complex data distributions.

2. **VAE-GAN**
   - **Description**: Combines VAE with Generative Adversarial Networks (GANs). The VAE generates samples, while a GAN's discriminator helps refine these samples for better quality and realism.

3. **VAE with Multiscale Pyramid Decomposition**
   - **Description**: Implements a multiscale pyramid approach to capture data features at different frequency resolutions. This technique improves the model's ability to handle high frequency components

4. **VAE with Large-Sparse Kernel**
   - **Description**: Features a large-sparse kernel architecture in the encoder, enhancing its capacity to model complex and high-dimensional data while maintaining computational efficiency.
