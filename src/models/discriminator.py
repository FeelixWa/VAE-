import torch.nn as nn
import torch.nn.functional as F

class DConv(nn.Module):
    """
    A Convolutional Discriminator Network for use in Generative Adversarial Networks (GANs).

    This network takes an image as input and processes it through several convolutional layers,
    batch normalization layers, and fully connected layers to produce a single output score.

    Attributes:
        conv1, conv2, conv3, conv4 (nn.Module): Convolutional layers with increasing depth.
        bn1, bn2, bn3, bn4 (nn.Module): Batch normalization layers corresponding to each convolutional layer.
        flatten (nn.Module): Flattening layer to reshape tensor before fully connected layers.
        fc1, fc2, fc3, fc4, fc5 (nn.Module): Fully connected layers to process features and output a score.
        activation (function): Activation function used after each convolutional and fully connected layer.
    
    """

    def __init__(self):
        super(DConv, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, 6, padding=1)
        self.bn1 = nn.BatchNorm2d(256)     
        self.conv2 = nn.Conv2d(256, 512, 3, stride=2, padding=1)  
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 1024, 3, stride=2, padding=1)  
        self.bn3 = nn.BatchNorm2d(1024)
        self.conv4 = nn.Conv2d(1024, 2048, 3, stride=2, padding=1)  
        self.bn4 = nn.BatchNorm2d(2048)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4 * 4 * 2048, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 1)
        self.activation = F.relu

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W) where
                N is the batch size, C is the number of channels, H is the height,
                and W is the width of the image.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - The output score tensor from the final fully connected layer.
                - The flattened tensor before the final fully connected layers.
        """
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.activation(self.bn4(self.conv4(x)))

        x_l = self.flatten(x)

        x = self.activation(self.fc1(x_l))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.fc5(x)

        return x, x_l
