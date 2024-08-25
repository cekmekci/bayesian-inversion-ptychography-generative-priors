import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class Generator(nn.Module):
    """
    A PyTorch neural network module that defines a generator model for generating magnitude/phase images from a latent space.

    Args:
        latent_dim (int, optional): The dimensionality of the input latent space. Default is 100.
        dim (int, optional): A scaling factor for the number of convolutional filters in each layer. Default is 64.

    Methods:
        sample_latent(num_samples):
            Generates random samples from the latent space.

            Args:
                num_samples (int): The number of latent samples to generate.

            Returns:
                torch.Tensor: A tensor of shape (num_samples, latent_dim) containing random latent vectors.

        forward(input_data):
            Forward pass through the generator network to produce images from latent vectors.

            Args:
                input_data (torch.Tensor): A tensor of shape (batch_size, latent_dim) representing a batch
                                           of latent vectors.

            Returns:
                torch.Tensor: A tensor of shape (batch_size, 1, 64, 64) representing the generated images.
    """

    def __init__(self, latent_dim = 100, dim = 64):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.dim = dim
        self.latent_to_image = nn.Sequential(
            # layer 1 : the output is (dim*8, 4, 4)
            nn.ConvTranspose2d(in_channels = latent_dim, out_channels = dim * 8, kernel_size = 4, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(dim * 8),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True),
            # layer 2 : the output is (dim*4, 8, 8)
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels = dim * 8, out_channels = dim * 4, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(dim * 4),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True),
            # layer 3 : the output is (dim*2, 16, 16)
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels = dim * 4, out_channels = dim * 2, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(dim * 2),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True),
            # layer 4 : the output is (dim, 32, 32)
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels = dim * 2, out_channels = dim, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True),
            # layer 5 : the output is (1, 64, 64)
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels = dim, out_channels = 1, kernel_size = 3, stride = 1, padding = 1),
            nn.Sigmoid())

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))

    def forward(self, input_data):
        # input_data is (batch_size, latent_dim)
        # make it (batch_size, latent_dim, 1, 1)
        input_data = torch.unsqueeze(torch.unsqueeze(input_data, 2), 3)
        # Map latent to image
        x = self.latent_to_image(input_data)
        # Return generated image
        return x


class Discriminator(nn.Module):
    """
    A PyTorch neural network module that defines a discriminator model for distinguishing between real and
    generated magnitude/phase images.

    Args:
        dim (int, optional): A scaling factor for the number of convolutional filters in each layer. Default is 64.

    Methods:
        forward(input_data):
            Forward pass through the discriminator network to produce a scalar output for each input image.

            Args:
                input_data (torch.Tensor): A tensor of shape (batch_size, 1, 64, 64) representing a batch
                                           of input images.

            Returns:
                torch.Tensor: A tensor of shape (batch_size, 1) containing the discriminator's output for
                              each image in the batch. The output is typically interpreted as a probability
                              or score indicating whether the input is real or generated.
    """
    def __init__(self, dim = 64):
        super(Discriminator, self).__init__()
        self.dim = dim
        self.image_to_scalar = nn.Sequential(
            # layer 1 : the output is (dim,32,32)
            nn.Conv2d(in_channels = 1, out_channels = dim, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True),
            # layer 2 : the output is (dim*2,16,16)
            nn.Conv2d(in_channels = dim, out_channels = dim * 2, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(dim * 2),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True),
            # layer 3 : the output is (dim*4,8,8)
            nn.Conv2d(in_channels = dim * 2, out_channels = dim * 4, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(dim * 4),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True),
            # layer 4 : the output is (dim*8,4,4)
            nn.Conv2d(in_channels = dim * 4, out_channels = dim * 8, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(dim * 8),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True),
            # layer 5 : the output is (1,1,1)
            nn.Conv2d(in_channels = dim * 8, out_channels = 1, kernel_size = 4, stride = 1, padding = 0, bias = False),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True))

    def forward(self, input_data):
        x = self.image_to_scalar(input_data) # (B,1,1,1)
        x = torch.squeeze(x, dim=(2,3)) # (B,1)
        return x



class ComplexGenerator(nn.Module):
    """
    A PyTorch neural network module that defines a complex-valued generator model. The model generates
    complex images by separately generating magnitude and phase images and then combining them to produce
    real and imaginary components.

    Args:
        mag_latent_dim (int, optional): The dimensionality of the latent space for the magnitude generator.
                                        Default is 100.
        mag_dim (int, optional): A scaling factor for the number of convolutional filters in each layer of
                                 the magnitude generator. Default is 64.
        phase_latent_dim (int, optional): The dimensionality of the latent space for the phase generator.
                                          Default is 100.
        phase_dim (int, optional): A scaling factor for the number of convolutional filters in each layer of
                                   the phase generator. Default is 64.

    Methods:
        sample_latent(num_samples):
            Generates random samples from the combined latent space for both magnitude and phase generators.

            Args:
                num_samples (int): The number of latent samples to generate.

            Returns:
                torch.Tensor: A tensor of shape (num_samples, mag_latent_dim + phase_latent_dim) containing
                              random latent vectors.

        forward(input_data):
            Forward pass through the complex generator network to produce real and imaginary components of
            the generated complex image.

            Args:
                input_data (torch.Tensor): A tensor of shape (batch_size, mag_latent_dim + phase_latent_dim)
                                           representing a batch of latent vectors.

            Returns:
                torch.Tensor: A tensor of shape (batch_size, 2, 64, 64) containing the real and imaginary
                              components of the generated complex images. The real component is in channel 0,
                              and the imaginary component is in channel 1.
    """

    def __init__(self, mag_latent_dim = 100, mag_dim = 64, phase_latent_dim = 100, phase_dim = 64):
        super(ComplexGenerator, self).__init__()
        self.mag_latent_dim = mag_latent_dim
        self.phase_latent_dim = phase_latent_dim
        self.latent_dim = mag_latent_dim + phase_latent_dim
        self.magnitude_generator = Generator(latent_dim = mag_latent_dim, dim = mag_dim)
        self.phase_generator = Generator(latent_dim = phase_latent_dim, dim = phase_dim)

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))

    def forward(self, input_data):
        # Divide the latent variable into two parts
        latent_magnitude = input_data[:, :self.mag_latent_dim]
        latent_phase = input_data[:, self.mag_latent_dim:]
        # Obtain the magnitude and phase images
        out_magnitude = self.magnitude_generator(latent_magnitude)
        out_phase = self.phase_generator(latent_phase)
        # Convert them into real and imaginary images
        out_real = out_magnitude * torch.cos(out_phase)
        out_imag = out_magnitude * torch.sin(out_phase)
        # Concatenate them along the channel dimension
        out = torch.cat((out_real, out_imag), 1)
        return out


if __name__ == '__main__':

    # Test the generator implementation
    generator = Generator(latent_dim = 100, dim = 64)
    latent_vectors = generator.sample_latent(num_samples = 16)
    generated_images = generator(latent_vectors)
    print("Output of the magnitude/phase generator:", generated_images.shape)

    # Test the discriminator implementation
    discriminator = Discriminator(dim = 64)
    image_batch = torch.randn((16, 1, 64, 64))  # Example batch of images
    outputs = discriminator(image_batch)
    print("Output of the discriminator:", outputs.shape)

    # Test the discriminator implementation
    generator = ComplexGenerator(mag_latent_dim = 100, mag_dim = 64, phase_latent_dim = 100, phase_dim = 64)
    latent_vectors = generator.sample_latent(num_samples = 16)
    generated_images = generator(latent_vectors)
    print("Output of the complex generator:", generated_images.shape)
