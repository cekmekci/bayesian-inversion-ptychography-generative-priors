import torch
import imageio
import numpy as np
import torch.nn as nn

from torchvision.utils import make_grid
from torch.autograd import grad as torch_grad


class Trainer():
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer,
                 gp_weight = 10, critic_iterations = 5, use_cuda = False):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def _critic_train_iteration(self, data):
        # Get generated data
        batch_size = data.shape[0]
        generated_data = self.sample_generator(batch_size) # G(z)
        # Calculate the output of the discriminator on real and generated data
        if self.use_cuda:
            data = data.cuda()
        d_real = self.D(data) # D(x)
        d_generated = self.D(generated_data) # D(G(z))
        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(data, generated_data)
        # Create total loss and optimize
        self.D_opt.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()
        self.D_opt.step()
        # Record loss
        return d_loss.item()

    def _generator_train_iteration(self, data):
        self.G_opt.zero_grad()
        # Get generated data
        batch_size = data.shape[0]
        generated_data = self.sample_generator(batch_size)
        # Calculate loss and optimize
        d_generated = self.D(generated_data) # D(G(z))
        g_loss = - d_generated.mean()
        g_loss.backward()
        self.G_opt.step()
        # Record loss
        return g_loss.item()

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.shape[0]
        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        if self.use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated.requires_grad = True
        if self.use_cuda:
            interpolated = interpolated.cuda()
        # Feed the interpolated examples to the discriminator
        D_interpolated = self.D(interpolated)
        # Calculate gradients of the output of the discriminator with respect to examples
        gradients = torch_grad(outputs=D_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(D_interpolated.size()).cuda() if self.use_cuda else torch.ones(
                               D_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]
        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        # Train loop
        for i, data in enumerate(data_loader):
            self.num_steps += 1
            # Train the discriminator (critic) and accumulate the loss
            d_loss = self._critic_train_iteration(data[0])
            epoch_d_loss = epoch_d_loss + d_loss
            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                g_loss = self._generator_train_iteration(data[0])
                epoch_g_loss = epoch_g_loss + g_loss
        # Average losses over the epoch
        num_batches = len(data_loader)
        epoch_d_loss /= num_batches
        epoch_g_loss /= (num_batches / self.critic_iterations)
        self.losses['D'].append(epoch_d_loss)
        self.losses['G'].append(epoch_g_loss)
        print(f"D Loss: {epoch_d_loss:.4f} | G Loss: {epoch_g_loss:.4f}")

    def train(self, data_loader, epochs, save_training_gif = True):
        if save_training_gif:
            # Fix latents to see how image generation improves during training
            fixed_latents = self.G.sample_latent(64)
            if self.use_cuda:
                fixed_latents = fixed_latents.cuda()
        for epoch in range(epochs):
            print("\nEpoch {}".format(epoch + 1))
            self._train_epoch(data_loader)
            if save_training_gif:
                # Generate batch of images and convert to grid
                images = self.G(fixed_latents).detach().cpu().data
                images = make_grid(images, normalize = True, scale_each = True)
                images = np.transpose(images.numpy(), (1, 2, 0)) * 255
                images = images.astype(np.uint8)
                imageio.imwrite('./training_{}_epoch.png'.format(epoch),
                    images)

    def sample_generator(self, num_samples):
        latent_samples = self.G.sample_latent(num_samples)
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        generated_data = self.G(latent_samples)
        return generated_data
