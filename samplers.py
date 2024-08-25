import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


class ULA_Poisson_Sampler():
    """
    Implements a Poisson Unadjusted Langevin Algorithm (ULA) sampler for generating samples
    from a posterior distribution given a set of measurements and a generative model.

    The ULA sampler performs gradient-based sampling to generate latent variables `z` from
    a posterior distribution p(z|y), where `y` is the observed measurement. The posterior
    distribution is based on a Poisson likelihood. The sampler can generate multiple samples
    of the latent variable `z`, which can be used to reconstruct samples of the observed variable `x`.

    Parameters:
    -----------
    A : function
        The forward operator that models the measurement process, taking as input a generated sample
        and returning the corresponding measurement.

    AT : function
        The adjoint operator corresponding to `A`.

    generator : torch.nn.Module
        The generative model that maps a latent variable `z` to the data space `x`.

    measurement : torch.Tensor
        The observed measurement data `y`.

    z_init : torch.Tensor
        The initial value of the latent variable `z` from which the sampling process starts.

    num_iter : int, optional
        The number of iterations to run the ULA sampler. Default is 10000.

    step_size : float, optional
        The step size for the gradient update in the ULA sampler. Default is 1e-5.

    burn_in : int, optional
        The number of initial iterations to discard (burn-in period) before collecting samples.
        Default is 5000.

    use_cuda : bool, optional
        If True, the computation will be performed on a CUDA-enabled GPU. Default is False.

    Attributes:
    -----------
    eps : float
        A small constant added for numerical stability in gradient computation.

    Methods:
    --------
    grad_log_p_z_given_y(z):
        Computes the gradient of the log posterior distribution p(z|y) with respect to `z`.

    sample():
        Generates a list of samples of the latent variable `z` after the burn-in period using ULA.

    generate_samples():
        Generates a list of samples in the data space `x` by passing the latent variable samples
        through the generative model.

    Returns:
    --------
    z_samples : list of torch.Tensor
        A list of sampled latent variables `z` after the burn-in period.

    x_samples : list of np.ndarray
        A list of generated samples in the data space `x` corresponding to the sampled latent variables.
    """
    def __init__(self, A, AT, generator, measurement, z_init, num_iter = 10000, step_size = 1e-5, burn_in = 5000, use_cuda = False):
        self.A = A
        self.AT = AT
        self.generator = generator
        self.measurement = measurement
        self.z_init = z_init
        self.num_iter = num_iter
        self.step_size = step_size
        self.burn_in = burn_in
        self.use_cuda = use_cuda
        self.eps = 1e-5
        if self.use_cuda:
            self.generator = self.generator.cuda()
            self.z_init = self.z_init.cuda()
            self.measurement = self.measurement.cuda()

    def grad_log_p_z_given_y(self, z):
        # gradient of log_p_y_given_z
        AGz = self.A(self.generator(z)) # (1,S,2,H2,W2)
        vec = self.AT(AGz * (self.measurement / (torch.sum(AGz**2,2,keepdim = True) + self.eps)  - 1)) # (1,2,64,64)
        _, vjp = torch.autograd.functional.vjp(self.generator, z, v = vec, create_graph = False, strict = True) # (1, latent_dim)
        grad = 2 * vjp
        # add the gradient of log_p_z, which is -z
        grad = grad - z
        return grad

    def sample(self):
        z = self.z_init
        z_samples = []
        for k in range(self.num_iter):
            # Compute the gradient of log p(z|y)
            grad = self.grad_log_p_z_given_y(z)
            # Determine the candidate
            z_new = z + self.step_size * grad + (2 * self.step_size)**0.5 * torch.randn_like(z)
            # Transition
            z = z_new
            # If we pass the burn-in period, collect the sample.
            if k > self.burn_in:
                z_samples.append(z.detach().cpu())
        return z_samples

    def generate_samples(self):
        z_samples = self.sample()
        x_samples = []
        for z_sample in z_samples:
            if self.use_cuda:
                z_sample = z_sample.cuda()
            x_sample = self.generator(z_sample).detach().cpu().numpy()[0,:,:,:]
            x_samples.append(x_sample)
        return x_samples


def optimize_latent_variable(G, x, lr = 1e-4, num_steps = 1000, verbose = True):
    """

    """
    # Ensure x is on the same device as G
    device = next(G.parameters()).device
    x = x.to(device)
    # Initialize the latent variable z with requires_grad=True
    z = torch.randn(1, G.latent_dim, device = device, requires_grad = True)
    # Define the optimizer
    optimizer = optim.Adam([z], lr = lr)
    # Define the loss function
    loss_fn = nn.MSELoss()
    for step in range(num_steps):
        optimizer.zero_grad()
        # Generate image from z
        generated_image = G(z)
        # Compute the loss
        loss = loss_fn(generated_image, x)
        # Backpropagate the loss
        loss.backward()
        # Update z
        optimizer.step()
        if verbose and (step % 100 == 0):
            print(f'Step {step}/{num_steps}, Loss: {loss.item()}')
    return z.detach()
