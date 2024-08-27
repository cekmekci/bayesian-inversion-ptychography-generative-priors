import os
import torch
import random
import pickle
import numpy as np

from utils.utils_data import get_mnist_test_dataloader
from models import ComplexGenerator
from utils.utils_ptycho import cartesian_scan_pattern, create_disk_probe, ptycho_forward_op, ptycho_adjoint_op, free_space_tensor, calculate_overlap, rPIE
from samplers import optimize_latent_variable, ULA_Poisson_Sampler


# Fix the random seed
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# dataset settings
dataset_path = "./data/"
image_size = 64
offset = 0.2
# model settings
latent_dim = 100
generator_hidden_dim = 64
model_path = "./models/gen_mnist_model.pt"
# scan settings
step_size_scan = 7
perturbation_std_scan = 0.01
# probe settings
probe_amplitude = 10
probe_shape = (16, 16)
probe_width = 8
# latent variable initialization
z_lr = 1e-4
z_num_iter = 1000
z_verbose = False
# sampler settings
sampler_num_iter = 1000
sampler_burn_in = 500
sampler_step_size = 1e-5
# savings settings
output_dir = "./results/"
# rPIE settings
rpie_num_iter = 1000

# Get the test dataloader
test_dataloader = get_mnist_test_dataloader(dataset_path, batch_size = 1, image_size = image_size, offset = offset)

# Obtain the scan pattern
scan = cartesian_scan_pattern((image_size, image_size), probe_shape, step_size = step_size_scan, sigma = perturbation_std_scan)

# Obtain the probe
probe = create_disk_probe(size = probe_shape, width = probe_width, magnitude = probe_amplitude)

# Obtain the forward and adjoint operator
A = lambda x: ptycho_forward_op(x, scan, probe)
AH = lambda x: ptycho_adjoint_op(x, scan, probe, (image_size, image_size))

# Calculate the overlap ratio
overlap_ratio = calculate_overlap(probe, step_size_scan)

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
results_path = output_dir + "overlap_ratio_" + str(overlap_ratio) + "_probe_amplitude_" + str(probe_amplitude) + "/"
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Test the proposed method on complex test objects
for i , (gt_object, _) in enumerate(test_dataloader):
    print("Test example", i)
    # Obtain the intensity patterns
    farplane = A(gt_object) # (1,S,2,H2,W2)
    intensity = torch.sum(farplane**2, 2, keepdim = True) # (1,S,1,H2,W2)
    intensity = torch.poisson(intensity)

    # Obtain the complex generator that generates objects of interest
    complex_generator = ComplexGenerator(mag_latent_dim = latent_dim, mag_dim = generator_hidden_dim, phase_latent_dim = latent_dim, phase_dim = generator_hidden_dim)
    if torch.cuda.is_available():
        complex_generator = complex_generator.cuda()
        complex_generator.magnitude_generator.load_state_dict(torch.load(model_path))
        complex_generator.phase_generator.load_state_dict(torch.load(model_path))
    else:
        complex_generator.magnitude_generator.load_state_dict(torch.load(model_path), map_location = torch.device('cpu'))
        complex_generator.phase_generator.load_state_dict(torch.load(model_path), map_location = torch.device('cpu'))
    complex_generator.eval()

    # Initialize the latent variable
    free_space = free_space_tensor((image_size, image_size))
    z_init = optimize_latent_variable(complex_generator, free_space, lr = z_lr, num_steps = z_num_iter, verbose = z_verbose)

    # Get the sampler
    sampler = ULA_Poisson_Sampler(A, AH, complex_generator, intensity, z_init, num_iter = sampler_num_iter, step_size = sampler_step_size, burn_in = sampler_burn_in, use_cuda = torch.cuda.is_available())

    # Generate samples
    x_samples = sampler.generate_samples()

    # Save the results of our method
    with open(results_path + str(i) + "_ula_poisson_results.pkl", "wb") as fp:
        pickle.dump(x_samples, fp)

    # Run the rPIE algorithm
    rpie_result = rPIE(intensity, (image_size, image_size), scan, probe, rpie_num_iter)

    # Save the results of rPIE
    with open(results_path + str(i) + "_rpie_results.pkl", "wb") as fp:
        pickle.dump(rpie_result, fp)

    # Save the true object
    with open(results_path + str(i) + "_true_object.pkl", "wb") as fp:
        pickle.dump(gt_object, fp)

    # Only do this for 100 test examples
    if i == 99:
        break

# Save the probe and the scan pattern
with open(results_path + "probe.pkl", "wb") as fp:
    pickle.dump(probe, fp)
with open(results_path + "scan.pkl", "wb") as fp:
    pickle.dump(scan, fp)
