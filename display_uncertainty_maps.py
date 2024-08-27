import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Specify the overlap ratio and probe amplitude values
overlap_ratios = [0.045454545454545456, 0.09090909090909091, 0.22727272727272727]
probe_amplitudes = [10, 100]

# Index of the test example
test_sample_idx = "99"

# Dictionary containing phase uncertainty maps
results = dict()

for overlap_ratio in overlap_ratios:

    for probe_amplitude in probe_amplitudes:

        # Path of the folder containing the pickle files
        result_path = "./results/overlap_ratio_" + str(overlap_ratio) + "_probe_amplitude_" + str(probe_amplitude) + "/"

        # Calculate the phase uncertainty image for our method
        ula_poisson_results_path = result_path + test_sample_idx + "_ula_poisson_results.pkl"
        with open(ula_poisson_results_path, 'rb') as f:
            ula_samples = pickle.load(f)
            ula_samples = np.array(ula_samples)
            # Calculate the uncertainty for the phase images
            ula_phase_std = np.arctan2(ula_samples[:,1,:,:], ula_samples[:,0,:,:] + 1e-5)
            ula_phase_std = np.std(ula_phase_std, 0)

        # Store the phase uncertainty map in the dictionary
        results[(overlap_ratio, probe_amplitude)] = ula_phase_std

# Find the global min and max across all images
all_images = np.array([image for image in results.values()])
vmin = np.min(all_images)
vmax = np.max(all_images)

# Extract unique probe_amplitudes and overlap_ratios
probe_amplitudes = sorted(set(key[1] for key in results.keys()))
overlap_ratios = sorted(set(key[0] for key in results.keys()))

# Create a figure with subplots
fig, axes = plt.subplots(len(probe_amplitudes), len(overlap_ratios), figsize=(len(overlap_ratios) * 3, len(probe_amplitudes) * 2.5))

# Plot each image in the corresponding subplot
for i, probe_amplitude in enumerate(probe_amplitudes):
    for j, overlap_ratio in enumerate(overlap_ratios):
        image = results.get((overlap_ratio, probe_amplitude))
        if image is not None:
            im = axes[i,j].imshow(image, cmap = "jet", vmin = vmin, vmax = vmax)
        axes[i, j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        axes[i, j].spines['top'].set_visible(False)
        axes[i, j].spines['right'].set_visible(False)
        axes[i, j].spines['left'].set_visible(False)
        axes[i, j].spines['bottom'].set_visible(False)
        if i == 0:
            axes[i,j].set_title(f'Overlap Ratio: {overlap_ratio:.2f}', fontsize = 10)
    axes[i,0].set_ylabel(f'Probe Amplitude: {probe_amplitude}', fontsize = 10)
plt.tight_layout()
fig.colorbar(im, ax = axes)
plt.savefig("example_uncertainty_maps.pdf", format='pdf', bbox_inches = 'tight', pad_inches = 0)
plt.close()
