import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Specify the overlap ratio and probe amplitude values
overlap_ratio = 0.36363636363636365
probe_amplitude = 100

# Path of the folder containing the pickle files
result_path = "./results/overlap_ratio_" + str(overlap_ratio) + "_probe_amplitude_" + str(probe_amplitude) + "/"

# Index of the test example
test_sample_idx = "0"

# Calculate the summary images for our method
ula_poisson_results_path = result_path + test_sample_idx + "_ula_poisson_results.pkl"
with open(ula_poisson_results_path, 'rb') as f:
    ula_samples = pickle.load(f)
    ula_samples = np.array(ula_samples)
    # Calculate the average magnitude image
    ula_magnitude_avg = np.sqrt(ula_samples[:,0,:,:]**2 + ula_samples[:,1,:,:]**2)
    ula_magnitude_avg = np.mean(ula_magnitude_avg, 0)
    # Calculate the average phase image
    ula_phase_avg = np.arctan2(ula_samples[:,1,:,:], ula_samples[:,0,:,:] + 1e-5)
    ula_phase_avg = np.mean(ula_phase_avg, 0)
    # Calculate the std of the magnitude images
    ula_magnitude_std = np.sqrt(ula_samples[:,0,:,:]**2 + ula_samples[:,1,:,:]**2)
    ula_magnitude_std = np.std(ula_magnitude_std, 0)
    # Calculate the std of the phase images
    ula_phase_std = np.arctan2(ula_samples[:,1,:,:], ula_samples[:,0,:,:] + 1e-5)
    ula_phase_std = np.std(ula_phase_std, 0)

# Obtain the result of rPIE
rpie_results_path = result_path + test_sample_idx + "_rpie_results.pkl"
with open(rpie_results_path, 'rb') as f:
    rpie_reconstruction = pickle.load(f)
    rpie_reconstruction = rpie_reconstruction.numpy()
    rpie_magnitude = np.sqrt(rpie_reconstruction[0,0,:,:]**2 + rpie_reconstruction[0,1,:,:]**2)
    rpie_phase = np.arctan2(rpie_reconstruction[0,1,:,:], rpie_reconstruction[0,0,:,:] + 1e-5)

# Obtain the probe
probe_path = result_path + "probe.pkl"
with open(probe_path, 'rb') as f:
    probe = pickle.load(f)
    probe = probe.numpy()
    probe_magnitude = np.sqrt(probe[0,0,:,:]**2 + probe[0,1,:,:]**2)
    probe_phase = np.arctan2(probe[0,1,:,:], probe[0,0,:,:] + 1e-5)

# Obtain the gt
ground_truth_path = result_path + test_sample_idx + "_true_object.pkl"
with open(ground_truth_path, 'rb') as f:
    gt_object = pickle.load(f)
    gt_object = gt_object.numpy()
    gt_object_magnitude = np.sqrt(gt_object[0,0,:,:]**2 + gt_object[0,1,:,:]**2)
    gt_object_phase = np.arctan2(gt_object[0,1,:,:], gt_object[0,0,:,:] + 1e-5)

# Adjusting the figure to have a 2x4 layout where each column shows a magnitude-phase pair
fig, axes = plt.subplots(nrows = 2, ncols = 5, figsize = (15, 6))

# Updated titles for the new layout
titles = [
    "Ground Truth (Magnitude)", "Probe (Magnitude)", "rPIE (Magnitude)", "Proposed Mean (Magnitude)", "Proposed Uncertainty (Magnitude)",
    "Ground Truth (Phase)", "Probe (Phase)", "rPIE (Phase)", "Proposed Mean (Phase)", "Proposed Uncertainty (Phase)"
]

# Updated data list for the new layout
data = [
    gt_object_magnitude, probe_magnitude, rpie_magnitude, ula_magnitude_avg, ula_magnitude_std,
    gt_object_phase, probe_phase, rpie_phase, ula_phase_avg, ula_phase_std
]

# Plot each array in the corresponding subplot
for i, ax in enumerate(axes.flat):
    if i == 4 or i == 9:
        im = ax.imshow(data[i], cmap = 'jet')
    else:
        im = ax.imshow(data[i], cmap = 'gray')
    ax.set_title(titles[i],  fontsize = 12)
    ax.axis('off')  # Hide axes ticks
    # Add colorbars to each plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax = cax, orientation='vertical')
plt.tight_layout()
plt.savefig("example_visual_result.pdf", format='pdf', bbox_inches = 'tight', pad_inches = 0)
plt.close()
