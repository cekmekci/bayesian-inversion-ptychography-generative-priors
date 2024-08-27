import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.utils_ptycho import l2_error
from scipy.stats import pearsonr, spearmanr


# Specify the overlap ratio and probe amplitude values
overlap_ratio = 0.22727272727272727
probe_amplitude = 100

# Specify the number of pixels to crop before calculating the metrics. Make sure
# that it is equal to the half of the width of the probe.
num_pix_crop = 8

# Path of the folder containing the pickle files
result_path = "./results/overlap_ratio_" + str(overlap_ratio) + "_probe_amplitude_" + str(probe_amplitude) + "/"

# Index of the test example
test_sample_idx = "7"

# Obtain the reconstructed image (mean) and the uncertainty (std) provided by our method
ula_poisson_results_path = result_path + test_sample_idx + "_ula_poisson_results.pkl"
with open(ula_poisson_results_path, 'rb') as f:
    ula_samples = pickle.load(f)
    ula_samples = np.array(ula_samples)
    ula_samples = ula_samples[:,0,:,:] + 1j * ula_samples[:,1,:,:]
    # Obtain the complex reconstructed image (mean)
    ula_mean = np.mean(ula_samples, 0)
    ula_mean = ula_mean[num_pix_crop:-num_pix_crop,num_pix_crop:-num_pix_crop]
    # Obtain the uncertainty map
    ula_std = np.std(ula_samples, 0)
    ula_std = ula_std[num_pix_crop:-num_pix_crop,num_pix_crop:-num_pix_crop]

# Obtain the true object
ground_truth_path = result_path + test_sample_idx + "_true_object.pkl"
with open(ground_truth_path, 'rb') as f:
    gt_object = pickle.load(f)
    gt_object = gt_object.numpy()
    gt_object = gt_object[0,0,:,:] + 1j * gt_object[0,1,:,:]
    gt_object = gt_object[num_pix_crop:-num_pix_crop,num_pix_crop:-num_pix_crop]

# Calculate the pixel-wise error
theta_opt = -1 * np.angle(np.vdot(ula_mean, gt_object))
ula_mean_corrected = ula_mean * np.exp(1j * theta_opt)
error_map = np.abs(ula_mean_corrected - gt_object)

# Save the error and uncertainty maps
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 6))
im = axes[0].imshow(ula_std, cmap = 'jet')
axes[0].set_title("Uncertainty",  fontsize = 12)
axes[0].axis('off')  # Hide axes ticks
divider = make_axes_locatable(axes[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax = cax, orientation='vertical')
im = axes[1].imshow(error_map, cmap = 'jet')
axes[1].set_title("Error",  fontsize = 12)
axes[1].axis('off')  # Hide axes ticks
divider = make_axes_locatable(axes[1])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax = cax, orientation='vertical')
plt.tight_layout()
plt.savefig("uncertainty_error_correlation.pdf", format='pdf', bbox_inches = 'tight', pad_inches = 0)
plt.close()

# Flatten the maps to 1D arrays for correlation computation
error_values = error_map.flatten()
uncertainty_values = ula_std.flatten()

# Compute Pearson and Spearman correlation coefficients
pearson_corr, pearson_p_value = pearsonr(uncertainty_values, error_values)
spearman_corr, spearman_p_value = spearmanr(uncertainty_values, error_values)

print("Pearson Correlation:", pearson_corr, "Pearson P Value:", pearson_p_value, "Spearman Correlation:", spearman_corr, "Spearman P Value:", spearman_p_value)
with open("uncertainty_error_correlation.txt", "w") as file:
    file.write("Pearson Correlation: " + str(pearson_corr) + " Pearson P Value: " + str(pearson_p_value) + " Spearman Correlation: " + str(spearman_corr) + " Spearman P Value: " + str(spearman_p_value) + "\n")

# Scatter plot of uncertainty vs. error
plt.figure(figsize=(6, 6))
plt.scatter(uncertainty_values, error_values, alpha = 0.5, s = 1)
plt.title('Scatter Plot of Uncertainty vs. Error')
plt.xlabel('Uncertainty')
plt.ylabel('Error')
plt.grid(True, which = 'both', linestyle = '--', color = 'gray', linewidth = 0.5)
plt.savefig("uncertainty_error_scatter.pdf", format='pdf', bbox_inches = 'tight', pad_inches = 0)
plt.close()

# Joint histogram of uncertainty and error
plt.figure(figsize=(6, 6))
plt.hist2d(uncertainty_values, error_values, bins = 100, cmap='viridis')
plt.colorbar(label='Frequency')
plt.title('Joint Histogram of Uncertainty and Error')
plt.xlabel('Uncertainty')
plt.ylabel('Error')
plt.grid(True, which = 'both', linestyle = '--', color = 'gray', linewidth = 0.5)
plt.savefig("uncertainty_error_histogram.pdf", format='pdf', bbox_inches = 'tight', pad_inches = 0)
plt.close()
