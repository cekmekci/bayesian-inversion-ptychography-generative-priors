import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Specify the overlap ratio and probe amplitude values
overlap_ratio = 0.045454545454545456
probe_amplitude = 100

# Path of the folder containing the pickle files
result_path = "./results/overlap_ratio_" + str(overlap_ratio) + "_probe_amplitude_" + str(probe_amplitude) + "/"

# Index of the test example
test_sample_idx = "8"

# Calculate the summary images for our method
ula_poisson_results_path = result_path + test_sample_idx + "_ula_poisson_results.pkl"
with open(ula_poisson_results_path, 'rb') as f:
    ula_samples = pickle.load(f)
    ula_samples = np.array(ula_samples)
    # Calculate the average magnitude image
    ula_magnitude_samples = np.sqrt(ula_samples[:,0,:,:]**2 + ula_samples[:,1,:,:]**2)
    # Calculate the average phase image
    ula_phase_samples = np.arctan2(ula_samples[:,1,:,:], ula_samples[:,0,:,:] + 1e-5)

# Magnitude animation
fig, ax = plt.subplots()
im = ax.imshow(ula_magnitude_samples[0,:,:], cmap = 'gray')
ax.set_title("Magnitude Samples (k = 1)",  fontsize = 12)
ax.axis('off')
def update(frame):
    im.set_array(ula_magnitude_samples[frame,:,:])
    ax.set_title("Magnitude Samples (k = "+str(frame) + ")",  fontsize = 12)
    return [im]
ani = animation.FuncAnimation(fig, update, frames = ula_magnitude_samples.shape[0], blit = True, interval = 50)
ani.save('magnitude_samples.mp4', writer='ffmpeg')
plt.close()

# Phase animation
fig, ax = plt.subplots()
im = ax.imshow(ula_phase_samples[0,:,:], cmap='gray')
ax.set_title("Phase Samples (k = 1)",  fontsize = 12)
ax.axis('off')
def update(frame):
    im.set_array(ula_phase_samples[frame,:,:])
    ax.set_title("Phase Samples (k = "+str(frame) + ")",  fontsize = 12)
    return [im]
ani = animation.FuncAnimation(fig, update, frames = ula_phase_samples.shape[0], blit = True, interval = 50)
ani.save('phase_samples.mp4', writer='ffmpeg')
plt.close()
