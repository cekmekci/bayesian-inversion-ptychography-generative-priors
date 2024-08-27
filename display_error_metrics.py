import numpy as np
import os
import pickle
import pandas as pd

from utils.utils_ptycho import l2_error

# Specify the overlap ratios and probe amplitudes we used in our experiments
overlap_ratios = [0.045454545454545456]
probe_amplitudes = [100]

# Specify the number of pixels to crop before calculating the metrics. Make sure
# that it is equal to the half of the width of the probe.
num_pix_crop = 8

# Dictionary containing the results
results = dict()

for overlap_ratio in overlap_ratios:

    for probe_amplitude in probe_amplitudes:

        # Path of the folder containing the pickle files
        result_path = "./results/overlap_ratio_" + str(overlap_ratio) + "_probe_amplitude_" + str(probe_amplitude) + "/"

        ula_poisson_errors = []
        rpie_errors = []

        for i in range(100):

            # Index of the test example
            test_sample_idx = str(i)

            # Obtain the samples of the proposed method and their mean
            ula_poisson_results_path = result_path + test_sample_idx + "_ula_poisson_results.pkl"
            with open(ula_poisson_results_path, 'rb') as f:
                ula_samples = pickle.load(f)
                ula_samples = np.array(ula_samples)
                ula_samples_mean = np.mean(ula_samples, 0, keepdims = True)
                ula_samples_mean = ula_samples_mean[:,:,num_pix_crop:-num_pix_crop,num_pix_crop:-num_pix_crop]

            # Obtain the true object
            ground_truth_path = result_path + test_sample_idx + "_true_object.pkl"
            with open(ground_truth_path, 'rb') as f:
                gt_object = pickle.load(f)
                gt_object = gt_object.numpy()
                gt_object = gt_object[:,:,num_pix_crop:-num_pix_crop,num_pix_crop:-num_pix_crop]

            # Obtain the result of rPIE
            rpie_results_path = result_path + test_sample_idx + "_rpie_results.pkl"
            with open(rpie_results_path, 'rb') as f:
                rpie_reconstruction = pickle.load(f)
                rpie_reconstruction = rpie_reconstruction.numpy()
                rpie_reconstruction = rpie_reconstruction[:,:,num_pix_crop:-num_pix_crop,num_pix_crop:-num_pix_crop]

            # Calculate the errors
            ula_error = l2_error(gt_object, ula_samples_mean)
            rpie_error = l2_error(gt_object, rpie_reconstruction)

            # Append the lists
            ula_poisson_errors.append(ula_error)
            rpie_errors.append(rpie_error)

        ula_poisson_errors = np.array(ula_poisson_errors)
        rpie_errors = np.array(rpie_errors)

        # Append the dictionary
        results[(overlap_ratio, probe_amplitude)] = dict()
        results[(overlap_ratio, probe_amplitude)]["ula_poisson"] = ula_poisson_errors
        results[(overlap_ratio, probe_amplitude)]["rpie"] = rpie_errors

# Prepare the data for the table
table_data = []
for (overlap_ratio, probe_amplitude), methods in results.items():
    for method, errors in methods.items():
        mean_value = np.mean(errors)
        std_value = np.std(errors)
        table_data.append({
            'Ovelap Ratio': overlap_ratio,
            'Probe Amplitude': probe_amplitude,
            'Method': method,
            'Mean Error': mean_value,
            'Std Dev': std_value
        })

# Convert to a DataFrame
df = pd.DataFrame(table_data)

# Save the table
df.to_csv('error_metrics.csv', index = False)
