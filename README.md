
# Bayesian Inversion for Ptychography with Deep Generative Priors

This repository contains the official implementation of the paper **"xxxx"** by xxxx.

## Overview

Include the abstract of the paper here.

## Key Features:

- **Bayesian Framework:** Incorporates a Bayesian approach to inversion, allowing for principled uncertainty quantification in the reconstructed images.
- **Generative Priors:** Utilizes generative models as priors, improving the reconstruction quality by leveraging learned representations of image data.
- **Modular Design:** The code is designed to be modular, making it easy to adapt and extend for different generative models or imaging settings.
- **GPU Support:** The implementation is optimized for GPU acceleration using PyTorch, enabling efficient computation even for large-scale problems.
- **Reproducibility:** Scripts to reproduce the experiments from the paper, including comparisons with rPIE.

## Repository Structure

- `./models/` - Includes a pre-trained Wasserstein GAN model for generating ptychographic objects of interest.
- `./utils/` - Contains some utility functions for data-loading, ptychography, and training.
- `models.py` -  Contains the implementation of the Wasserstein GAN architecture used for the experiments.
- `samplers.py` -  Contains the implementation of the unadjusted Langevin algorithm proposed in the paper.
- `train.py` - Trains the Wasserstein GAN on the MNIST dataset.
- `reconstruction.py` - Runs the proposed method and the rPIE on test data and saves the results.

- `display_animation.py` - Creates an animation showing the samples generated by the proposed method.
- `display_error_metrics.py` - Calculates the error for different conditions of the problem.
- `display_iter_errors.py` - Displays the error as a function of iterations for the proposed method.
- `display_uncertainty_error_correlation.py` - Demonstrates the correlation between the error and the uncertainty estimates.
- `display_uncertainty_maps.py` - Provides some example uncertainty maps.
- `display_visual_result.py` - Shows visuals demonstrating the reconstruction performance.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/cekmekci/bayesian-inversion-ptychography-generative-priors.git
   cd bayesian-inversion-ptychography-generative-priors
   ```

2. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the proposed Wasserstein GAN, you can use the following command:

```bash
python train.py
```

### Running the Proposed Method

To run the proposed method, you can use the following command:

```bash
python reconstruction.py
```

### Reproducing the Results

The scripts starting with the phrase "display" reproduce the results presented in the paper. To run an experiment, execute the corresponding script:

```bash
python display_animation.py
```

```bash
python display_error_metrics.py
```

```bash
python display_iter_errors.py
```

```bash
python display_uncertainty_error_correlation.py
```

```bash
python display_uncertainty_maps.py
```

```bash
python display_visual_result.py
```

## Citation

If you find this repository useful in your research, please consider citing:

```
@article{
}
```

## License

xxxx

## Acknowledgements

xxxx
