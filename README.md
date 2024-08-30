
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

### Running the ULA Algorithm

To run the ULA algorithm with the provided Wasserstein GAN model, you can use the following command:

```bash
python 
```

### Reproducing Experiments

The scripts starting with the phrase "display" reproduce the results presented in the paper. To run an experiment, execute the corresponding script:

```bash
python 
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
