
# Bayesian Inversion for Ptychography with Deep Generative Priors

This repository contains the official implementation of the paper **"xxxx"** by xxxx.

## Overview

Plug-and-Play (PnP) algorithms have been a popular framework for solving inverse problems by leveraging advanced image denoisers as implicit priors. This repository implements a Provable Plug-and-Play ADMM algorithm with denoisers that converge to Gaussian distributed solutions. The main contribution of the paper is to provide convergence guarantees for PnP algorithms using a wide range of modern denoisers.

## Features

- **Implementation of Provable PnP-ADMM:** The core implementation of the Provable PnP-ADMM algorithm as described in the paper.
- **Integration with Denoisers:** The code allows the use of different denoisers, including CNN-based ones, as part of the ADMM iterations.
- **Numerical Experiments:** Scripts to reproduce the experiments from the paper, including comparisons with other methods.

## Repository Structure

- `./data/` - Contains sample datasets used in the experiments.
- `./models/` - Includes pre-trained models for denoising.
- `./src/` - The main source code for implementing the PnP-ADMM algorithm.
  - `./src/pnp_admm.py` - The core PnP-ADMM implementation.
  - `./src/denoisers/` - Contains the implementation of various denoisers.
  - `./src/utils/` - Utility functions used across the codebase.
- `./experiments/` - Scripts and notebooks to run and evaluate the experiments from the paper.
- `./results/` - Folder to store the results of the experiments.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/cekmekci/Provable_Plug_and_Play.git
   cd Provable_Plug_and_Play
   ```

2. **Install the required dependencies:**

   It is recommended to use a virtual environment:

   ```bash
   python -m venv pnp_env
   source pnp_env/bin/activate  # On Windows, use `pnp_env\Scripts\activate`
   ```

   Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the PnP-ADMM Algorithm

To run the PnP-ADMM algorithm with a specific denoiser, you can use the following command:

```bash
python src/pnp_admm.py --input ./data/sample_image.png --denoiser cnn --output ./results/output_image.png
```

### Reproducing Experiments

The `experiments` folder contains scripts to reproduce the results presented in the paper. To run an experiment, navigate to the `experiments` directory and execute the corresponding script:

```bash
cd experiments
python run_experiment.py --config config.yaml
```

### Custom Denoisers

To use a custom denoiser, implement it as a function in the `src/denoisers/` directory, and update the corresponding script to call your custom denoiser.

## Citation

If you find this repository useful in your research, please consider citing:

```
@article{pnp_admm,
  title={Provable Plug-and-Play ADMM with Denoisers Converging to Gaussian Distributed Denoised Solutions},
  author={Can Emre Kısmekci, Yi Li, Saiprasad Ravishankar},
  journal={},
  year={2024},
  url={https://github.com/cekmekci/Provable_Plug_and_Play}
}
```

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

We acknowledge the support from [Institution/Project] that made this work possible. We also thank the contributors and maintainers of the open-source libraries used in this project.
