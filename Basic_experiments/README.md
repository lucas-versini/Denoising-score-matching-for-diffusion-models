# Sampling and score matching techniques

This folder contains scripts and notebooks for experimenting with various sampling and score matching techniques, including **Langevin Sampling**, **Denoising Score Matching**, and **Sliced Score Matching**.

## Folder structure

### `utils` folder

Contains tools for Gaussian Mixture Models, banana-shaped densities, sampling techniques such as Langevin and Annealed Langevin dynamics and utilities for visualizing scores and trajectories.

- `density_utils.py`
  Functions for Gaussian Mixture Models and "banana-shaped" density (density, log density, score, sampling), and functions for Langevin dynamics (Langevin dynamics sampling and Annealed Langevin dynamics).

- `plot_utils.py`
  Functions to plot scores, Langevin dynamics, and densities (both empirical and true ones).

### `models` folder

Contrains scripts to define and train models, focusing on Noise Conditional Score Networks (NCSNs). It includes modules for a basic neural network and conditional neural networks (models_CondRefine.py), a framework for training and sampling NCSNs (models.py), and utilities for Denoising and Sliced Score Matching (train.py). 

- `models_CondRefine.py`
  Defines conditional neural network modules for score-based generative modeling, including convolutional utilities, conditional residual blocks (CondRCUBlock, CondCRPBlock), multi-scale feature fusion (CondMSFBlock), and refinement blocks (CondRefineBlock). Includes ConditionalInstanceNorm2dPlus for class-specific normalization and the CondRefineNetDilated, a hierarchical, dilated refinement network for Noise Conditional Score Networks (NCSN).

- `models.py`
  Implements models and utilities for training and sampling with Noise Conditional Score Networks (NCSNs), including a basic feedforward network (BasicNetwork) for baseline comparisons and the NCSN model leveraging CondRefineNetDilated for score-based generative modeling. Provides utilities for generating and visualizing samples (image_samples) and a training framework (train_NCSN) that uses noisy data augmentation, score-matching loss, checkpointing, and optional sample visualization during training.

- `train.py`
  Implements two classes, DenoisingScoreMatching and SlicedScoreMatching. DenoisingScoreMatching class uses noise-perturbed data to train models by minimizing the discrepancy between predicted and true noise gradients. The SlicedScoreMatching class optimizes a sliced score-matching loss, leveraging random vector projections. Both classes include methods for loss computation, gradient updates, and iterative training, with support for tracking progress using the tqdm library.

### `notebooks` folder
Includes notebooks for score-based learning and implicit sampling from probability distributions. 
- `score_matching.ipynb`
  Trains a basic model to learn the score of a Gaussian Mixture Model or a banana-shaped density (using the analytical expression of the score) with **Denoising Score Matching** or **Sliced Score Matching**.

- `langevin_dynamics.ipynb`
  Samples from the score of a Gaussian Mixture Model using either **basic Langevin dynamics** or **annealed Langevin dynamics**.
