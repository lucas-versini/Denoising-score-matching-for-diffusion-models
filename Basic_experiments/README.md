# Sampling and Score Matching Techniques

This folder contains scripts and notebooks for experimenting with various sampling and score matching techniques, including **Langevin Sampling**, **Denoising Score Matching**, and **Sliced Score Matching**.

## Folder Structure

### `utils` Folder

- **`density_utils.py`**  
  Functions for Gaussian Mixture Models (density, log density, score, sampling), and functions for Langevin dynamics.

- **`plot_utils.py`**  
  Functions to plot scores, Langevin dynamics, and densities (both empirical and true ones).

### `notebooks` Folder

- **`score_matching.ipynb`**  
  Train a basic model to learn the score of a Gaussian Mixture Model or a banana-shaped density (using the analytical expression of the score) with **Denoising Score Matching** or **Sliced Score Matching**.

- **`langevin_dynamics.ipynb`**  
  Sample from the score of a Gaussian Mixture Model using either **basic Langevin dynamics** or **annealed Langevin dynamics**.
