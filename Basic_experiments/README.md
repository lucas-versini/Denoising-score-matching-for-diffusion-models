- `utils` folder:
    - `density_utils.py`: functions for Gaussian Mixture Models (density, log density, score, sampling), functions for Langevin dynamics (which should be put somewhere else I guess).
    - `plot_utils.py`: functions to plot scores, Langevin dynamics, densities (empirical ones and true ones).

- `notebooks` folder:
    - `denoising_GMM.ipynb`: train a very basic model to learn the score of a GMM (using directly the expression of the score).
    - `langevin_dynamics.ipynb`: sample from the score of a GMM, either using basic Langevin dynamics, or annealed Langevin dynamics.
