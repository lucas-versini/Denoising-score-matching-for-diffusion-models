# Denoising-score-matching-for-diffusion-models

- `utils` folder:
    - `density_utils.py`: functions for Gaussian Mixture Models (density, log density, score, sampling), functions for Langevin dynamics (which should be put somewhere else I guess).
    - `plot_utils.py`: functions to plot scores, Langevin dynamics, densities (empirical ones and true ones).

- `notebooks` folder:
    - `denoising_GMM.ipynb`: train a very basic model to learn the score of a GMM (using directly the expression of the score).
    - `langevin_dynamics.ipynb`: sample from the score of a GMM, either using basic Langevin dynamics, or annealed Langevin dynamics.

**Remarques:**
- `langevin_dynamics.ipynb`: comme dit dans le papier, avec un simple Langevin dynamics on ne retrouve pas les bons poids pour un mélange gaussien. En revanche, en utilisant annealed Langevin dynamics, ça marche bien mieux. J'ai aussi essayé d'utiliser une sorte de Metropolis-Hastings, mais ça ne fonctionne pas... Le code traîne dans `density_utils.py`, mais si on ne le fait pas marcher on l'enlèvera.
