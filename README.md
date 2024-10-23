# Denoising-score-matching-for-diffusion-models

Dans `notebooks`:
- `denoising_GMM.ipynb` entraîne un modèle basique qui apprend le score d'un modèle de mélange gaussien (en l'entraînant directement pour qu'il apprenne le vrai score, dont on utilise l'expression)
- `langevin_dynamics.ipynb` illustre comment on peut échantillonner à partir du score. Mais (comme dit dans le papier) on ne retrouve pas les bons poids pour un mélange gaussien. Pour ça, il faudrait (par exemple) ajouter un sorte de Metropolis-Hastings, ce que j'ai essayé de faire, mais ça ne fonctionne pas...

TO DO: essayer de corriger la fin de `langevin_dynamics.ipynb` (sinon pas grave, on se contente d'utiliser des poids égaux...).
