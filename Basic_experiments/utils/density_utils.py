import numpy as np
from tqdm import tqdm
import scipy as sp
import matplotlib.pyplot as plt

# Gaussian Mixture Model (GMM)

def gaussian_mixture_model_score(samples, weights, means, covariances):
    """
    Gaussian Mixture scoring

    Args:
        samples (np.array): Samples to score, shape (n_samples, dim)
        weights (np.array): Mixing coefficients of the Gaussian components, shape (n_components,)
        means (np.array): Means of the Gaussian components, shape (n_components, dim)
        covariances (np.array): Standard deviations of the Gaussian components, shape (n_components, dim, dim)      
    
    Returns:
        np.array: Scores of the samples, shape (n_samples, dim)
    """

    # Check if the input is correct
    assert len(means) == len(covariances) == len(weights), "The input arrays must have the same length"
    assert np.isclose(np.sum(weights), 1), "The mixing coefficients must sum to 1"

    # Compute the score
    inv_covariances = np.linalg.inv(covariances)

    samples_means = samples[:, None] - means[None, :] # (n_samples, n_components, dim)
    res = np.einsum('ijk,jkl,ijl->ij', samples_means, inv_covariances, samples_means) # (n_samples, n_components)

    coefficients = weights[None, :] * np.exp(- 0.5 * res - 0.5 * np.linalg.slogdet(covariances)[1]) # (n_samples, n_components)
    coefficients /= coefficients.sum(axis = 1)[:, None] # (n_samples, n_components)

    scores = coefficients[:, :, None] * np.einsum('ijk,lik->lij', inv_covariances, samples_means) # (n_samples, n_components, dim)
    scores = -scores.sum(axis = 1) # (n_samples, dim)

    return scores

def gaussian_mixture_model_sampling(n_samples, weights, means, covariances):
    """
    Gaussian Mixture Sampling

    Args:
        n_samples (int): Number of samples to generate
        weights (np.array): Mixing coefficients of the Gaussian components, shape (n_components,)
        means (np.array): Means of the Gaussian components, shape (n_components, dim)
        covariances (np.array): Standard deviations of the Gaussian components, shape (n_components, dim, dim)  
    
    Returns:
        np.array: Samples generated from the Gaussian Mixture Model, shape (n_samples, dim)
        np.array: Cluster assignment of the samples, shape (n_samples,)
        np.array: Scores of the samples, shape (n_samples, dim)
    """

    # Check if the input is correct
    assert len(means) == len(covariances) == len(weights), "The input arrays must have the same length"
    assert np.isclose(np.sum(weights), 1), "The mixing coefficients must sum to 1"

    # Generate the samples
    n_components = means.shape[0]

    choice_components = np.random.choice(n_components, size = n_samples, p = weights)
    samples = np.array([np.random.multivariate_normal(means[component], covariances[component]) for component in choice_components])
    scores = gaussian_mixture_model_score(samples, weights, means, covariances)

    return samples, choice_components, scores

def gaussian_mixture_model_density(x, weights, means, covariances):
    """
    Gaussian Mixture Density

    Args:
        x (np.array): Points at which to compute the density, shape (n_points, dim)
        weights (np.array): Mixing coefficients of the Gaussian components, shape (n_components,)
        means (np.array): Means of the Gaussian components, shape (n_components, dim)
        covariances (np.array): Standard deviations of the Gaussian components, shape (n_components, dim, dim)  
    
    Returns:
        np.array: Density at the points, shape (n_points,)
    """

    # Check if the input is correct
    assert len(means) == len(covariances) == len(weights), "The input arrays must have the same length"
    assert np.isclose(np.sum(weights), 1), "The mixing coefficients must sum to 1"

    # Compute the density
    inv_covariances = np.linalg.inv(covariances) # (n_components, dim, dim)

    x_means = x[:, None] - means[None, :] # (n_points, n_components, dim)
    res = np.einsum('ijk,jkl,ijl->ij', x_means, inv_covariances, x_means) # (n_points, n_components)

    coefficients = weights[None, :] * np.exp(- 0.5 * res - 0.5 * np.linalg.slogdet(covariances)[1]) # (n_points, n_components)

    density = np.sum(coefficients, axis = -1) * 1 / ((2 * np.pi)**(means.shape[1] / 2))

    return density

def gaussian_mixture_model_log_density(x, weights, means, covariances):
    """
    Gaussian Mixture log density

    Args:
        x (np.array): Points at which to compute the density, shape (n_points, dim)
        weights (np.array): Mixing coefficients of the Gaussian components, shape (n_components,)
        means (np.array): Means of the Gaussian components, shape (n_components, dim)
        covariances (np.array): Standard deviations of the Gaussian components, shape (n_components, dim, dim)  
    
    Returns:
        np.array: log density at the points, shape (n_points,)
    """

    # Check if the input is correct
    assert len(means) == len(covariances) == len(weights), "The input arrays must have the same length"
    assert np.isclose(np.sum(weights), 1), "The mixing coefficients must sum to 1"

    # Compute the density
    inv_covariances = np.linalg.inv(covariances) # (n_components, dim, dim)

    x_means = x[:, None] - means[None, :] # (n_points, n_components, dim)
    res = np.einsum('ijk,jkl,ijl->ij', x_means, inv_covariances, x_means) # (n_points, n_components)

    coefficients = np.log(weights[None, :]) - 0.5 * res - 0.5 * np.linalg.slogdet(covariances)[1] # (n_points, n_components)

    log_density = np.log(1 / (2 * np.pi)**(means.shape[1] / 2)) + sp.special.logsumexp(coefficients, axis = -1) # (n_points,)
    
    return log_density


# Banana distribution : an additional to example

def banana_density(x, b = 0.5):
    x_ = np.copy(x)
    x_[:, 1] = x[:, 1] + b * (x[:, 0]**2 - 1)
    return gaussian_mixture_model_density(x_, np.array([1.]), np.array([[0., 0.]]), np.eye(2))

def log_density_banana(x, b = 0.5):
    x_ = np.copy(x)
    x_[:, 1] = x[:, 1] + b * (x[:, 0]**2 - 1)
    return gaussian_mixture_model_log_density(x_, np.array([1.]), np.array([[0., 0.]]), np.array([np.eye(2)]))

def banana_score(x, b = 0.5):
    x_ = np.copy(x)
    x_[:, 1] = x[:, 1] + b * (x[:, 0]**2 - 1)
    return gaussian_mixture_model_score(x_, np.array([1.]), np.array([[0., 0.]]), np.array([np.eye(2)]))

def banana_sampling(n_samples, b = 0.5):
    samples = np.random.randn(n_samples, 2)
    samples[:, 1] = samples[:, 1] + b * (samples[:, 0]**2 - 1)
    
    score = banana_score(samples, b)
    
    return samples, score

# Langevin dynamics

def langevin_sampling(x, score_function, n_samples, coefficient = 1e-4, T = 300):
    """
    Sample from the Langevin dynamics.
    
    Parameters
    x (array, shape (2,)): Initial point.
    score_function (function of x): Score function.
    n_samples (int): Number of samples to generate.
    coefficient (float, optional): Coefficient of the Langevin dynamics.
    T (int, optional): Number of iterations.

    Returns
    all_x (array, shape (T + 1, n_samples, 2)): All the samples generated.
    """
    all_x = np.zeros((T + 1, n_samples, 2))
    all_x[0][:, :] = x

    for i in tqdm(range(1, T + 1), desc = 'Iteration', total = T):
        z = np.random.randn(n_samples, 2)
        all_x[i][:, :] = all_x[i - 1] \
                        + coefficient / 2 * score_function(all_x[i - 1]) \
                        + np.sqrt(coefficient) * z
    return all_x

#Metropolis Hastings correction for Langevin dynamics
def langevin_metropolis_hastings_sampling(x, score_function, log_density_function, n_samples, step, T = 100):
    """
    Sample from the corrected Langevin Metropolis-Hastings dynamics.

    Parameters
    x : array, shape (2,)
        Initial point.
    score_function : function of x
        Score function.
    log_density_function : function of x
        Log density function.
    n_samples : int
        Number of samples to generate.
    step : float
        Step size of the Langevin dynamics.
    T : int, optional
        Number of iterations.
    """
    all_x = np.zeros((T + 1, n_samples, 2))
    all_x[0][:, :] = x

    def log_kernel(x, y):
        return -1 / (2 * step) * np.sum((y - x - step / 2 * score_function(x))**2, axis = -1)
    
    acceptance_rate = 0

    for i in tqdm(range(1, T + 1), desc = 'Iteration', total = T):
        z = np.random.randn(n_samples, 2)
        x_proposal = all_x[i - 1] + step / 2 * score_function(all_x[i - 1]) + np.sqrt(step) * z
        log_ratio = log_density_function(x_proposal) - log_density_function(all_x[i - 1]) \
                    + log_kernel(x_proposal, all_x[i - 1]) - log_kernel(all_x[i - 1], x_proposal)
        accept_prob = np.minimum(1, np.exp(log_ratio))
        accept = np.random.rand(n_samples) < accept_prob
        all_x[i] = np.where(accept[:, None], x_proposal, all_x[i - 1])
        acceptance_rate += accept.mean()
    acceptance_rate /= T
    print(f'Acceptance rate: {acceptance_rate}')
    return all_x

def annealed_langevin_dynamics(x, score_function, n_samples, sigmas, epsilon = 1e-3, T = 300):
    """
    Sample from the Annealed Langevin dynamics.
    
    Parameters
    x : array, shape (2,)
        Initial point.
    score_function : function of (x, sigma)
        Score function.
    n_samples : int
        Number of samples to generate.
    T : int, optional
        Number of iterations.
    epsilon : double
        Step size for ajusting the noise level.
    n_steps : int, optional
        Number of steps.
    
    Returns
    all_x : array, shape (T + 1, n_samples, 2)
        All the samples generated.
    """
    all_x = np.zeros((T * len(sigmas) + 1, n_samples, 2))
    all_x[0][:, :] = x

    for i, sigma in tqdm(enumerate(sigmas), desc = 'Sigma', total = len(sigmas)):
        alpha = epsilon * (sigma / sigmas[-1])**2
        for t in range(T):
            z = np.random.randn(n_samples, 2)
            all_x[i * T + t + 1][:, :] = all_x[i * T + t] \
                            + alpha / 2 * score_function(all_x[i * T + t], sigma) \
                            + np.sqrt(alpha) * z
    return all_x