import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.density_utils import gaussian_mixture_model_score
import matplotlib as mpl

# KernelDensity
from sklearn.neighbors import KernelDensity

def plot_real_predicted_scores(samples, scores, predicted_scores, model, device, weights, means, covariances):
    """
    Visualizes and compares real and predicted scores for a dataset, along with error analysis.

    Args:
        samples (np.ndarray): Array of sampled data points, shape (n_samples, 2).
        scores (np.ndarray): True score vectors for the samples, shape (n_samples, 2).
        predicted_scores (np.ndarray): Predicted score vectors for the samples, shape (n_samples, 2).
        model (torch.nn.Module): PyTorch model used to compute predicted scores on a grid.
        device (torch.device): Device (CPU/GPU) where the model computations are performed.
        weights (np.ndarray): Weights of the Gaussian components in the GMM, shape (n_components,).
        means (np.ndarray): Means of the Gaussian components in the GMM, shape (n_components, 2).
        covariances (np.ndarray): Covariance matrices of the Gaussian components in the GMM, 
                                  shape (n_components, 2, 2).
    """


    # Plot the real and predicted scores on two separate plots
    plt.figure(figsize = (12, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(samples[:, 0], samples[:, 1], c = "black", s = 2, label = 'Samples')
    plt.quiver(samples[:, 0], samples[:, 1], scores[:, 0], scores[:, 1], color = 'r', label = 'Real scores')
    plt.legend()
    plt.title('Real scores')

    plt.subplot(1, 2, 2)
    plt.scatter(samples[:, 0], samples[:, 1], c = "black", s = 2, label = 'Samples')
    plt.quiver(samples[:, 0], samples[:, 1], predicted_scores[:, 0], predicted_scores[:, 1], color = 'g', label = 'Predicted scores')
    plt.legend()
    plt.title('Predicted scores')

    plt.show()

    # Plot the scores on a grid
    x_min, x_max = samples[:, 0].min() - 3, samples[:, 0].max() + 3
    y_min, y_max = samples[:, 1].min() - 3, samples[:, 1].max() + 3

    x = np.linspace(x_min, x_max, int(3 * (x_max - x_min)))
    y = np.linspace(y_min, y_max, int(3 * (y_max - y_min)))

    X, Y = np.meshgrid(x, y)
    samples_ = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    predicted_scores = model(torch.tensor(samples_, dtype=torch.float32, device=device)).detach().cpu().numpy()
    scores = gaussian_mixture_model_score(samples_, weights, means, covariances)

    plt.figure(figsize = (14, 4))

    plt.subplot(1, 3, 1)
    plt.quiver(samples_[:, 0], samples_[:, 1], scores[:, 0], scores[:, 1], color='r')
    plt.scatter(samples[:, 0], samples[:, 1], c = "yellow", edgecolors = 'black')
    plt.title('Real scores')

    plt.subplot(1, 3, 2)
    plt.quiver(samples_[:, 0], samples_[:, 1], predicted_scores[:, 0], predicted_scores[:, 1], color='g')
    plt.scatter(samples[:, 0], samples[:, 1], c = "yellow", edgecolors = 'black')
    plt.title('Predicted scores')

    plt.subplot(1, 3, 3)
    plt.quiver(samples_[:, 0], samples_[:, 1], predicted_scores[:, 0] - scores[:, 0], predicted_scores[:, 1] - scores[:, 1], color='b')
    plt.scatter(samples[:, 0], samples[:, 1], c = "yellow", edgecolors = 'black')
    plt.title('Error')

    plt.show()

    # Plot the magnitude of the error
    x_min, x_max = samples[:, 0].min() - 20, samples[:, 0].max() + 20
    y_min, y_max = samples[:, 1].min() - 5, samples[:, 1].max() + 5
    x = np.linspace(x_min, x_max, int(5 * (x_max - x_min)))
    y = np.linspace(y_min, y_max, int(5 * (y_max - y_min)))

    X, Y = np.meshgrid(x, y)
    samples_ = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    predicted_scores = model(torch.tensor(samples_, dtype = torch.float32, device = device)).detach().cpu().numpy()
    scores = gaussian_mixture_model_score(samples_, weights, means, covariances)

    error = np.linalg.norm(predicted_scores - scores, axis=1)

    plt.figure(figsize = (12, 4))
    plt.contourf(X, Y, error.reshape(X.shape), cmap = "seismic", levels = np.linspace(0, error.max(), 100))
    plt.colorbar()
    plt.scatter(samples[:, 0], samples[:, 1], s = 2, c = "yellow", edgecolors = 'yellow')
    plt.title('Magnitude of the error')
    plt.show()

def plot_langevin_trajectory(x_history, score_function, x_min = None, x_max = None, y_min = None, y_max = None, *args):
    """ 
    Visualizes the trajectories of Langevin dynamics overlaid on the magnitude of the score function.

    Args:
        x_history (np.ndarray): Array of sampled trajectories, shape (T + 1, n_samples, 2).
        score_function (callable): Function that computes the score at given points.
        x_min (float, optional): Minimum x-coordinate for the plot. Defaults to dynamic range of x_history.
        x_max (float, optional): Maximum x-coordinate for the plot. Defaults to dynamic range of x_history.
        y_min (float, optional): Minimum y-coordinate for the plot. Defaults to dynamic range of x_history.
        y_max (float, optional): Maximum y-coordinate for the plot. Defaults to dynamic range of x_history.
        *args: Additional arguments passed to the score function.
    """
    x_min = x_history[:, :, 0].min() - 3 if x_min is None else x_min
    x_max = x_history[:, :, 0].max() + 3 if x_max is None else x_max
    y_min = x_history[:, :, 1].min() - 3 if y_min is None else y_min
    y_max = x_history[:, :, 1].max() + 3 if y_max is None else y_max
    
    x = np.linspace(x_min, x_max, int(3 * (x_max - x_min)))
    y = np.linspace(y_min, y_max, int(3 * (y_max - y_min)))

    X, Y = np.meshgrid(x, y)
    samples = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    scores = score_function(samples, *args)

    magnitude = np.linalg.norm(scores, axis=1)

    plt.figure(figsize = (12, 4))
    plt.contourf(X, Y, magnitude.reshape(X.shape), cmap = "seismic", levels = np.linspace(0, magnitude.max(), 100))
    plt.colorbar()

    for i in range(x_history.shape[1]):
        plt.plot(x_history[:, i, 0], x_history[:, i, 1], color = plt.cm.viridis(i / x_history.shape[1]))
    
    plt.title('Trajectory of the Langevin dynamics')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()

def plot_empirical_true_density(x_samples, density_function, x_min = None, x_max = None, y_min = None, y_max = None, smooth = False):
    """
        Visualizes the empirical and true density of a dataset on a 2D grid.

        Args:
            x_samples (np.ndarray): Array of sample points, shape (n_samples, 2).
            density_function (callable): Function to compute the true density at given points.
            x_min (float, optional): Minimum x-coordinate for the plot. Defaults to dynamic range of `x_samples`.
            x_max (float, optional): Maximum x-coordinate for the plot. Defaults to dynamic range of `x_samples`.
            y_min (float, optional): Minimum y-coordinate for the plot. Defaults to dynamic range of `x_samples`.
            y_max (float, optional): Maximum y-coordinate for the plot. Defaults to dynamic range of `x_samples`.
            smooth (bool, optional): Whether to smooth the empirical density using kernel density estimation (KDE). Defaults to `False`.
    """
    # Create a grid
    x_min = x_samples[:, 0].min() - 3 if x_min is None else x_min
    x_max = x_samples[:, 0].max() + 3 if x_max is None else x_max
    y_min = x_samples[:, 1].min() - 3 if y_min is None else y_min
    y_max = x_samples[:, 1].max() + 3 if y_max is None else y_max

    x = np.linspace(x_min, x_max, int(3 * (x_max - x_min)))
    y = np.linspace(y_min, y_max, int(3 * (y_max - y_min)))

    X, Y = np.meshgrid(x, y)

    grid = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis = 1)

    # Empirical density
    if smooth:
        kde = KernelDensity(kernel = 'gaussian', bandwidth = 0.5).fit(x_samples)
        log_dens = kde.score_samples(grid)
        dens = np.exp(log_dens)

    # True density
    true_dens = density_function(grid)

    # Plot
    plt.figure(figsize = (12, 4))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 4))

    max_level = max(dens.max(), true_dens.max()) if smooth else true_dens.max()

    if smooth:
        contour = ax1.contourf(X, Y, dens.reshape(X.shape), cmap = "seismic", levels = np.linspace(0, max_level, 100))
        fig.colorbar(contour, ax = ax1)
    else:
        h, _, _ = np.histogram2d(x_samples[:, 0], x_samples[:, 1])
        h /= np.sum(h)
        vmin = np.min(h)
        vmax = np.max(h)
        norm = mpl.colors.Normalize(vmin = vmin, vmax = vmax)
        sm = mpl.cm.ScalarMappable(norm = norm, cmap = mpl.cm.seismic)
        ax1.hist2d(x_samples[:, 0], x_samples[:, 1], bins = 200, cmap = "seismic", range = [[x_min, x_max], [y_min, y_max]])
        fig.colorbar(sm, ax = ax1)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_title('Empirical density')

    contour = ax2.contourf(X, Y, true_dens.reshape(X.shape), cmap = "seismic", levels = np.linspace(0, max_level, 100))
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_title('True density')

    # if smooth:
    #     error = np.sum(np.abs(dens - true_dens))
    #     ax2.suptitle(f'Error: {error:.2f}')
        
    fig.colorbar(contour, ax = ax2)

    plt.show()

