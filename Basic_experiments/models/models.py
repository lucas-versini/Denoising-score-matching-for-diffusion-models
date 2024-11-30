import torch
import torch.nn as nn
from tqdm import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt

from models.models_CondRefine import CondRefineNetDilated

from torchvision.utils import make_grid
import os

import numpy as np

class BasicNetwork(nn.Module):
    """
    Basic neural network with 2 input units, 2 hidden layers with ReLU activation, and 2 output units.
    """
    def __init__(self, hidden_size = 128):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x

####################################################################################################

# Noise Conditional Score Network using the CondRefineNetDilated architecture
class NCSN(nn.Module):
    def __init__(self, n_channels = 1, num_classes = 10, ngf = 64, image_size = 28, device = 'cpu'):
        super().__init__()

        self.device = device
        self.image_size = image_size
        self.n_channels = n_channels
        self.net = CondRefineNetDilated(image_size, num_classes, n_channels, ngf)
    
    def forward(self, input, sigma):
        return self.net(input, sigma)

    def sample(self, n_samples, n_steps, sigmas, eps = 5e-5, save_intermediate = False, save_freq = 100):
        x = torch.randn(n_samples, self.n_channels, self.image_size, self.image_size).to(self.device)
        sigmas = sigmas.to(self.device)
        intermediate = []

        with torch.no_grad():
            for i, sigma in enumerate(sigmas):
                alpha = eps * (sigma / sigmas[-1])**2
                for t in range(n_steps):
                    sigma_labels = i * torch.ones(n_samples, device = self.device, dtype = torch.int).to(self.device)
                    predicted_score = self(x, sigma_labels)
                    x += alpha / 2 * predicted_score.detach() + alpha**0.5 * torch.randn_like(x, device = self.device)
                    
                    if save_intermediate and t % save_freq == 0:
                        intermediate.append(x.cpu())

        if save_intermediate:
            return x.cpu(), intermediate

        return x.cpu()

def image_samples(refine_net, sigmas, epoch, save_path, n_samples = 10):
    n_samples = 10

    refine_net.eval()
    samples, intermediate = refine_net.sample(n_samples = n_samples, n_steps = 100,
                                        eps = 2e-5, sigmas = sigmas, save_intermediate = True)

    grid_samples = make_grid(samples, nrow = int(np.ceil(n_samples / 2)))

    grid_img = grid_samples.permute(1, 2, 0).clip(0, 1)

    # Directory to save samples
    path = save_path + "/" + "samples_" + str(epoch)
    if not os.path.exists(path):
        os.makedirs(path)

    plt.imshow(grid_img, cmap = 'Greys')
    plt.axis("off")
    plt.savefig(path + "/samples_" + str(epoch) + ".png")

    for i in range(len(intermediate)):
        grid_samples = make_grid(intermediate[i], nrow = 5)

        grid_img = grid_samples.permute(1, 2, 0).clip(0, 1)
        plt.imshow(grid_img, cmap = 'Greys')
        plt.axis("off")
        plt.savefig(path + "/samples_" + str(epoch) + "_" + str(i) + ".png")

def train_NCSN(model, train_loader, n_epochs, lr, sigmas, device, save_path = '', save_freq = 20, visualize = True, do_samples = False):
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas = (0.9, 0.999))

    loss_history = []
    for epoch in range(n_epochs):
        for x in tqdm(train_loader,
                      desc = f'Epoch {epoch + 1}/{n_epochs}',
                      total = len(train_loader)):
            x = x[0].to(device)
            batch_size = x.shape[0]

            labels_sigmas = torch.randint(len(sigmas), (batch_size,))
            batch_sigmas = sigmas[labels_sigmas].to(device).reshape([-1] + [1] * (x.dim() - 1))

            noise = torch.randn_like(x) * batch_sigmas
            x_noisy = x + noise

            optimizer.zero_grad()
            predicted_scores = model(x_noisy, labels_sigmas.to(device))

            noisy_scores = (-1 / (batch_sigmas**2) * noise).reshape(batch_size, -1) # shape (batch_size, n_channels * image_size * image_size)
            predicted_scores = predicted_scores.reshape(batch_size, -1) # shape (batch_size, n_channels * image_size * image_size)

            losses = 0.5 * torch.sum((predicted_scores - noisy_scores)**2, axis = -1)
            loss = torch.mean(losses * batch_sigmas.squeeze()**2, axis = 0)

            loss.backward()
            optimizer.step()

            loss_history.append(loss.data.cpu().numpy())

        if visualize:
            clear_output()
            plt.plot(loss_history)
            plt.title('Train loss')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.show()

        if (epoch + 1) % save_freq == 0:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), save_path + f"/epoch_{epoch + 1}.pth")

            if do_samples:
                image_samples(model, sigmas, epoch + 1, save_path)


    return loss_history
