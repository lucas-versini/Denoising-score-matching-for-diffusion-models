import torch
import torch.nn as nn
from tqdm import tqdm

class DenoisingScoreMatching:
    """ Train a model using denoising score matching """
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, sigma: float, device: str):
        self.model = model
        self.optimizer = optimizer
        self.sigma = sigma
        self.device = device
    
    def loss(self, x):
        x_noisy = x + torch.randn_like(x) * self.sigma
        predicted_score = self.model(x_noisy)
        target = -1 / self.sigma**2 * (x_noisy - x)

        return 0.5 * torch.sum((predicted_score - target) ** 2) / x.size(0)

    def step(self, x):
        self.optimizer.zero_grad()
        loss = self.loss(x)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def train(self, x, num_steps: int, step_print: int = 100):
        list_loss = []
        x = torch.tensor(x, dtype = torch.float32).to(self.device)
        tqdm_bar = tqdm(range(num_steps), desc = "Denoising Score Matching", total = num_steps)
        for i in tqdm_bar:
            loss = self.step(x)
            list_loss.append(loss)

            # Update the progress bar
            if i % step_print == 0:
                tqdm_bar.set_postfix(loss=loss)
            
        return list_loss

class SlicedScoreMatching:
    """ Train a model using sliced score matching """
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, device: str):
        self.model = model
        self.optimizer = optimizer
        self.device = device
    
    def loss(self, x):
        v = torch.randn_like(x)
        v = v / torch.norm(v, dim = -1, keepdim = True)

        score, Jv = torch.autograd.functional.jvp(self.model, x, v, create_graph = True)
        vJv = (Jv * v).sum(dim = -1)
        
        loss = torch.mean(vJv + 0.5 * torch.sum(score ** 2, dim = -1), dim = -1)

        return loss

    def step(self, x):
        self.optimizer.zero_grad()
        loss = self.loss(x)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def train(self, x, num_steps: int, step_print: int = 100):
        list_loss = []
        x = torch.tensor(x, dtype = torch.float32).to(self.device)
        tqdm_bar = tqdm(range(num_steps), desc = "Sliced Score Matching", total = num_steps)
        for i in tqdm_bar:
            loss = self.step(x)
            list_loss.append(loss)

            # Update the progress bar
            if i % step_print == 0:
                tqdm_bar.set_postfix(loss=loss)
            
        return list_loss