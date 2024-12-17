import numpy as np
import tqdm
from losses.dsm import anneal_dsm_score_estimation

import torch.nn.functional as F
import logging
import torch
import os
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from models.ncsnv2 import NCSNv2
from datasets import get_dataset
from losses import get_optimizer
from models import anneal_Langevin_dynamics
from models import get_sigmas
from models.ema import EMAHelper

__all__ = ['NCSNRunner']

""" This script contains the training and sampling procedures """

def get_model(config):
    if config.data.dataset in ['CIFAR10', 'MNIST', 'OxfordPets']:
        return NCSNv2(config).to(config.device)

class NCSNRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok = True)

    def train(self):
        """ Training function """

        # Get the datasets and dataloaders
        dataset, test_dataset = get_dataset(self.args, self.config)
        dataloader = DataLoader(dataset, batch_size = self.config.training.batch_size, shuffle = True,
                                num_workers = self.config.data.num_workers)
        test_loader = DataLoader(test_dataset, batch_size = self.config.training.batch_size, shuffle = True,
                                 num_workers = self.config.data.num_workers)
        test_iter = iter(test_loader)

        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        score = torch.nn.DataParallel(get_model(self.config))

        # Get the optimizer
        optimizer = get_optimizer(self.config, score.parameters())

        start_epoch = 0
        step = 0

        print(f"Ema is {self.config.model.ema}")
        if self.config.model.ema:
            ema_helper = EMAHelper(mu = self.config.model.ema_rate)
            ema_helper.register(score)

        sigmas = get_sigmas(self.config)

        for epoch in range(start_epoch, self.config.training.n_epochs):
            for X, _ in dataloader:
                score.train()
                step += 1

                X = X.to(self.config.device)

                # Compute the loss for the current X
                loss = anneal_dsm_score_estimation(score, X, sigmas, self.config.training.anneal_power)

                logging.info("step: {}, loss: {}".format(step, loss.item()))

                # Optimize the parameters to reduce the loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(score)

                if step >= self.config.training.n_iters:
                    return

                if step % 100 == 0:
                    # Use EMA: update the model
                    if self.config.model.ema:
                        test_score = ema_helper.ema_copy(score)
                    else:
                        test_score = score

                    # Test the model
                    test_score.eval()
                    try:
                        test_X, _ = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, _ = next(test_iter)

                    test_X = test_X.to(self.config.device)

                    with torch.no_grad():
                        test_dsm_loss = anneal_dsm_score_estimation(test_score, test_X, sigmas,
                                                                    self.config.training.anneal_power)

                        logging.info("step: {}, test_loss: {}".format(step, test_dsm_loss.item()))

                        del test_score

                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    # torch.save(states, os.path.join(self.args.log_path, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log_path, 'checkpoint.pth'))

                    # Create some samples during training
                    if self.config.training.snapshot_sampling:
                        if self.config.model.ema:
                            test_score = ema_helper.ema_copy(score)
                        else:
                            test_score = score

                        test_score.eval()

                        init_samples = torch.rand(36, self.config.data.channels,
                                                  self.config.data.image_size, self.config.data.image_size,
                                                  device=self.config.device)

                        all_samples = anneal_Langevin_dynamics(init_samples, test_score, sigmas.cpu().numpy(),
                                                               self.config.sampling.n_steps_each,
                                                               self.config.sampling.step_lr,
                                                               final_only=True, verbose=True,
                                                               denoise=self.config.sampling.denoise)

                        sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                      self.config.data.image_size,
                                                      self.config.data.image_size)

                        sample = torch.clamp(sample, 0.0, 1.0)

                        image_grid = make_grid(sample, 6)
                        save_image(image_grid,
                                   os.path.join(self.args.log_sample_path, 'image_grid_{}.png'.format(step)))
                        torch.save(sample, os.path.join(self.args.log_sample_path, 'samples_{}.pth'.format(step)))

                        del test_score
                        del all_samples

    def sample(self):
        """ Sampling function to use once the model is trained """

        # Get the model. -1 in the config file corresponds to the model 'checkpoint.pth'
        if self.config.sampling.ckpt_id == -1:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'), map_location=self.config.device)
        else:
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{self.config.sampling.ckpt_id}.pth'),
                                map_location=self.config.device)

        score = get_model(self.config)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0], strict=True)
        
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(score)

        # Get the noise sequence
        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()

        score.eval()

        if not self.config.sampling.fid:
            # Generate pictures meant to be seen
            
            # Starting point
            init_samples = torch.rand(self.config.sampling.batch_size, self.config.data.channels,
                                        self.config.data.image_size, self.config.data.image_size,
                                        device=self.config.device)

            # Annealed Langevin dynamics
            all_samples = anneal_Langevin_dynamics(init_samples, score, sigmas,
                                                    self.config.sampling.n_steps_each,
                                                    self.config.sampling.step_lr, verbose=True,
                                                    final_only=self.config.sampling.final_only,
                                                    denoise=self.config.sampling.denoise)

            if not self.config.sampling.final_only:
                # Generate and save intermediate (noisy) pictures
                for i, sample in tqdm.tqdm(enumerate(all_samples), total=len(all_samples),
                                            desc="saving image samples"):
                    sample = sample.view(sample.shape[0], self.config.data.channels,
                                            self.config.data.image_size,
                                            self.config.data.image_size)

                    sample = torch.clamp(sample, 0.0, 1.0)

                    image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                    save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_{}.png'.format(i)))
                    torch.save(sample, os.path.join(self.args.image_folder, 'samples_{}.pth'.format(i)))
            else:
                # Generate and save only final pictures
                sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                self.config.data.image_size,
                                                self.config.data.image_size)

                sample = torch.clamp(sample, 0.0, 1.0)

                image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                save_image(image_grid, os.path.join(self.args.image_folder,
                                                    'image_grid_{}.png'.format(self.config.sampling.ckpt_id)))
                torch.save(sample, os.path.join(self.args.image_folder,
                                                'samples_{}.pth'.format(self.config.sampling.ckpt_id)))

        else:
            # Generate pictures meant to be used to compute FID for instance
            
            total_n_samples = self.config.sampling.num_samples4fid
            n_rounds = total_n_samples // self.config.sampling.batch_size

            img_id = 0
            for _ in tqdm.tqdm(range(n_rounds), desc='Generating image samples for FID/inception score evaluation'):
                samples = torch.rand(self.config.sampling.batch_size, self.config.data.channels,
                                        self.config.data.image_size,
                                        self.config.data.image_size, device=self.config.device)
                
                all_samples = anneal_Langevin_dynamics(samples, score, sigmas,
                                                       self.config.sampling.n_steps_each,
                                                       self.config.sampling.step_lr, verbose=False,
                                                       denoise=self.config.sampling.denoise)

                samples = all_samples[-1]
                for img in samples:
                    img = torch.clamp(img, 0.0, 1.0)

                    save_image(img, os.path.join(self.args.image_folder, 'image_{}.png'.format(img_id)))
                    img_id += 1
