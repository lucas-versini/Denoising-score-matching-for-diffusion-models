import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, OxfordIIITPet

import random
from torch.utils.data import Dataset
from PIL import Image

""" This script contains functions to create the datasets """

class FilteredMNIST(Dataset):
    """
    Class to handle an unbalanced MNIST dataset.
    """
    def __init__(self, root, train = True, transform = None, target_label = 5, reduction_ratio = 0.35):
        # target_label is the digit whose proportion is to be reduced; reduction_ratio is the proportion of images we keep for this digit.
        self.mnist = MNIST(root=root, train = train, download = True)
        self.transform = transform

        # Separate images and labels
        data, targets = self.mnist.data, self.mnist.targets

        # Filter the dataset for the target class
        target_indices = (targets == target_label).nonzero(as_tuple = True)[0]
        keep_target_count = int(len(target_indices) * reduction_ratio)
        keep_target_indices = random.sample(target_indices.tolist(), keep_target_count)

        # Keep the other classes
        non_target_indices = (targets != target_label).nonzero(as_tuple=True)[0]

        # Combine indices and create the filtered dataset
        selected_indices = torch.cat((non_target_indices, torch.tensor(keep_target_indices)))
        self.data = data[selected_indices]
        self.targets = targets[selected_indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform:
            img = self.transform(img)

        return img, target

def get_dataset(args, config):
    """ Given the configuration, returns training and test datasets. """

    # Create transformations
    if config.data.random_flip is False:
        train_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
            ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.ToTensor()
            ])
        
    test_transform = transforms.Compose([
        transforms.Resize(config.data.image_size),
        transforms.ToTensor()
        ])

    # Create the datasets
    if config.data.dataset == 'CIFAR10':
        train_dataset = CIFAR10(os.path.join(args.exp, 'datasets', 'cifar10'),
                          train = True, download = True,
                          transform = train_transform)
        test_dataset = CIFAR10(os.path.join(args.exp, 'datasets', 'cifar10_test'),
                               train = False, download = True,
                               transform = test_transform)
    elif config.data.dataset == 'MNIST':
        train_dataset = MNIST(os.path.join(args.exp, 'datasets', 'mnist'),
                        train = True, download = True,
                        transform = train_transform)
        test_dataset = MNIST(os.path.join(args.exp, 'datasets', 'mnist_test'),
                                train = False, download = True,
                                transform = test_transform)
    elif config.data.dataset == 'FilteredMNIST':
        train_dataset = FilteredMNIST(os.path.join(args.exp, 'datasets', 'mnist'),
                                      train = True,
                                      transform = train_transform)
        test_dataset = MNIST(os.path.join(args.exp, 'datasets', 'mnist_test'),
                                train = False, download = True,
                                transform = test_transform)
    elif config.data.dataset == "OxfordPets":
        train_dataset = OxfordIIITPet(os.path.join(args.exp, 'datasets', 'oxford_iiit_pet'),
                                      split = 'trainval', download = True,
                                      transform = train_transform)
        test_dataset = OxfordIIITPet(os.path.join(args.exp, 'datasets', 'oxford_iiit_pet'),
                                    split = 'test', download = True,
                                    transform = test_transform)
    else:
        raise ValueError('Dataset not recognized')
    
    return train_dataset, test_dataset
