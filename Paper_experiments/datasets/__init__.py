import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, OxfordIIITPet

def get_dataset(args, config):
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
