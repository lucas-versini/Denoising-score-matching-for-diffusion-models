import argparse
import yaml
import sys
import os
import torch
import numpy as np
from runners import *

import logging

import traceback
import shutil
import copy

import os

# To train a model:
# python main.py --config cifar10.yml --doc CIFAR_results
# To sample from a model:
# python main.py --sample --config cifar10.yml --doc CIFAR_results

def parse_args_and_config():
    parser = argparse.ArgumentParser(description = globals()['__doc__'])

    parser.add_argument('--config', type = str, required = True,  help = "Path to config file")
    parser.add_argument('--doc', type = str, required = True, help = "Name of the log folder")

    parser.add_argument('--sample', action = 'store_true', help = "Sample from the model")
    parser.add_argument('--resume_training', action = 'store_true', help = "Resume training")

    parser.add_argument('--seed', type = int, default = 42, help = "Random seed")
    parser.add_argument('--exp', type = str, default = 'exp', help = "Folder to save outputs")

    parser.add_argument('--comment', type = str, default = '', help = 'A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('-i', '--image_folder', type = str, default = 'images', help = "The folder name of samples")

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, 'logs', args.doc)

    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.load(f, Loader = yaml.FullLoader)
    new_config = dict2namespace(config)

    if not args.sample:
        if not args.resume_training:
            if os.path.exists(args.log_path):
                overwrite = False
                response = input("Folder already exists. Overwrite? (Y/N)")
                if response.upper() == 'Y':
                    overwrite = True

                if overwrite:
                    shutil.rmtree(args.log_path)
                    os.makedirs(args.log_path)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)

            with open(os.path.join(args.log_path, 'config.yml'), 'w') as f:
                yaml.dump(new_config, f, default_flow_style=False)

        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, 'stdout.txt'))
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

        os.makedirs(os.path.join(args.exp, 'image_samples'), exist_ok=True)
        args.image_folder = os.path.join(args.exp, 'image_samples', args.image_folder)
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            overwrite = False
            response = input("Image folder already exists. Overwrite? (Y/N)")
            if response.upper() == 'Y':
                overwrite = True

            if overwrite:
                shutil.rmtree(args.image_folder)
                os.makedirs(args.image_folder)
            else:
                print("Output image folder exists. Program halted.")
                sys.exit(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: {}".format(device))
    logging.info("Using device: {}".format(device))
    new_config.device = device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")
    print(">" * 80)
    config_dict = copy.copy(vars(config))
    print(yaml.dump(config_dict, default_flow_style=False))
    print("<" * 80)

    try:
        runner = NCSNRunner(args, config)
        if args.sample:
            runner.sample()
        else:
            runner.train()
    except:
        logging.error(traceback.format_exc())

    return 0


if __name__ == '__main__':
    sys.exit(main())
