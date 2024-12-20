# Generative Modeling by Estimating Gradients of the Data Distribution

The content of this folder is adapted from the following GitHub repositories:

- [NCSN GitHub Repository](https://github.com/ermongroup/ncsn)  
- [NCSNv2 GitHub Repository](https://github.com/ermongroup/ncsnv2)  

These repositories implement the article: **[Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600)**  by Yang Song and Stefano Ermon.

This part of the project was primarily handled by Lucas Versini, who wrote most of the modifications for the experiments.
All three students contributed ideas for experiments and datasets. However, due to computational constraints, Lucas Versini conducted the majority of the experiments on MNIST and CIFAR-10.

---

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Folder structure

The folder names are fairly self-explanatory, and many scripts are similar to those in the original repositories. Here are some of the modifications we made:

- `models/__init__.py`:
Implemented additional schedulers for the sigma parameter.

- `datasets/__init__.py`:
Added support for the OxfordIIITPet dataset, and for the unbalanced MNIST dataset.

- `FID.py`:
Added a script to compute the FID score for pre-generated images.

- `config` folder:
Contains several configuration files for different experiments. More details are provided at the beginning of each file.

- `notebook` folder:
Contains a notebook to train a model using Denoising Score Matching. This notebook is a simplified version of the rest of the code. Its main goal is to simplify the code to make it understandable more easily, but it is much less efficient than the rest of the code.

## How to use

To train a model on the MNIST dataset, run:

```bash
python main.py --config MNIST.yml --doc MNIST
```

Once the model has been trained, you can generate samples by running:

```bash
python main.py --sample --config MNIST.yml -i MNIST
```

## Computing the FID score

To compute the FID score:

- First ensure you have generated samples by running the previous commands.

- Download the images from the target dataset in image format.

- Modify `relative_path_to_folder1` and `relative_path_to_folder2` in FID.py.

- Run:
```bash
python FID.py
```
