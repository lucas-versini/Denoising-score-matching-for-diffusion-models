# Generative Modeling by Estimating Gradients of the Data Distribution

The content of this folder is adapted from the following GitHub repositories:

- [NCSN GitHub Repository](https://github.com/ermongroup/ncsn)  
- [NCSNv2 GitHub Repository](https://github.com/ermongroup/ncsnv2)  

These repositories implement the article:  
**[Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600)**  
by Yang Song and Stefano Ermon.

---

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt```

## Folder structure

The folder names are fairly self-explanatory, and many scripts are similar to those in the original repositories. Here are some of the key modifications we made:

- models/__init__.py
Implemented additional schedulers for the sigma parameter.

- datasets/__init__.py
Added support for the OxfordIIITPet dataset.

- FID.py
Added a script to compute the FID score for pre-generated images.

- config folder
Contains several configuration files for different experiments. More details are provided at the beginning of each file.

