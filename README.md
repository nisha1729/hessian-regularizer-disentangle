## Hessian regularizer for Disentanglement
This repository contains my original work on a novel Hessian regularizer for disentangling (HeRD) neural networks. This work was done as part of my master's thesis in Computer Science at Saarland University. 

## Features extracted by HeRD from different datasets
<img src="/assets/sweep.png?raw=true"> 

### How to use the code

- To train a model with HeRD: `python main.py --train --cfg "/path/to/config/file.yaml"`
- To generate reconstructed images while sweeping the latent variables: `python main.py --sweep --cfg "/path/to/config/file.yaml`

The model and datasets are defined in the config files under `"./configs/"`. We use different models for different datasets. The regularizer is independent of the model architecture since it depends only on the loss and the latent dimension.


This work contains two variants of HeRD:
1. universal HeRD that can be used for unsupervised disentanglement where the goal is to disentangle factors of variation in the data
2. class-specific HeRD that is aimed towards disentangling class-specific features

### Disentanglement in a nutshell
The latent space of autoencoders contains a compressed representation of the
data. That is, it encodes different factors of variation present in the data. However, in normal
autoencoders and VAEs, these features often occur in combination with each other resulting
in, what we call, an _entangled representation_. For better adversarial robustness, generalization
and interpretability, it is desirable to have the features disentangled. Disentanglement focuses
on achieving more structure in the latent space.
There are multiple definitions for disentanglement:
- According to [Bengio et al. [2012]](https://arxiv.org/abs/1206.5538), disentanglement is separating out the "distinct but informative factors" of variation present in the data.
- [Locatello et al. [2019]](https://arxiv.org/abs/1811.12359) defines a disentangled representation to be one where only one factor in the latent
representation changes when one factor of variation changes.

In other words, in a disentangled representation, the axes of the latent space align with the factors of variation. That is, we want the latent dimensions to align with the factors of variation [[Shen et al., 2020]](https://arxiv.org/abs/2010.02637). In GANs and VAEs, disentanglement is usually achieved by adding constraints to the loss function that force the latent representation to be disentangled (for example: $\beta$-VAE, $\beta$-TCVAE, Factor-VAE). In this work, we also follow a similar approach but we try to achieve a disentangled representation for a deterministic autoencoder. HeRD can be easily extended to classifiers as well.
