# Learning Differential Equations that are Easy to Solve
Code for the paper:

> Jacob Kelly*, Jesse Bettencourt*, Matthew James Johnson, David Duvenaud. "Learning Differential Equations that are Easy to Solve" _arXiv preprint_ (2020).
> [[arxiv]](https://arxiv.org/abs/2007.04504) [[bibtex]](#bibtex)

\*Equal Contribution

Includes JAX implementations of:
- [Neural ODEs](https://arxiv.org/abs/1806.07366) for classification
- [Latent ODEs](https://arxiv.org/abs/1907.03907) for time series
- [FFJORD](https://arxiv.org/abs/1810.01367) for density estimation

## Requirements

### JAX
Follow installation instructions [here](https://github.com/google/jax#installation).

### Haiku
Follow installation instructions [here](https://github.com/deepmind/dm-haiku#installation).

### Tensorflow Datasets
For using the MNIST dataset, follow installation instructions [here](https://github.com/tensorflow/datasets#installation).

## Usage
Different scripts are provided for each task and dataset.

### MNIST Classification

```
python mnist.py --reg r3 --lam 6e-5
```

### Latent ODEs

```
python latent_ode.py --reg r3 --lam 1e-2
```

### FFJORD (Tabular)

```
python ffjord_tabular.py --reg r2 --lam 1e-2
```

### FFJORD (MNIST)

```
python ffjord_mnist.py --reg r2 --lam 3e-4
```

## Datasets

### MNIST
`tensorflow-datasets` (instructions for installing above) will download the data when called from the training script.

### Physionet
The file `physionet_data.py`, adapted from [Latent ODEs for Irregularly-Sampled Time Series](https://github.com/YuliaRubanova/latent_ode) will download and process the data when called from the training script. A preprocessed version is available in [releases](https://github.com/jacobjinkelly/easy-neural-ode/releases/tag/1.0.0).

### Tabular (FFJORD)
Data must be downloaded following instructions from [gpapamak/maf](https://github.com/gpapamak/maf) and placed in `data/`. Only `MINIBOONE` is needed for experiments in the paper.

Code in `datasets/`, adapted from [Free-form Jacobian of Reversible Dynamics (FFJORD)](https://github.com/rtqichen/ffjord), will create an interface for the `MINIBOONE` dataset once it's downloaded. 
It is called from the training script.

## Acknowledgements

Code in `lib` is modified from [google/jax](https://github.com/google/jax) under the [license](https://github.com/google/jax/blob/master/LICENSE).

## BibTeX

```
@article{kelly2020easynode,
  title={Learning Differential Equations that are Easy to Solve},
  author={Kelly, Jacob and Bettencourt, Jesse and Johnson, Matthew James and Duvenaud, David},
  journal={arXiv preprint arXiv:2007.04504},
  year={2020}
}
```
