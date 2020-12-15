# Learning Differential Equations that are Easy to Solve
Code for the paper:

> Jacob Kelly*, Jesse Bettencourt*, Matthew James Johnson, David Duvenaud. "Learning Differential Equations that are Easy to Solve." _Neural Information Processing Systems_ (2020).
> [[arxiv]](https://arxiv.org/abs/2007.04504) [[bibtex]](#bibtex)

\*Equal Contribution

<p align="center">
<img align="middle" src="./assets/anim.gif" width="500" />
</p>

Includes JAX implementations of the following models:
- [Neural ODEs](https://arxiv.org/abs/1806.07366) for classification
- [Latent ODEs](https://arxiv.org/abs/1907.03907) for time series
- [FFJORD](https://arxiv.org/abs/1810.01367) for density estimation

Includes JAX implementations of the following adaptive-stepping numerical solvers:
- Heun-Euler `heun` (2nd order)
- Fehlberg (RK1(2)) `fehlberg` (2nd order)
- Bogacki-Shampine `bosh` (3rd order)
- Cash-Karp `cash_karp` (4th order)
- Fehlberg `rk_fehlberg` (4th order)
- Owrenzen `owrenzen` (4th order)
- Dormand-Prince `dopri` (5th order)
- Owrenzen `owrenzen5` (5th order)
- Tanyam `tanyam` (7th order)
- Adams `adams` (adaptive order)
- RK4 `rk4` (4th order, fixed step-size)

## Requirements

### JAX
Follow installation instructions [here](https://github.com/google/jax#installation).

### Haiku
Follow installation instructions [here](https://github.com/deepmind/dm-haiku#installation).

### Tensorflow Datasets
For using the MNIST dataset, follow installation instructions [here](https://www.tensorflow.org/datasets/overview).

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

Several numerical solvers were adapted from [torchdiffeq](https://github.com/rtqichen/torchdiffeq) and [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl).

## BibTeX

```
@inproceedings{kelly2020easynode,
  title={Learning Differential Equations that are Easy to Solve},
  author={Kelly, Jacob and Bettencourt, Jesse and Johnson, Matthew James and Duvenaud, David},
  booktitle={Neural Information Processing Systems},
  year={2020},
  url={https://arxiv.org/abs/2007.04504}
}
```
