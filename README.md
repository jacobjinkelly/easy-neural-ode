# Learning Differential Equations that are Fast to Solve
Code for the paper "Learning Differential Equations that are Fast to Solve"

## Requirements

### JAX

Follow installation instructions [here](https://github.com/google/jax).

### Haiku
Follow installation instructions [here](https://github.com/deepmind/dm-haiku).

### Tensorflow Datasets
For using the MNIST dataset, install `tensorflow-datasets` with
```
pip install tensorflow-datasets
```

## Usage
Different scripts are provided for each task and dataset.

### MNIST Classification

### Latent ODEs

### FFJORD (Tabular)

### FFJORD (MNIST)

## Datasets

### MNIST
`tensorflow-datasets` (instructions for installing above) will download the data when called from the training script.

### Physionet
The file `physionet_data.py`, adapted from [Latent ODEs for Irregularly-Sampled Time Series](https://github.com/YuliaRubanova/latent_ode) will download and process the data when called from the training script. 

### Tabular (FFJORD)
Data must be downloaded following instructions from [gpapamak/maf](https://github.com/gpapamak/maf) and placed in `data/`. Only `MINIBOONE` is needed for experiments in the paper.

Code in `datasets/`, adapted from [Free-form Jacobian of Reversible Dynamics (FFJORD)](https://github.com/rtqichen/ffjord), will create an interface for the `MINIBOONE` dataset once it's downloaded. 
It is called from the training script.

## Acknowledgements

Code in `lib` is modified from [google/jax](https://github.com/google/jax) under the [license](https://github.com/google/jax/blob/master/LICENSE).
