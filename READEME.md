# AMAGOLD: Amortized Metropolis Adjustment for Efficient Stochastic Gradient MCMC
This repository contains code for [AMAGOLD: Amortized Metropolis Adjustment for Efficient Stochastic Gradient MCMC], AISTATS, 2020

# Dependencies
* Python 2.7
* [PyTorch 1.2.0](https://pytorch.org/)

## Double-well Potential Density
To generate samples from the double-well potential by SGHMC and AMAGOLD, please run `simulation/doublewell_sghmc.m` and `simulation/doublewell_amagold.m` respectively.

To visualize the results, please use ipython notebook with Python to open the file `simulation/dooublewell.ipynb`. Cached results from our runs are included.

## Bayesian Neural Networks on MNIST
To get SGD initialization, run
```
python bnn/train_SGD.py
```
To run SGHMC and AMAGOLD:
```
python bnn/train_sghmc.py
```
and
```
python bnn/train_amagold.py
```
With `lr = 0.0005` and `a = 1e-5`, SGHMC will diverge after about 500 epochs.

