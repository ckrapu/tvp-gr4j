![animation_v3](https://user-images.githubusercontent.com/10506722/160662562-00b72c03-b391-41f0-968a-378630e4223c.gif)

## Overview
This page serves as the official repository for the `tvp-gr4j` project aimed at developing dynamic (i.e. with time-varying parameters) rainfall-runoff models for change and trend detection via NumPyro and PyMC3.

## Acknowledgements
About 1/2 of the Jax implementation of GR4J in this project was copy/pasted from [Frederik Kratzert's Numpy implementation](https://github.com/kratzert/RRMPG). For Python programmers interested in simply hydrology models, I strongly recommend checking it out.

## Why should you care?
The environment definitely isn't static - it's always changing. The best methods for change detection outside of hydrology make use of Bayesian models fit using MCMC. Until now, it was impossible to use standard MCMC techniques for problems with >100 parameters like Hamiltonian Monte Carlo with hydrology models  because they required backpropagating gradients through the hydrology model. Now we can do that! This connects the hydro change detection literature to the broader research community by giving us all the MCMC-based tools for analysis that we'd been lacking until now. 

You may ask yourself the following questions:

- **But what about data assimilation methods?** *Yes, these can work, but when MCMC with a well-defined generative model is compared to data assilimation, the former is almost always easier and less brittle from a workflow perspective and I say this as someone who has implemented filtering methods many, many times. With PyMC, Stan, or Pyro, all you need to do is write the forward model - no matrix equations for variational data assimilation or extended Kalman filters*
- **Why don't we use a neural network?** *Well, you could replace parts of the hydrology model with a NN. The beauty of Hamiltonian Monte Carlo is that it doesn't care what the forward model is, it just works! Also, the methods shown here work even when 99% of the data is thrown out. Fun fact - [HMC was partially reintroduced to statistics by researchers interested in Bayesian neural networks](https://arxiv.org/pdf/1206.1901.pdf).*
- **Does it work on other hydrology models?** *It should, and I would love to try this with more models. Ping me at ckrapu@gmail.com if you want to collaborate on this.*


## Reproduction
To reproduce the findings contained in *A comparison of novel dynamic priors for Bayesian estimation of time-varying parameters in rainfall-runoff modeling via Hamiltonian Monte Carlo* by Christopher Krapu and Mark Borsuk (under review), execute the code blocks in the IPython notebooks as follows:

1. To regenerate MCMC traces for the simulation study comparing multiple prior distributions, run the notebook titled "simulation-study.ipynb". This notebook will save several files to disk including a series of MCMC traces stored as pickled MultiTrace objects from PyMC3

2. For reproduction of the figures from the simulation study, run the notebook "simulation-study-figures.ipynb".

3. To execute the real-world case study with 20 year daily streamflow records, run the notebook "jax-real-data.ipynb".

## Dependencies
The main package dependencies for this repository include Theano/PyMC3 and Jax/NumPyro. Running `pip install -f requirements.txt` from within this directory should obtain the correct packages. Using a virtual environment is highly recommended.

