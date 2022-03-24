This page serves as the official repository for the `tvp-gr4j` project aimed at developing dynamic (i.e. with time-varying parameters) rainfall-runoff models for change and trend detection via NumPyro and PyMC3.

To reproduce the findings contained in *A comparison of novel dynamic priors for Bayesian estimation of time-varying parameters in rainfall-runoff modeling via Hamiltonian Monte Carlo* by Christopher Krapu and Mark Borsuk (under review), execute the code blocks in the IPython notebooks as follows:

1. To regenerate MCMC traces for the simulation study comparing multiple prior distributions, run the notebook titled "simulation-study.ipynb". This notebook will save several files to disk including a series of MCMC traces stored as pickled MultiTrace objects from PyMC3

2. For reproduction of the figures from the simulation study, run the notebook "simulation-study-figures.ipynb".

3. To execute the real-world case study with 20 year daily streamflow records, run the notebook "jax-real-data.ipynb".

