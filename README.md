# Truly Multivariate Structured Additive Distributional Regression

## Abstract
Generalized additive models for location, scale and shape (GAMLSS) are a popular extension to mean regression models where each parameter of an arbitrary parametric distribution is modeled through covariates. While such models have been developed for univariate and bivariate responses, the truly multivariate case remains extremely challenging for both computational and theoretical reasons. Alternative approaches to GAMLSS may allow for higher-dimensional response vectors to be modeled jointly but often assume a fixed dependence structure not depending on covariates or are limited with respect to modeling flexibility or computational aspects. We contribute to this gap in the literature and propose a truly multivariate distributional model, which allows one to benefit from the flexibility of GAMLSS even when the response has dimension larger than two or three. Building on copula regression, we model the dependence structure of the response through a Gaussian copula, while the marginal distributions can vary across components. Our model is highly parameterized but estimation becomes feasible with Bayesian inference employing shrinkage priors. We demonstrate the competitiveness of our approach in a simulation study and illustrate how it complements existing approaches along the examples of childhood malnutrition and a yet unexplored dataset on traffic detection in Berlin. 

The paper can be found at https://doi.org/10.1080/10618600.2024.2434181

## Files in this repository

1. Benchmark methods

1a. bamlss.R contains the code for the benchmark method labelled bamlss.

1b. mctm.R contains the code for the benchmark method labelled mctm. 

1c. mvngam.R contains the code for the benchmark method labelled mvngam.

1d. vgam.R contains the code for the benchmark method labelled vgam.

2. multgamlss code

2a. multgamlss.py contains code for the MCMC algorithm as described in Section 2.4. Further documentation also for the input and output arguments 
required to call the different functions can be found in the corresponding headers.

2b. b_splines.py constructs different design matrices and corresponding penalty matrices including the B-Spline basis used to estimate functional effects.

3. Data generating process

3a. generate_data.py generates the data sets analyzed in Section 3 of the paper.
