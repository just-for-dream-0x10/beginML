# Probabilistic Programming in PyTorch (with Pyro and NumPyro as Examples)

Similar to TensorFlow Probability (TFP), the PyTorch ecosystem offers powerful libraries for probabilistic programming, Bayesian inference, and statistical analysis. The primary libraries are **Pyro** and **NumPyro**. Pyro, developed by Uber AI Labs, is built directly on top of PyTorch. NumPyro, a sister project of Pyro, uses NumPy as its primary backend and leverages JAX for just-in-time compilation and hardware acceleration, often providing superior performance. Additionally, **GPyTorch** is a specialized library for Gaussian process modeling, while **BoTorch**, built on GPyTorch and PyTorch, focuses on Bayesian optimization.

This document follows the structure of TFP-related content to organize and summarize these technologies in the PyTorch ecosystem.

## 1. Fundamentals of Probabilistic Programming in PyTorch

The core idea of Pyro/NumPyro aligns with TFP: explicitly representing and handling uncertainty through probabilistic models. Probabilistic programming enables users to define the data-generating process (including randomness) and then infer unknown parameters or latent variables from observed data.

### 1.1 Basic Concepts of Probabilistic Programming

These foundational concepts are universal in PyTorch’s probabilistic programming frameworks:

* **Probability Distribution**: A mathematical function describing all possible values of a random variable and their corresponding probabilities. In Pyro, these distributions are in the `pyro.distributions` module; in NumPyro, they are in the `numpyro.distributions` module.
* **Random Variable**: A variable whose value is determined by random phenomena. In Pyro/NumPyro, random variables are declared using `pyro.sample("name", distribution)` or `numpyro.sample("name", distribution)`, where "name" is the variable’s name and `distribution` is its probability distribution.
* **Joint Distribution**: The probability of multiple random variables taking specific values simultaneously. In Pyro/NumPyro, a probabilistic model (typically a Python function with multiple `sample` statements) implicitly defines the joint distribution of all random variables.
* **Conditional Distribution**: The probability distribution of a random variable given the values of one or more other random variables. Dependencies between variables in the model naturally form conditional distributions.
* **Prior Distribution**: The initial belief or assumption about model parameters before observing data, expressed as a probability distribution.
* **Posterior Distribution**: The updated parameter distribution after observing data, obtained via Bayes’ theorem, combining prior beliefs with data-driven information.
* **Likelihood Function**: The probability of observing the current dataset given specific values of model parameters, typically denoted as \( P(D|\theta) \), where \( D \) is the data and \( \theta \) is the parameter.

### 1.2 Bayesian Inference

Bayesian inference is the cornerstone of probabilistic programming, grounded in Bayes’ theorem:
\[
P(\theta|D) = \frac{P(D|\theta) \times P(\theta)}{P(D)}
\]
where:
* \( P(\theta|D) \): **Posterior probability** – the probability of parameters \( \theta \) given observed data \( D \).
* \( P(D|\theta) \): **Likelihood function** – the probability of observing data \( D \) given parameters \( \theta \).
* \( P(\theta) \): **Prior probability** – the initial belief about parameters \( \theta \).
* \( P(D) \): **Marginal likelihood** (or evidence) – the total probability of observing data \( D \), typically a normalizing constant.

Pyro and NumPyro’s primary role is to help users define models (i.e., prior \( P(\theta) \) and likelihood \( P(D|\theta) \)) and efficiently compute or approximate the posterior distribution \( P(\theta|D) \).

## 2. Core Components of Pyro/NumPyro

### 2.1 Probability Distributions (`pyro.distributions` / `numpyro.distributions`)

Pyro and NumPyro provide extensive libraries of probability distributions, similar in functionality to `tfp.distributions`. These distributions are built on PyTorch (Pyro) or JAX (NumPyro) tensor operations, seamlessly integrating into their respective computational graphs and supporting automatic differentiation.

```python
# Pyro Example
import torch
import pyro
import pyro.distributions as dist

# Create a normal distribution (mean=0, std=1)
normal_dist_pyro = dist.Normal(loc=0., scale=1.)

# Sample from the distribution
samples_pyro = normal_dist_pyro.sample(sample_shape=torch.Size())  # Generate 1000 samples

# Compute probability density (for continuous distributions)
# P(X = 0.5) - Note: For continuous distributions, the probability at a single point is 0; this is the density
prob_density_pyro = normal_dist_pyro.log_prob(torch.tensor(0.5)).exp()

# Compute log probability density
log_prob_density_pyro = normal_dist_pyro.log_prob(torch.tensor(0.5))  # log(P(X = 0.5))

# Compute cumulative distribution function (CDF)
# P(X <= 0.5)
cdf_pyro = normal_dist_pyro.cdf(torch.tensor(0.5))

# Compute entropy of the distribution
entropy_pyro = normal_dist_pyro.entropy()

print(f"Pyro Normal(0,1) - Sample mean: {samples_pyro.mean():.4f}")
print(f"Pyro Normal(0,1) - log_prob(0.5): {log_prob_density_pyro:.4f}")
print(f"Pyro Normal(0,1) - CDF(0.5): {cdf_pyro:.4f}")
print(f"Pyro Normal(0,1) - Entropy: {entropy_pyro:.4f}")

# NumPyro Example (very similar syntax)
import numpyro
import numpyro.distributions as dist_np
import jax.numpy as jnp
from jax import random

key = random.PRNGKey(0)  # JAX requires an explicit PRNG key
normal_dist_numpyro = dist_np.Normal(loc=0., scale=1.)
samples_numpyro = normal_dist_numpyro.sample(key, (1000,))
log_prob_density_numpyro = normal_dist_numpyro.log_prob(0.5)

print(f"\nNumPyro Normal(0,1) - Sample mean: {samples_numpyro.mean():.4f}")
print(f"NumPyro Normal(0,1) - log_prob(0.5): {log_prob_density_numpyro:.4f}")
```

Common distribution types include:

- Continuous Distributions: Normal, Beta, Gamma, StudentT, Uniform, Exponential, Laplace, LogNormal, Chi2, etc.
- Discrete Distributions: Bernoulli, Binomial, Categorical, Poisson, Geometric, NegativeBinomial, etc.
- Multivariate Distributions: MultivariateNormal, Dirichlet, Multinomial, Wishart, LKJCholesky (for covariance matrix priors), etc.
- Transformed Distributions: Distributions obtained by applying invertible transformations to base distributions, e.g., TransformedDistribution.
- Mixture Distributions: Such as MixtureSameFamily (mixing distributions of the same type) or Mixture (more general mixtures).

### 2.2 Probabilistic Models and Joint Distributions
In Pyro and NumPyro, probabilistic models are typically defined as Python functions. These functions use pyro.sample (or numpyro.sample) statements to declare random variables (including latent variables and parameters) and their probability distributions. The sequence of sample statements and their dependencies (e.g., a distribution parameter depending on a previous sample result) implicitly define the joint probability distribution of all random variables in the model.

```python
# Pyro Model Example: Simple Linear Regression
# x_data and y_data are observed data
def linear_regression_model_pyro(x_data, y_data):
    # Prior distributions
    # weight ~ Normal(0, 1)
    weight = pyro.sample("weight", dist.Normal(0., 1.))
    # bias ~ Normal(0, 5)
    bias = pyro.sample("bias", dist.Normal(0., 5.))
    # sigma (observation noise std) ~ Uniform(0, 10)
    sigma = pyro.sample("sigma", dist.Uniform(0., 10.))

    # Deterministic part of the model (mean function)
    mean_prediction = weight * x_data + bias

    # Likelihood (observation model)
    # Use pyro.plate to indicate conditional independence of data points
    with pyro.plate("data_plate", size=len(y_data)):
        # y_observed ~ Normal(mean_prediction, sigma)
        pyro.sample("obs", dist.Normal(mean_prediction, sigma), obs=y_data)

# NumPyro Model Example (very similar structure)
def linear_regression_model_numpyro(x_data, y_data=None):  # y_data=None allows running only the prior during sampling
    weight = numpyro.sample("weight", dist_np.Normal(0., 1.))
    bias = numpyro.sample("bias", dist_np.Normal(0., 5.))
    sigma = numpyro.sample("sigma", dist_np.Uniform(0., 10.))

    mean_prediction = weight * x_data + bias

    # In NumPyro, plate is implemented via numpyro.plate
    with numpyro.plate("data_plate", size=x_data.shape):  # Assumes x_data and y_data have the same first dimension
        numpyro.sample("obs", dist_np.Normal(mean_prediction, sigma), obs=y_data)
```
Unlike TFP’s explicit JointDistributionCoroutine or JointDistributionSequential, Pyro/NumPyro models implicitly define joint distributions through their execution flow and sample statements. The pyro.plate (or numpyro.plate) context manager is crucial for declaring conditional independence among data subsets, essential for efficiently handling batched data.

### 2.3 Variational Inference (Pyro: pyro.infer.SVI, NumPyro: numpyro.infer.SVI)

Variational Inference (VI) is a core inference algorithm in Pyro and NumPyro. It transforms posterior inference into an optimization problem: finding a distribution ( q(\theta;\phi) ) from a simple, parameterized family (called the variational family or guide) that is as close as possible to the true posterior ( P(\theta|D) ). This “closeness” is typically measured by minimizing the Kullback-Leibler (KL) divergence, which is equivalent to maximizing the Evidence Lower Bound (ELBO).

```python
# Pyro Variational Inference Example (continuing the linear regression model)
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam  # Pyro’s optimizer, or use torch.optim

# 1. Define the model (as above: linear_regression_model_pyro)

# 2. Define the guide function (variational posterior q(θ; φ))
# The guide should have the same parameter signature as the model (except for observed values)
# and include pyro.sample statements corresponding to each non-observed pyro.sample in the model,
# but parameterized by learnable variational parameters via pyro.param.
def linear_regression_guide_pyro(x_data, y_data):  # y_data often unused in guide but included for signature consistency
    # Variational parameters (φ): each parameter has a mean (loc) and log-standard deviation (scale_log)
    w_loc = pyro.param("w_loc", torch.tensor(0.))
    w_scale_log = pyro.param("w_scale_log", torch.tensor(0.))
    w_scale = torch.exp(w_scale_log)  # Ensure scale is positive

    b_loc = pyro.param("b_loc", torch.tensor(0.))
    b_scale_log = pyro.param("b_scale_log", torch.tensor(0.))
    b_scale = torch.exp(b_scale_log)

    sigma_loc_log = pyro.param("sigma_loc_log", torch.tensor(0.))  # Sigma is typically positive, use LogNormal or constraints
    sigma_scale = pyro.param("sigma_scale", torch.tensor(1.0), constraint=dist.constraints.positive)

    # Variational distributions
    pyro.sample("weight", dist.Normal(w_loc, w_scale))
    pyro.sample("bias", dist.Normal(b_loc, b_scale))
    # For constrained parameters like sigma (must be positive), use TransformedDistribution
    sigma_param_unconstrained = pyro.param("sigma_param_unconstrained", torch.tensor(1.))
    sigma = pyro.sample("sigma", dist.TransformedDistribution(
        dist.Normal(sigma_param_unconstrained, torch.tensor(0.1)),
        dist.transforms.ExpTransform()  # Ensure sigma is positive
    ))

# Assume x_data_tensor, y_data_tensor are defined
pyro.clear_param_store()  # Clear any previous parameters from the global store

# Set up the optimizer
adam_params = {"lr": 0.005}  # Learning rate
optimizer = Adam(adam_params)

# Create SVI object
svi = SVI(model=linear_regression_model_pyro,
          guide=linear_regression_guide_pyro,
          optim=optimizer,
          loss=Trace_ELBO())  # Trace_ELBO is a general ELBO computation method

# Training loop
num_epochs = 2000
for epoch in range(num_epochs):
    loss = svi.step(x_data_tensor, y_data_tensor)  # Computes gradient of loss and updates parameters
    if epoch % 100 == 0:
        print(f"Epoch {epoch} : ELBO = {-loss:.4f}")  # SVI returns -ELBO

# NumPyro Variational Inference Example (similar structure)
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam as AdamNumPyro  # NumPyro’s optimizer

# Define NumPyro model and guide (e.g., linear_regression_model_numpyro and corresponding guide_numpyro)
optimizer_numpyro = AdamNumPyro(step_size=0.005)
svi_numpyro = SVI(model=linear_regression_model_numpyro,
                  guide=linear_regression_guide_numpyro,
                  optim=optimizer_numpyro,
                  loss=Trace_ELBO())

# svi_result = svi_numpyro.run(random.PRNGKey(1), num_steps=2000, x_data=x_data_np, y_data=y_data_np)
# params_numpyro = svi_result.params  # Get learned variational parameters
# w_loc_learned = params_numpyro["w_loc"]
```

### 2.4 Markov Chain Monte Carlo (Pyro: pyro.infer.mcmc, NumPyro: numpyro.infer.mcmc)
Markov Chain Monte Carlo (MCMC) methods are another key technique for posterior inference. They construct a Markov chain whose stationary distribution is the target posterior ( P(\theta|D) ), then draw samples from this chain to approximate the posterior. NumPyro’s MCMC implementation, leveraging JAX’s backend, is particularly efficient, especially with its No-U-Turn Sampler (NUTS) algorithm.

```python

# NumPyro MCMC Example (NumPyro is typically preferred for MCMC due to performance)
import numpyro
import numpyro.distributions as dist_np
from numpyro.infer import MCMC, NUTS
from jax import random

# Assume linear_regression_model_numpyro(x_data, y_data) is defined
key_mcmc = random.PRNGKey(0)

# Initialize NUTS kernel
nuts_kernel = NUTS(model=linear_regression_model_numpyro)

# Create MCMC object
mcmc = MCMC(kernel=nuts_kernel,
            num_warmup=500,       # "Burn-in" or "warm-up" steps
            num_samples=1000,     # Actual sampling steps
            num_chains=1,         # Number of Markov chains (running multiple chains is good practice)
            progress_bar=True)    # Show progress bar

# Run MCMC
# Assume x_data_np and y_data_np are JAX/NumPy arrays
mcmc.run(key_mcmc, x_data=x_data_np, y_data=y_data_np)

# Print summary of sampling results (mean, std, quantiles, etc.)
mcmc.print_summary()

# Get posterior samples
posterior_samples_numpyro = mcmc.get_samples()
# posterior_samples_numpyro is a dictionary with keys as parameter names (e.g., "weight", "bias", "sigma")

# Pyro MCMC Example (similar syntax but different performance and implementation)
from pyro.infer import MCMC, NUTS

# Assume linear_regression_model_pyro(x_data_tensor, y_data_tensor) is defined
nuts_kernel_pyro = NUTS(model=linear_regression_model_pyro)
mcmc_pyro = MCMC(kernel=nuts_kernel_pyro,
                 warmup_steps=500,
                 num_samples=1000,
                 num_chains=1)

mcmc_pyro.run(x_data_tensor, y_data_tensor)  # Pass model parameters

posterior_samples_pyro = mcmc_pyro.get_samples()
# Pyro’s MCMC results are also a dictionary
```

Hamiltonian Monte Carlo (HMC) and its adaptive variant, NUTS, are among the most advanced MCMC algorithms, particularly suited for high-dimensional continuous parameter spaces. They use gradient information from the target distribution to guide the sampling process, enabling more efficient exploration.


## 3. Practical Application Examples
Below, we briefly describe how to implement probabilistic models in the PyTorch ecosystem similar to those in TFP documentation.

### 3.1 Bayesian Linear Regression

Steps to implement Bayesian linear regression using Pyro/NumPyro:
 
1. Define the Model Function: As shown in linear_regression_model_pyro or linear_regression_model_numpyro, set prior distributions for weights (weight) and biases (bias) (e.g., Normal distributions), a prior for the observation noise standard deviation (sigma) (e.g., Uniform or Half-Normal), and define the likelihood (typically a Normal distribution with mean ( $ \text{weight} \times x + \text{bias} $ ) and standard deviation sigma).
2. Choose Inference Algorithm:
    - `Variational Inference (SVI)`: Define a guide function, specifying parameterized variational distributions (e.g., Normal distributions with learnable mean and standard deviation via pyro.param or numpyro.param) for each latent variable (weight, bias, sigma). Use the SVI object to learn these variational parameters by optimizing the ELBO.
    - `MCMC`: Use the MCMC object (typically with the NUTS kernel) to directly sample from the posterior distribution. 
3. Result Analysis
    - For VI, the learned variational parameters (e.g., w_loc, w_scale) directly describe the approximate posterior distribution.
    - For MCMC, the posterior sample set can be used to compute parameter means, medians, credible intervals (e.g., 95% credible interval), and plot posterior histograms or density plots.
    - Parameters can be sampled from the posterior (or approximate posterior) to make predictions for new ( x ) values, yielding a predictive distribution and quantifying prediction uncertainty.

Visualization can use matplotlib or seaborn, similar to TFP examples, by replacing TensorFlow/TFP tensor and distribution operations with PyTorch/Pyro or JAX/NumPy/NumPyro operations.

### 3.2 Bayesian Neural Networks (BNN)
In Pyro/NumPyro, constructing a Bayesian Neural Network (BNN) involves assigning prior distributions to the network’s weights and biases instead of treating them as fixed point estimates.

```python
# Pyro BNN Model Example (conceptual)
import torch.nn as nn
import pyro.nn as pynn  # Pyro’s module for building Bayesian neural networks

class BayesianNN(pynn.PyroModule):  # Inherits from PyroModule
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = pynn.PyroLinear(in_features, hidden_features)  # Bayesian linear layer
        self.fc1.weight = pynn.PyroSample(dist.Normal(0., 1.).expand([hidden_features, in_features]).to_event(2))
        self.fc1.bias = pynn.PyroSample(dist.Normal(0., 1.).expand([hidden_features]).to_event(1))

        self.fc2 = pynn.PyroLinear(hidden_features, out_features)
        self.fc2.weight = pynn.PyroSample(dist.Normal(0., 1.).expand([out_features, hidden_features]).to_event(2))
        self.fc2.bias = pynn.PyroSample(dist.Normal(0., 1.).expand([out_features]).to_event(1))

        self.relu = nn.ReLU()

    def forward(self, x, y=None):  # y is the observed value
        x = x.view(-1, self.fc1.in_features)
        x = self.relu(self.fc1(x))
        mu = self.fc2(x).squeeze()
        sigma = pyro.sample("sigma_obs", dist.Uniform(0., 1.))  # Observation noise

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
        return mu

# Corresponding Guide Function (e.g., mean-field variational family):
class BayesianNNGuide(pynn.PyroModule):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        # Define variational parameters and distributions for each parameter (typically Normal)
        self.fc1_weight_loc = nn.Parameter(torch.randn(hidden_features, in_features))
        self.fc1_weight_scale = nn.Parameter(torch.randn(hidden_features, in_features))
        # ... other parameters similarly ...
        self.sigma_obs_loc = nn.Parameter(torch.tensor(0.5))  # Example

    def forward(self, x, y=None):
        # Sample from variational distributions corresponding to PyroSample statements in the model
        # fc1_weight = pyro.sample("fc1.weight", dist.Normal(self.fc1_weight_loc, torch.exp(self.fc1_weight_scale)).to_event(2))
        # ...
        # sigma_obs = pyro.sample("sigma_obs", dist.LogNormal(self.sigma_obs_loc, torch.tensor(0.1)))  # Ensure positivity
        # Typically, the guide doesn’t need to execute the neural network’s forward pass; it only ensures all model parameters have corresponding variational samples.
        # A simpler approach is to use pyro.infer.autoguide, which auto-generates common guides like AutoDiagonalNormal.

model_bnn = BayesianNN(input_size, hidden_size, output_size)
guide_bnn = pyro.infer.autoguide.AutoDiagonalNormal(model_bnn)  # Auto-generate mean-field Normal guide
svi_bnn = SVI(model_bnn, guide_bnn, optimizer, loss=Trace_ELBO())
# Training loop...
```

`pynn.PyroModule, pynn.PyroLinear, and pynn.PyroSample` simplify converting standard PyTorch `nn.Module` into Bayesian versions. Inference can use SVI (with AutoGuide or manually defined guides) or MCMC.

### 3.3 Gaussian Process Regression (GPyTorch)
For Gaussian Processes (GPs), GPyTorch is the go-to library in the PyTorch ecosystem. It is modular, extensible, and leverages PyTorch’s GPU acceleration and automatic differentiation capabilities.

```python
# GPyTorch Example (conceptual)
import gpytorch

# Assume train_x_tensor and train_y_tensor are PyTorch tensors
class ExactGPModel(gpytorch.models.ExactGP):  # Exact GP model
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        # Define mean function (e.g., constant mean)
        self.mean_module = gpytorch.means.ConstantMean()
        # Define kernel function (e.g., RBF kernel, aka squared exponential kernel)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # Return multivariate normal distribution as GP’s prediction
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Initialize likelihood (e.g., Gaussian likelihood for observation noise)
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model_gp = ExactGPModel(train_x_tensor, train_y_tensor, likelihood)

# Train GP model (learn hyperparameters by optimizing marginal log likelihood)
model_gp.train()
likelihood.train()

# Use PyTorch optimizer
optimizer_gp = torch.optim.Adam(model_gp.parameters(), lr=0.1)  # model.parameters() includes kernel and mean function parameters

# "Loss" function is negative marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model_gp)

training_iter = 100
for i in range(training_iter):
    optimizer_gp.zero_grad()
    output = model_gp(train_x_tensor)  # Model forward pass, get predictive distribution
    loss = -mll(output, train_y_tensor)  # Compute negative marginal log likelihood
    loss.backward()  # Backpropagate to compute gradients
    if (i + 1) % 10 == 0:
        print(f"Iter {i+1}/{training_iter} - Loss: {loss.item():.3f} "
              f"lengthscale: {model_gp.covar_module.base_kernel.lengthscale.item():.3f} "
              f"noise: {likelihood.noise.item():.3f}")
    optimizer_gp.step()  # Update parameters

# Make predictions
model_gp.eval()
likelihood.eval()

# test_x_tensor is new input points
with torch.no_grad(), gpytorch.settings.fast_pred_var():  # Context manager for fast variance prediction
    observed_pred = likelihood(model_gp(test_x_tensor))  # likelihood() converts latent function predictions to observed predictions
    predictive_mean = observed_pred.mean
    lower_ci, upper_ci = observed_pred.confidence_region()  # Get confidence interval (typically 95%)
```

GPyTorch offers various mean functions, kernel functions (e.g., RBF, Matern, Periodic, and their combinations), likelihoods, and inference strategies (including exact GPs and sparse GP approximations for large datasets).

### Evaluation of Gaussian Process Regression Models
Evaluation metrics for GP models are similar to those described in TFP documentation:

- Prediction Accuracy Metrics:
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Error (MAE)
    - Coefficient of Determination (( $R^2$ ))
-  Uncertainty Quantification Metrics:
    - Prediction Interval Coverage Probability (PICP): The proportion of true observations falling within the prediction interval (e.g., 95% confidence interval). Ideally, it should be close to the nominal coverage rate (e.g., 95%).
    - Mean Prediction Interval Width (MPIW): The average width of prediction intervals; narrower is better (while maintaining good coverage).
    - Negative Log Predictive Density (NLPD): Evaluates the quality of the entire predictive distribution.
- Kernel Parameter Evaluation: Learned kernel parameters (e.g., RBF kernel’s lengthscale, outputscale, and likelihood’s noise) provide insights into data smoothness, variability, and noise levels.

These metrics can be computed using GPyTorch’s predictions and true test data

## 4. Advanced Topics
### 4.1 Variational Autoencoders (VAE)
VAEs are generative models combining neural networks and variational inference. In Pyro/NumPyro, implementing a VAE involves:

- Encoder (Encoder) Network: A neural network (typically implemented using torch.nn.Module) that maps input data $x$ to the parameters of the approximate posterior distribution $q_ϕ (z∣x)$ of the latent variable $z$. These parameters are usually the mean and log variance of a normal distribution.
- Decoder Network: Another neural network that maps sampled latent variables ( $z$ ) to the parameters of the data distribution ( $p_\theta(x|z)$ ) (e.g., logits for Bernoulli distributions for binary images or mean for Normal distributions for continuous data).

- Probabilistic Model (Model): Defines the generative process ( $p(x,z) = p_\theta(x|z)p(z)$ ), where ( $p(z)$ ) is the prior (typically standard Normal). In Pyro/NumPyro, the decoder is part of the model; latent variables ( $z$ ) are sampled via pyro.sample from the prior, passed through the decoder to generate parameters for ( $x$ )’s distribution, and observed ( $x$ ) is sampled via pyro.sample.
- Guide Function: Defines the approximate posterior ( $q_\phi(z|x)$ ). The encoder is part of the guide, taking input ( $x$ ), outputting parameters for ( $q_\phi(z|x)$ ), and sampling ( $z$ ) via pyro.sample.
- Loss Function: Trained using SVI and ELBO. The ELBO decomposes into a reconstruction term (expected $( \log p_\theta(x|z) ))$ and a regularization term $(( KL(q_\phi(z|x) || p(z)) ))$.

```python
# Pyro VAE Example (conceptual structure)
class Encoder(nn.Module):
    # ... (e.g., convolutional layers, fully connected layers, outputting latent distribution parameters)
    def forward(self, x):
        # ...
        return z_loc, z_scale

class Decoder(nn.Module):
    # ... (e.g., fully connected layers, transposed convolutional layers, outputting data distribution parameters)
    def forward(self, z):
        # ...
        return reconstruction_params

class VAE(pynn.PyroModule):  # Or standard nn.Module with external model/guide
    def __init__(self, latent_dim, use_cuda=False):
        super().__init__()
        self.encoder = Encoder(...)
        self.decoder = Decoder(...)
        self.latent_dim = latent_dim
        # ...

    def model(self, x):
        # Register decoder with Pyro (if VAE is not a PyroModule, use pyro.module("decoder", self.decoder) inside the model function)
        self.decoder.train()  # Ensure training mode
        pyro.module("decoder", self.decoder)  # If VAE inherits from nn.Module

        with pyro.plate("data", x.size(0)):
            # Prior P(z)
            z_loc = torch.zeros(x.size(0), self.latent_dim, dtype=x.dtype, device=x.device)
            z_scale = torch.ones(x.size(0), self.latent_dim, dtype=x.dtype, device=x.device)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

            # Decoder P(x|z)
            loc_img = self.decoder(z)  # Decoder outputs parameters for reconstructed image
            # pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.view(-1, 784))  # e.g., for MNIST
            # Or other suitable observation distributions, like Normal

    def guide(self, x):
        # Register encoder with Pyro
        self.encoder.train()
        pyro.module("encoder", self.encoder)  # If VAE inherits from nn.Module

        with pyro.plate("data", x.size(0)):
            # Encoder q(z|x)
            z_loc, z_scale_raw = self.encoder(x)
            z_scale = torch.nn.functional.softplus(z_scale_raw)  # Ensure scale is positive
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

# Loss function (ELBO) computation and optimization steps follow the standard SVI workflow.
# optimizer_vae = Adam({"lr": 1.0e-3})
# svi_vae = SVI(model_vae.model, model_vae.guide, optimizer_vae, loss=Trace_ELBO())
# Training loop...
```

### 4.2 Hidden Markov Models (HMM)
Pyro and NumPyro support Hidden Markov Models (HMMs). They can be built using their distribution and control flow primitives (e.g., Python loops or Pyro/NumPyro’s scan operation). NumPyro also provides a high-level numpyro.distributions.HiddenMarkovModel distribution, similar to TFP’s HMM distribution, simplifying HMM definition and inference.

```python
# NumPyro HMM Example Using High-Level Distribution (conceptual)
key_hmm = random.PRNGKey(2)
num_timesteps = 100
num_hidden_states = 2
num_obs_dim = 1  # Assume 1D observations

# Assume HMM parameters are estimated or defined
initial_probs_np = jnp.array([0.8, 0.2])
transition_matrix_np = jnp.array([[0.7, 0.3], [0.2, 0.8]])
# Observation distribution parameters (e.g., mean and std for each hidden state)
obs_locs_np = jnp.array([-2.0, 2.0])
obs_scales_np = jnp.array([1.0, 1.5])

# Create HMM distribution object
hmm_dist = dist_np.HiddenMarkovModel(
    initial_distribution=dist_np.Categorical(probs=initial_probs_np),
    transition_distribution=dist_np.Categorical(probs=transition_matrix_np),
    observation_distribution=dist_np.Normal(loc=obs_locs_np, scale=obs_scales_np),  # Observation distribution parameters match number of hidden states
    num_steps=num_timesteps
)

# Sample observation sequence from HMM
# observations_np = hmm_dist.sample(key_hmm)  # observations_np.shape will be (num_timesteps,)

# For parameter inference (e.g., given observation sequence observations_np)
def hmm_parameter_inference_model(observations, num_hidden_states, num_timesteps):
    # Set priors for initial_probs, transition_matrix, obs_locs, obs_scales
    # init_probs_prior = numpyro.sample("init_probs", dist_np.Dirichlet(jnp.ones(num_hidden_states)))
    # ... priors for other parameters ...

    # Construct HMM distribution (parameters are sampled from priors above)
    hmm_for_inference = dist_np.HiddenMarkovModel(...)
    numpyro.sample("obs_seq", hmm_for_inference, obs=observations)

# Use MCMC (e.g., NUTS) or SVI for posterior inference of HMM parameters.
# Classic algorithms like the forward algorithm (compute likelihood), Viterbi algorithm (decode most likely hidden state sequence),
# and Baum-Welch algorithm (EM for parameter estimation) can be implemented within Pyro/NumPyro
# or their functionality obtained indirectly through these libraries’ inference engines.
# For example, hmm_dist.log_prob(observed_sequence) uses the forward algorithm.
# hmm_dist.posterior_mode(observed_sequence) (if available) or infer hidden states via MCMC/VI.
```

### 4.3 Dirichlet Process Mixture Models (DPMM)
Dirichlet Process Mixture Models (DPMMs) are non-parametric Bayesian methods commonly used for clustering, with the advantage of automatically inferring the number of clusters from data. In Pyro/NumPyro, DPMMs are typically implemented via their Stick-Breaking construction.

```python
# Pyro DPMM Example (conceptual structure - Stick-Breaking)
def dpmm_model_pyro(data, max_num_components, alpha_dp):  # alpha_dp is the concentration parameter of the Dirichlet Process
    # Stick-breaking process to generate mixture weights (pi_k)
    # beta_k ~ Beta(1, alpha_dp) for k = 1 to max_num_components-1
    with pyro.plate("beta_plate", max_num_components - 1):
        beta_k = pyro.sample("beta_k", dist.Beta(1., alpha_dp))

    # Compute actual mixture weights pi via stick-breaking
    # pi_k = beta_k * product_{j<k} (1 - beta_j)
    # Requires careful indexing and cumprod
    # Pyro’s dist.DirichletProcess can also simplify this

    # Priors for cluster parameters (e.g., mean and variance in a Gaussian mixture model)
    with pyro.plate("components_plate", max_num_components):
        # mu_k ~ Normal(loc_prior, scale_prior)
        mu_k = pyro.sample("mu_k", dist.Normal(0., 5.))  # Assumed prior
        # sigma_k ~ LogNormal(loc_prior, scale_prior) or InverseGamma
        sigma_k = pyro.sample("sigma_k", dist.LogNormal(0., 1.))

    # Assign clusters to each data point and generate data from that cluster
    with pyro.plate("data_plate", data.shape[0]):
        # z_n ~ Categorical(pi) (assign data point n to a cluster)
        assignment = pyro.sample("assignment", dist.Categorical(probs=calculated_pi_from_beta))
        # obs_n ~ Normal(mu_k[z_n], sigma_k[z_n])
        pyro.sample("obs", dist.Normal(mu_k[assignment], sigma_k[assignment]), obs=data)

# Inferring DPMMs is complex and may involve:
# 1. Variational Inference: Requires careful guide design, possibly involving structured variational inference.
# 2. MCMC: Collapsed Gibbs Sampling or general samplers like NUTS (but may be less efficient).
# Pyro and NumPyro provide tools to help build such models, but implementation details can be intricate.
```

## 5. Principles of Variational Inference
The core principles of Variational Inference (VI) are consistent across probabilistic programming frameworks:

- Goal: Approximate a complex or intractable true posterior ( $p(\theta|D)$ ) with a simple, parameterized distribution ( $q(\theta;\phi)$ ) (called the variational distribution or guide).
- Method: Optimize variational parameters ( $\phi$ ) to minimize the KL divergence between ( $q(\theta;\phi)$ ) and ( $p(\theta|D)$ ): $KL(q(\theta;\phi) || p(\theta|D)) = \int q(\theta;\phi) \log \frac{q(\theta;\phi)}{p(\theta|D)} d\theta$
- Evidence Lower Bound (ELBO): Directly minimizing KL divergence is typically infeasible since ( $p(\theta|D)$ ) is unknown. Instead, maximize the $ELBO$: $ELBO(q) = \mathbb{E}{q(\theta;\phi)} [\log p(D,\theta)] - \mathbb{E}{q(\theta;\phi)} [\log q(\theta;\phi)] \ ELBO(q) = \mathbb{E}_{q(\theta;\phi)} [\log p(D|\theta)] - KL(q(\theta;\phi) || p(\theta))$


Maximizing the ELBO is equivalent to minimizing the KL divergence. The first term ( $\mathbb{E}_{q(\theta;\phi)} [\log p(D|\theta)] $ ) encourages the variational distribution to explain the data (likelihood expectation), while the second term ( $KL(q(\theta;\phi) || p(\theta)) $ ) encourages closeness to the prior.

- Mean-Field Assumption: A common simplification assumes the variational posterior factorizes as a product of independent distributions for each parameter (or parameter group): ($ q(\theta) = \prod_i q_i(\theta_i)$ ). This simplifies optimization but may fail to capture posterior dependencies. Pyro/NumPyro’s AutoDiagonalNormal guide is a mean-field variational family where each latent variable is approximated by an independent Normal distribution.

- Stochastic Variational Inference (SVI): For large datasets, computing expectations in the ELBO (especially the log-likelihood term) can be costly. SVI uses stochastic gradient ascent, estimating ELBO gradients via mini-batches of data and samples from ( $q(\theta;\phi)$ ). Pyro/NumPyro’s SVI class is designed for this purpose.

## 6. Principles of MCMC Methods
Markov Chain Monte Carlo (MCMC) methods generate samples from the target posterior ($ p(\theta|D) $) by constructing a specialized Markov chain.

- Core Idea: Design a Markov chain whose unique stationary distribution is the desired posterior ( $p(\theta|D) $). Running the chain long enough produces samples approximately drawn independently from ( $p(\theta|D)$ ).
- Metropolis-Hastings (MH) Algorithm: A foundational MCMC algorithm.
    1. From the current state ( $\theta_t$ ), generate a candidate state ( $\theta^*$ ) using a proposal distribution ( $q(\theta^*|\theta_t)$ ), and decide whether to accept it based on an acceptance probability ( $\alpha$ ).
    2. Acceptance probability ( $\alpha$ ): $\alpha = \min\left(1, \frac{p(\theta^*|D) q(\theta_t|\theta^*)}{p(\theta_t|D) q(\theta^*|\theta_t)}\right)$
    3. Accept the candidate with probability ( $\alpha$ ), setting ( $\theta_{t+1} = \theta^*$ ); otherwise, reject it, setting ( $\theta_{t+1} = \theta_t$ ).

- Hamiltonian Monte Carlo (HMC): A more advanced MCMC method, ideal for high-dimensional continuous parameter spaces. It introduces auxiliary momentum variables ($ p $) and uses Hamiltonian dynamics to propose distant candidates with high acceptance probabilities.
    1. Treat parameters ($ \theta$ ) as positions and introduce momentum ( p ). Construct the Hamiltonian: $H(\theta, p) = -\log p(\theta|D) + \frac{1}{2} p^T M^{-1} p $ where ( $M $) is the mass matrix (often identity), ( $-\log p(\theta|D) $ ) is the potential energy, and ($ \frac{1}{2} p^T M^{-1} p $ ) is the kinetic energy.
    2. From the current state ( ($\theta_t, p_t$) ) (with ( p_t ) typically drawn from ( N(0, M) )), simulate Hamiltonian dynamics using a numerical integrator (e.g., Leapfrog) for a fixed time, yielding a candidate ( ($\theta^*, p^*$) ).
    3. Compute acceptance probability based on energy change: $\alpha = \min(1, \exp(-H(\theta^*, p^*) + H(\theta_t, p_t)))$ (With exact integrators, acceptance is always 1; numerical errors reduce it slightly.) HMC uses gradient information from ( $-\log p(\theta|D) $ ) to guide sampling, avoiding the random walk behavior of MH and improving efficiency.

- No-U-Turn Sampler (NUTS): An adaptive extension of HMC that automatically adjusts the number of Leapfrog steps, eliminating the need for manual tuning and efficiently exploring the posterior. NUTS is the default and recommended kernel for NumPyro’s MCMC.

## Practical Considerations
Similar to TFP documentation, using probabilistic programming in the PyTorch ecosystem requires attention to:

- Model Selection and Evaluation:
    - ELBO (for VI): While ELBO is the optimization target for variational inference, its absolute value is not directly equivalent to model evidence ( P(D) ). However, comparing optimal ELBO values for different models on the same data can aid model selection.
    - Posterior Predictive Checks (PPC): Sample parameters from the posterior (or approximate posterior) and generate simulated datasets. Compare statistical properties (e.g., mean, variance, quantiles, event frequencies) of simulated data with those of observed data. Pyro/NumPyro’s Predictive class (pyro.infer.Predictive / numpyro.infer.Predictive) facilitates posterior predictive checks.
    - Information Criteria: WAIC (Widely Applicable Information Criterion) and LOO-CV (Leave-One-Out Cross-Validation) are more sophisticated model comparison methods based on posterior predictions. External libraries like ArviZ can compute these metrics and integrate well with Pyro/NumPyro outputs.

- Computational Efficiency:
    - VI vs. MCMC: VI is typically much faster than MCMC, especially for large datasets and complex models, but its accuracy is limited by the expressiveness of the variational family. MCMC can theoretically converge to the true posterior but is computationally expensive, potentially requiring many samples and long runtimes.
    - Model Parameterization: The parameterization of the model significantly affects inference efficiency and convergence. For constrained parameters (e.g., standard deviation must be positive), transformations (e.g., log or softplus) can map them to an unconstrained space, simplifying optimization or sampling.
    - Gradient Computation: Pyro relies on PyTorch’s Autograd, while NumPyro uses JAX’s automatic differentiation. Both are highly efficient.
    - Hardware Acceleration: PyTorch and JAX support GPU acceleration. NumPyro often leverages JAX’s JIT compilation for superior performance on GPUs/TPUs.
    - Data Subsampling: For large datasets, SVI can use mini-batches. Some advanced MCMC algorithms support subsampling, but implementation is more complex.

- Diagnostics and Convergence Checks:
    - MCMC Diagnostics:
        - Trace Plots: Plot posterior samples for each parameter against iteration number to check if the chain has stabilized and shows no divergence trends.
        - ( \hat{R} ) (R-hat, Gelman-Rubin Statistic): For multiple parallel Markov chains, ( \hat{R} ) compares inter-chain and intra-chain variance. Values close to 1 indicate convergence to a similar distribution.
        - Effective Sample Size (ESS): Since MCMC samples are autocorrelated, ESS measures the information content of the samples relative to independent samples. Low ESS suggests more samples or higher sampling efficiency is needed.
        - Autocorrelation Plots: Show correlations between samples and their lagged versions, helping assess chain mixing speed. NumPyro’s mcmc.print_summary() reports ( \hat{R} ) and ESS. The ArviZ library provides comprehensive diagnostic tools.
    - VI Diagnostics:
        - ELBO Convergence: Monitor whether the ELBO stabilizes during training.
        - Variational Parameter Checks: Verify that learned variational parameters (e.g., mean and std of approximate posteriors) are reasonable.
        - Posterior Predictive Checks: Similar to MCMC, use samples from the approximate posterior for PPC.
        - Sensitivity Analysis: Assess the sensitivity of inference results (especially posterior distributions) to prior choices. High sensitivity to small prior changes may indicate insufficient data or model issues.

## Comparison and Summary of Probabilistic Models (PyTorch Ecosystem)
The following table summarizes the characteristics of key probabilistic models implemented in the PyTorch ecosystem (primarily using Pyro, NumPyro, GPyTorch):

| Model Type | Core Library/Method | Advantages | Disadvantages | Applicable Scenarios | Uncertainty Representation |
|------------|-------------------|------------|---------------|---------------------|---------------------------|
| Bayesian Linear Regression | Pyro/NumPyro (pyro.sample, dist, SVI/MCMC) | 1. Provides parameter uncertainty<br>2. Prevents overfitting<br>3. Suitable for small datasets | 1. Limited expressive power<br>2. Struggles with nonlinear relationships | 1. Small datasets<br>2. Need for interpretability<br>3. Parameter uncertainty quantification | Parameter posterior distribution |
| Gaussian Process Regression | GPyTorch (gpytorch.models.ExactGP, kernels, likelihoods) | 1. Non-parametric, highly flexible<br>2. Provides full predictive distribution<br>3. Performs well on small datasets, GPU-accelerated<br>4. Kernel engineering enables powerful modeling | 1. Exact GP has high computational complexity (O(N^3)) (GPyTorch offers sparse GP approximations)<br>2. Sensitive to kernel choice<br>3. Poor performance in high dimensions (curse of dimensionality) | 1. Small to medium datasets<br>2. Smooth function assumptions<br>3. Active learning, Bayesian optimization (BoTorch based on GPyTorch) | Posterior distribution of function values (predictive mean and variance) |
| Hidden Markov Model (HMM) | Pyro/NumPyro (dist.HiddenMarkovModel or manual construction) | 1. Suitable for sequence data modeling<br>2. Interpretable hidden states<br>3. Mature classical inference algorithms (forward, Viterbi) | 1. Limited by Markov assumption (current state depends only on previous state)<br>2. Number of hidden states typically predefined<br>3. Struggles with long-term dependencies | 1. Time series analysis<br>2. Speech recognition<br>3. NLP (part-of-speech tagging)<br>4. Biological sequence analysis | Posterior probabilities of hidden state sequences, parameter posterior distributions |
| Variational Autoencoder (VAE) | Pyro/NumPyro + torch.nn | 1. Powerful nonlinear generative capabilities<br>2. Learns meaningful low-dimensional latent representations<br>3. Scalable to large, high-dimensional data | 1. Training can be unstable, sensitive to hyperparameters<br>2. ELBO optimization may lead to "posterior collapse"<br>3. Generated samples may be blurry | 1. Generation of complex data (images, text)<br>2. Unsupervised representation learning<br>3. Anomaly detection, data compression | Approximate posterior distribution of latent variables |
| Dirichlet Process Mixture Model (DPMM) | Pyro/NumPyro (typically using Stick-Breaking, dist.DirichletProcess) | 1. Automatically infers number of clusters (non-parametric)<br>2. Provides clustering uncertainty<br>3. Avoids pre-specifying number of clusters | 1. High computational complexity, slow convergence<br>2. May perform poorly on very high-dimensional data<br>3. Model and inference implementation are complex | 1. Clustering with unknown number of clusters<br>2. Density estimation<br>3. Topic modeling | Posterior probabilities of cluster assignments, cluster parameter posteriors, posterior of cluster count |


## 9. Guidelines for Choosing the Right Model (PyTorch Ecosystem)
When selecting the appropriate probabilistic model for a specific application, consider the following factors:

1. Data Characteristics:
    - Data Size:
        - Small to medium datasets (< a few thousand samples): Bayesian linear regression, Gaussian processes (GPyTorch’s exact GP), HMM.
        - Large datasets: Variational autoencoders (VAE), Bayesian neural networks with SVI, sparse Gaussian processes (GPyTorch).
    - Data Dimensionality:
        - Low to medium dimensions: GPs perform well.
        - High dimensions: VAEs, BNNs are more suitable. GPs face the “curse of dimensionality” unless using specialized dimensionality-reducing kernels or methods.

    - Data Type and Structure:
        - Sequential data: HMM, Bayesian versions of recurrent neural networks (RNNs).
        - Image/video data: VAEs (typically based on convolutional networks), Bayesian convolutional neural networks.
        - Tabular data: Bayesian linear regression, GPs, BNNs.
        - Known dependencies between data points (e.g., graph structures): Bayesian versions of graph neural networks (GNNs, more cutting-edge).

2. Task Requirements:
    - Prediction vs. Inference: Is the goal accurate point predictions, full predictive distributions with uncertainty quantification, or understanding the data-generating process and parameter meanings?
    - Uncertainty Quantification: If reliable uncertainty estimates are critical (e.g., in medical, financial, or safety-critical applications), Bayesian methods (e.g., GPs, BNNs, MCMC-inferred parameter models) are preferred.
    - Interpretability: Simpler models (e.g., Bayesian linear regression, HMMs with few states) are typically more interpretable than complex models (e.g., deep BNNs or VAEs). GP kernels (e.g., periodic kernels indicating periodicity) can also provide some interpretability.
    - Automated Decision-Making: If model outputs feed into automated decisions (e.g., acquisition functions in Bayesian optimization), the quality of predictive distributions is crucial. BoTorch is specialized for this.

3. Computational Resources and Constraints:
    - Training Time: MCMC is generally much slower than VI. VI training time depends on model complexity and data size. Exact GPs scale cubically with data size.
    - Memory Requirements: Exact GPs are memory-intensive. Large-scale neural network models also require significant memory.
    - Inference Speed (Prediction): Trained VI models or point-estimate models are typically fast for predictions. MCMC posterior predictions may require averaging over multiple samples but are relatively fast. GP prediction speed depends on the number of test and training points.

4. Availability and Integration of Prior Knowledge:
    - Bayesian methods naturally incorporate domain knowledge via prior distributions. Strong, reliable priors can significantly improve inference on small datasets.
    - Carefully consider prior choices, as inappropriate priors can lead to biased or unstable results, especially with limited data. Prior predictive checks can assess prior reasonableness.
By weighing these factors, you can select the most suitable probabilistic model and leverage the powerful tools in the PyTorch ecosystem for complex uncertainty modeling tasks.