import numpy as np

import torch
from torch.nn.functional import relu

# NOTE: This requires https://github.com/cambridge-mlg/expressiveness-approx-bnns, which can be installed by cloning and pip installing in the root
# I had issues installing with setup.py :(
from inbetween.models import FFGBNN
from inbetween.likelihoods import HomoskedasticGaussianRegression
from inbetween.losses import elbo_batch_loss

# hyperparameters
width = 2000
C = 200
noise_variance = 1.0
num_samples = 10000 #number of MC samples to use when estimating quantities
X = torch.Tensor([[0],[5]])

likelihood = HomoskedasticGaussianRegression(noise_std=np.sqrt(noise_variance)) # Gaussian likelihood
# Build model, sigma_w is automatically scaled by sqrt(1/n)
bnn = FFGBNN(input_dim=1, output_dim=1, likelihood=likelihood, nonlinearity=relu,
             num_layers=1, width=width, num_train=X.shape[0], sigma_w=1.0, sigma_b=1.0)

# Build Q_n distribution described in note, we rescale by width instead of sqrt(width) to deal with 1/root(n) factor/non identity prior
def set_top_layers_weights():
    bnn.set_prior()
    bnn.layers[1].w_mean = torch.nn.Parameter(bnn.layers[1].w_mean + np.sqrt(C)/width)

set_top_layers_weights()
# Set Y to be predictive mean of Q_n, note this should be independent of n (up to MC estimation + numerical error etc)
Y = bnn.pred_mean_std(X, num_samples=num_samples)[0].detach()
# Compute elbo
print("Loss of Q_n:", -elbo_batch_loss(bnn, X, Y, inds=np.arange(X.shape[0]), samples=num_samples))
# optimal bias should be average y value for minimising SSE. We bias it toward each data point to allow for some variation in the mean
opt_bias = Y.mean() + torch.Tensor([[-.5], [.5]])
se = np.square(Y-opt_bias)
sse = se.sum()
# Compute bound by dropping KL and applying Jensen
print("Bound on loss for optimal BNN with near constant mean", -np.log(2 * np.pi) - 0.5 * sse / noise_variance)



