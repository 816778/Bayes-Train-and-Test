import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


class BayesianNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def model(x_data, y_data):
    fc1w_prior = dist.Normal(loc=torch.zeros_like(bnn.fc1.weight), scale=torch.ones_like(bnn.fc1.weight))
    fc1b_prior = dist.Normal(loc=torch.zeros_like(bnn.fc1.bias), scale=torch.ones_like(bnn.fc1.bias))
    fc2w_prior = dist.Normal(loc=torch.zeros_like(bnn.fc2.weight), scale=torch.ones_like(bnn.fc2.weight))
    fc2b_prior = dist.Normal(loc=torch.zeros_like(bnn.fc2.bias), scale=torch.ones_like(bnn.fc2.bias))

    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,
              'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior}

    lifted_module = pyro.random_module("module", bnn, priors)
    lifted_reg_model = lifted_module()

    with pyro.plate("data", x_data.size(0)):
        logits = lifted_reg_model(x_data)
        pyro.sample("obs", dist.Categorical(logits=logits), obs=y_data)


# Guía:
def guide(x_data, y_data):
    fc1w_mu = torch.randn_like(bnn.fc1.weight)
    fc1w_sigma = torch.randn_like(bnn.fc1.weight)
    fc1b_mu = torch.randn_like(bnn.fc1.bias)
    fc1b_sigma = torch.randn_like(bnn.fc1.bias)

    fc2w_mu = torch.randn_like(bnn.fc2.weight)
    fc2w_sigma = torch.randn_like(bnn.fc2.weight)
    fc2b_mu = torch.randn_like(bnn.fc2.bias)
    fc2b_sigma = torch.randn_like(bnn.fc2.bias)

    fc1w_dist = dist.Normal(loc=fc1w_mu, scale=torch.abs(fc1w_sigma))
    fc1b_dist = dist.Normal(loc=fc1b_mu, scale=torch.abs(fc1b_sigma))
    fc2w_dist = dist.Normal(loc=fc2w_mu, scale=torch.abs(fc2w_sigma))
    fc2b_dist = dist.Normal(loc=fc2b_mu, scale=torch.abs(fc2b_sigma))

    dists = {'fc1.weight': fc1w_dist, 'fc1.bias': fc1b_dist,
             'fc2.weight': fc2w_dist, 'fc2.bias': fc2b_dist}

    lifted_module = pyro.random_module("module", bnn, dists)
    return lifted_module()
