import math
import numpy as np
import torch
import gpytorch
from matplotlib import pyplot as plt

from helper_functions.artificial_functions import cost
from helper_functions.utils import to_unit_cube

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # additive models for model paramaters and system parameters
        # different lenghtscales
        # (ARD)
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def init_cost_model(train_x, train_y):
    # init gp model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    # train gp model
    training_iter = 50
    
    model.train() # put the gpytorch model in training mode?
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        #     i + 1, training_iter, loss.item(),
        #     model.covar_module.base_kernel.lengthscale.item(),
        #     model.likelihood.noise.item()
        # ))
        optimizer.step()

    return model,likelihood

def get_best_system_params(model, likelihood, candidates):
    # accepts phi, chi value from the candidates
    # returns a combined vector containing both (phi,chi) & best system configuration
    
    # Get all possible config options
    # compute all the possibilities of discrete system configurations combinations
    # enumerate possibilities for each system parameter
    nnodes = torch.arange(2,33)
    ntasks = torch.arange(2,11)
    mem = torch.tensor([4,8,16,32,64])

    # normalize the possibilities
    nnodes_normalized = to_unit_cube(nnodes, variable_type="nnodes")
    ntasks_normalized = to_unit_cube(ntasks, variable_type="ntasks")
    mem_normalized = to_unit_cube(mem, variable_type="mem")

    # form all permutations and combinattions
    config_opts = torch.tensor(
                    np.array(np.meshgrid(nnodes_normalized, ntasks_normalized, mem_normalized)).T.reshape(-1, 3))
    

    # combine system configs with model parameters (phi, chi)
    phi = candidates[0,0]
    chi = candidates[0,1]
    phi_array = (torch.ones(config_opts.shape[0])*phi).unsqueeze(-1)
    chi_array = (torch.ones(config_opts.shape[0])*chi).unsqueeze(-1)

    test_x = torch.hstack([phi_array, chi_array, config_opts])

    # generate predictions
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    
    
    with torch.no_grad():
        observed_pred = likelihood(model(test_x))
        lower, upper = observed_pred.confidence_region()
        confidence_interval = upper-lower
        confidence = observed_pred.mean + 0.1*confidence_interval

    # find the system config with least predicted cost
    min_cost_idx = np.argmin(observed_pred.mean.numpy())
    # min_cost_idx = torch.argmin(confidence)
    min_cost_config = test_x[min_cost_idx, :].unsqueeze(0)
    
    return min_cost_config


    