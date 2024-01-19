import numpy as np
import torch
import gpytorch
import botorch
from matplotlib import pyplot as plt
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.models.transforms.outcome import Standardize
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf

torch.set_default_dtype(torch.float64) # avoid matrix not positive semi-definite

# ackley function (black box function to be optimized)
def ack(x, y):
    x = x/10
    y = y/10
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))-np.exp(0.5 * (np.cos(2 * 
      np.pi * x)+np.cos(2 * np.pi * y))) + np.e + 20

# domain: -32,32

def from_unit_cube(x):
    # for ackley blackbox function
    return (x*64) - 32

def get_initial_data(n, f, show_plot):
    # generate init points
    train_x_1 = torch.rand(9,2)
    train_x_2 = torch.tensor([[0.5,0.5]])
    train_x = torch.cat((train_x_1, train_x_2))
    train_x = train_x_1
    
    train_x_real = from_unit_cube(train_x) # scale the points to the ackley function's domain
    train_y = f(train_x_real[:,0], train_x_real[:,1]).unsqueeze(-1)
    
    best_y = train_y.min().item()

    if show_plot:
        plot_init_data(f, train_x_real, train_y)

    return train_x, train_y, best_y
    
def evaluate_candidates(candidates, f):
    candidates_x_denormalized = from_unit_cube(candidates[:,0])
    candidates_y_denormalized = from_unit_cube(candidates[:,1])
    evaluation = f(candidates_x_denormalized, candidates_y_denormalized).unsqueeze(-1)
    return evaluation

def init_gp_model(train_x, train_y):
    covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        MaternKernel(nu=2.5, ard_num_dims=2)
    )
    single_model = SingleTaskGP(
                    train_x, 
                    train_y, 
                    covar_module=covar_module,
                    outcome_transform=Standardize(m=1))

    mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)

    # fit the gp model
    fit_gpytorch_model(mll)

    return single_model, mll

def init_acq_func(model, best_y):
    ei = qExpectedImprovement(
        model=model,
        best_f=best_y)
    return ei

def gen_candidates(n, acq_func):
    bounds = torch.tensor([[0.,0.],[1.,1.]]) # ack fn's domain

    candidates, _ = optimize_acqf(
                        acq_function=acq_func,
                        bounds=bounds,
                        q=1,
                        num_restarts=5,
                        raw_samples=20)
    
    return candidates

def plot_init_data(f):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(-32,32,100)
    y = np.linspace(-32,32,100)
    x_ids, y_ids = np.meshgrid(x,y)
    result = f(x_ids,y_ids)
    ax.plot_surface(x_ids, y_ids, result, color="grey", alpha=0.05)
    ax.scatter(train_x_real[:,0], train_x_real[:,1], train_y, c='blue')