import math
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
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf

from helper_functions.artificial_functions import ack, cost
from helper_functions.utils import *


torch.set_default_dtype(torch.float64) # avoid matrix not positive semi-definite
    


def get_initial_data(n, f, show_plot):
    # get 10 random points of 5 dimensions (phi, chi, nnodes, ntasks, mem)
    train_x = torch.rand(10,5)

    # evaluate the black box function and cost function at the acquired points
    train_y = evaluate_f(train_x, f)
    train_y_cost = evaluate_cost(train_x)
    
    best_y = train_y.max().item() # needed for the expected imrpovement acq fn

    if show_plot:
        plot_init_data(f, train_x_model_parameters, train_y)

    return train_x, train_y, best_y, train_y_cost
    
def evaluate_f(candidates, f):
    # de-normalize the acquired point
    candidates_denormalized = from_n_unit_cube(candidates)
    # extract only the model parameters
    candidates_denormalized_model_parameters = get_model_parameters(candidates_denormalized)
    evaluation = f(candidates_denormalized_model_parameters).unsqueeze(-1)
    return evaluation

def evaluate_cost(candidates):
    # de-normalize the acquired point
    candidates_denormalized = from_n_unit_cube(candidates)
    return cost(candidates_denormalized)

def init_gp_model(train_x, train_y):
    # extract only the model parameters
    train_x = get_model_parameters(train_x)
    
    # covariance kernel
    covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        MaternKernel(nu=2.5, ard_num_dims=2)
    )

    # init the the surrogate model 
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
    ei = ExpectedImprovement(
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

def plot_init_data(f, train_x_model_parameters, train_y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(-32,32,100)
    y = np.linspace(-32,32,100)
    x_ids, y_ids = np.meshgrid(x,y)
    f_input_vector = np.stack((x_ids, y_ids), axis=-1).reshape((100*100, 2))

    result = f(f_input_vector).reshape([100,100])
    ax.plot_surface(x_ids, y_ids, result, color="grey", alpha=0.05)
    ax.scatter(train_x_model_parameters[:,0], train_x_model_parameters[:,1], train_y, c='blue')

