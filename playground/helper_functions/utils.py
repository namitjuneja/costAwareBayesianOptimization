import torch
from copy import deepcopy

"""
Domain information [in model] [actual]
phi [0,1] [-32,32]
chi [0,1] [-32,32]

nnodes [0,1] [2,32]
ntasks [0,1] [2,10]
mem [0,1] [2,6] [4,64]

"""

def from_unit_cube(x, variable_type):
    # convert variables from [0,1] domain to their original domain
    # un-normalizing
    # result = x(ub-lb) + lb
    if variable_type == "phi":
        # [-32,32]
        return (x*64) - 32
    elif variable_type == "chi":
        # [-32,32]
        return (x*64) - 32
    elif variable_type == "nnodes":
        # [2,32]
        return x*(30) + 2
    elif variable_type == "ntasks":
        # [2,10]
        return x*(8) + 2
    elif variable_type == "mem":
        # [2,6] (**2)
        return 2**(x*(4) + 2)
    else:
        raise Exception()

def from_n_unit_cube(x):
    # accepts [n,5] dimensional tensor in [0,1]^5 domain
    # outputs [n,5] dimensional tensor in the original domain
    x_denormalized = deepcopy(x)
    x_denormalized[:,0] = from_unit_cube(x[:,0], variable_type="phi")
    x_denormalized[:,1] = from_unit_cube(x[:,1], variable_type="chi")
    x_denormalized[:,2] = from_unit_cube(x[:,2], variable_type="nnodes")
    x_denormalized[:,3] = from_unit_cube(x[:,3], variable_type="ntasks")
    x_denormalized[:,4] = from_unit_cube(x[:,4], variable_type="mem")
    return x_denormalized

def to_unit_cube(x, variable_type):
    # convert variables from their original domain to [0,1]
    # normalizing
    # result = x-lb/(ub-lb)
    if variable_type == "phi":
        # [-32,32]
        return (x+32)/64
    elif variable_type == "chi":
        # [-32,32]
        return (x+32)/64
    elif variable_type == "nnodes":
        # [2,32]
        return (x-2)/30
    elif variable_type == "ntasks":
        # [2,10]
        return (x-2)/8
    elif variable_type == "mem":
        # [2,6] (**2)
        xx = torch.log2(x)
        return (xx-2)/4
    else:
        raise Exception()

def to_n_unit_cube(x):
    # accepts [n,5] dimensional tensor in the original domain
    # outputs [n,5] dimensional tensor in [0,1]^5 domain
    x_normalized = deepcopy(x)
    x_normalized[:,0] = to_unit_cube(x[:,0], variable_type="phi")
    x_normalized[:,1] = to_unit_cube(x[:,1], variable_type="chi")
    x_normalized[:,2] = to_unit_cube(x[:,2], variable_type="nnodes")
    x_normalized[:,3] = to_unit_cube(x[:,3], variable_type="ntasks")
    x_normalized[:,4] = to_unit_cube(x[:,4], variable_type="mem")
    return x_normalized


def get_model_parameters(candidates):
    # Accepts [n,5] tensors and returns [n,2] tensor
    # consisting of only model paramaeter dimensions of the acquired points and
    # stripping the system parameters dimensions (last 3)
    return candidates[:,:2]