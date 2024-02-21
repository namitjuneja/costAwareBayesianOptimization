'''
This file contains the functions we are trying to model/optimize. 
- ackley function
- cost function
In real life these functions won't be avilable (black box). 
'''

import numpy as np


# ackley function (this function gets optimized by the surrogate gp)
def ack(candidate):
    x = candidate[:,0]/10
    y = candidate[:,1]/10
    result = -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))-np.exp(0.5 * (np.cos(2 * 
      np.pi * x)+np.cos(2 * np.pi * y))) + np.e + 20
    return -1*result
# domain: -32,32




# cost function (this function gets modeled by the cost function gp regression)
def f_phi(phi):
    return (0.46*phi) + 17
def f_chi(chi):
    return (0.12*chi) + 6
def time_component_1(nnodes, phi):
    return ((nnodes - f_phi(phi))/4)**2
def time_component_2(ntasks, chi):
    return ((ntasks - f_chi(chi))/4)**2
def time_component_3(mem):
    return 10/mem
def time_taken(nnodes, ntasks, mem, phi, chi):
    return time_component_1(nnodes, phi) + \
           time_component_2(ntasks, chi) + \
           time_component_3(mem) + 10
    
def aws_c6gn_cost_simulator(nnodes, ntasks, mem):
    # returns cost in per unit hour
    cost_per_node = 0.00864*ntasks + 0.01728*mem
    return nnodes*cost_per_node
    
def cost(candidate):
    phi = candidate[:,0]
    chi = candidate[:,1]
    nnodes = candidate[:,2]
    ntasks = candidate[:,3]
    mem = candidate[:,4]
    
    # get cost per unit time for the given system configuration
    unit_cost = aws_c6gn_cost_simulator(nnodes, ntasks, mem)

    # get time taken for the given model and system parameters
    time_cost = time_taken(nnodes, ntasks, mem, phi, chi)
    
    return 0#time_cost*unit_cost
    # return time_cost