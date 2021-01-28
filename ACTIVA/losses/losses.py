import math
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parallel import data_parallel
import torch.multiprocessing as multiprocessing

"""
Here we list the losses as separate functions (instead of ACTIVA methods)
for modular use, in case needed in debugging or evaluation
"""
    
def kl_loss(mu, covariance, prior_mu=0):
    """
        
    KL-divergence loss using the shorthand notation in this paper: https://arxiv.org/pdf/1807.06358.pdf
        
    """
    # inplace
    v_kl = mu.add(-prior_mu).pow(2).add_(covariance.exp()).mul_(-1).add_(1).add_(covariance)
    v_kl = v_kl.sum(dim=-1).mul_(-0.5)
        
    return v_kl
    
def reconstruction_loss(prediction, target, size_average=False):     
    """
        
    Reconstruction loss for the output of the decoder and the original data (essentially MSE)
        
    """
    error = (prediction - target).view(prediction.size(0), -1)
    error = error**2
    error = torch.sum(error, dim=-1)
        
    if size_average:
        error = error.mean()
    else:
        error = error.sum()
               
    return error
    
    
def classification_loss(cf_prediction, cf_target, size_average=False):     
    """
        
    Cell type prediction loss between then generated cells and the real cells
        
    """
    error = (cf_prediction - cf_target).view(cf_prediction.size(0), -1)
    error = error**2
    error = torch.sum(error, dim=-1)
        
    if size_average:
        error = error.mean()
    else:
        error = error.sum()
               
    return error