import math
import time
import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from torch.autograd import Variable
from torch.nn.parallel import data_parallel
import torch.multiprocessing as multiprocessing


class ACTIVA(nn.Module):
    """
    This is the class where we will be calling for training ACTIVA.
    This class will call the encoder and decoder and the reparametrization 
    
    """
    
    def __init__(self, latent_dim = 128, input_size = 256, threshold=0):
        """
        
        theshold -> to threshold the outputs of the model in order to promote sparsity
        
        """
        super(ACTIVA, self).__init__() 
        
        self.zdim = latent_dim;
        self.inp_size = input_size;
        self.threshold = threshold;
        # call the encoder
        self.encoder = Encoder(self.zdim, self.inp_size);
        # call the decoder
        self.decoder = Decoder(self.zdim, self.inp_size, threshold = self.threshold);
        
    
    def forward(self, x):
        """
        
        Forward pass through the network from the input data
        
        """
        mu, variance = self.encode(x);
        # the latent space mu and variance, but reparametrized
        z = self.reparameterize(mu, variance);
        # the reconstructed data
        x_r = self.decode(z);
        
        return  mu, variance, z, x_r

    
    def reparameterize(self, mu, variance):
        """
        
        To do the reparametrization trick of the VAEs
        
        """
        
        std = variance.mul(0.5).exp() 
        # check to see which device we are running on
        ####### WARNING: if GPU is available but we run on CPU, this could be a bug here!! ####### 
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()     
            
        eps = Variable(eps)
        return eps.mul(std).add(mu)
    
    def sample(self, z):
        """
        
        To sample from the all the means and covariance (just like in VAEs)
        
        """
        x_r = self.decode(z)
        return x_r
    
    
    def encode(self, x):  
        """
        
        Call to encoder 
        
        """
        mu, variance = data_parallel(self.encoder, x)
        return mu, variance
        
    def decode(self, z):  
        """
        
        Call to decoder 
        
        """
        x_r = data_parallel(self.decoder, z)
        
        return x_r
    
    def kl_loss(self, mu, covariance, prior_mu=0):
        """
        
        KL-divergence loss using the shorthand notation in this paper: https://arxiv.org/pdf/1807.06358.pdf
        
        """
        # inplace
        v_kl = mu.add(-prior_mu).pow(2).add_(covariance.exp()).mul_(-1).add_(1).add_(covariance)
        v_kl = v_kl.sum(dim=-1).mul_(-0.5)
        
        # not inplace (less efficient)
#         v_kl = mu.add(-prior_mu).pow(2).add(covariance.exp()).mul(-1).add(1).add(covariance)
#         v_kl = v_kl.sum(dim=-1).mul(-0.5) # (batch, 2)
        
        return v_kl
    
    def reconstruction_loss(self, prediction, target, size_average=False):     
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
    
    
    def classification_loss(self, cf_prediction, cf_target, size_average=False):     
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