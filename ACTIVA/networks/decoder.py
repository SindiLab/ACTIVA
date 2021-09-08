import torch
import torch.nn as nn
from .LSN_layer import LSN

class Decoder(nn.Module):
    def __init__(self, latent_dim = 128, input_size = None, scale = 20000, threshold = 0):
        """
            
        The decoder class
            
        """
        if latent_dim == None or input_size == None:
            raise ValueError('Must explicitly declare input size and latent space dimension (2*latent_dim for enc)')
            
        super(Decoder, self).__init__();
        self.inp_dim = input_size;
        self.zdim = latent_dim;
        self.scale = scale;
        self.lsn = LSN(scale=self.scale)
        
        # here we decide the thresholding 
        if threshold == 0:
            self.thres_layer = nn.ReLU()
        else:
            # here we will replace all values smaller than threshold with 0
            print(f"==> thresholding with {threshold}")
            self.thres_layer = nn.Threshold(threshold, 0)
        
        
        # feed forward layers
        self.dec_sequential = nn.Sequential(
                                            
                                nn.Linear(self.zdim, 256),
                                nn.ReLU(),
                                nn.BatchNorm1d(256),
                                            
                                nn.Linear(256, 512),
                                nn.ReLU(),
                                nn.BatchNorm1d(512),
                                            
                                nn.Linear(512, 1024),
                                nn.ReLU(),
                                nn.BatchNorm1d(1024),
                                            
                                nn.Linear(1024, self.inp_dim),
                                self.thres_layer
                                
                                # in our experiments, adding the LSN did not improve our results
                                ## but in case if you want to try it
                                #self.lsn
                                            )
    
    def forward(self, z):
        """
            
        Forward pass of the Decoder
            
        """
        
        out =  self.dec_sequential(z);
        return out



