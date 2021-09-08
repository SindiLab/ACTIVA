import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim = 128, input_size = None):
        """
        
        The Encoder class
          
        """
        if latent_dim == None or input_size == None:
            raise ValueError('Must explicitly declare input size and latent space dimension')
            
        super(Encoder, self).__init__();
        self.inp_dim = input_size;
        self.zdim = latent_dim;
        
        # feed forward layers  
        self.enc_sequential = nn.Sequential(
                                nn.Linear(self.inp_dim, 1024),
                                nn.ReLU(),
                                nn.BatchNorm1d(1024),
            
                                nn.Linear(1024, 512),
                                nn.ReLU(),
                                nn.BatchNorm1d(512),
            
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.BatchNorm1d(256),
            
                                nn.Linear(256, 2*self.zdim),
                                nn.ReLU(),
                                nn.BatchNorm1d(2*self.zdim)
                                           )
        
    def forward(self, x):        
        """
        
        Forward pass of the encoder
        
        """

        out = self.enc_sequential(x);
        # get mean and variance 
        mu, variance = out.chunk(2, dim=1)      
        
        return mu, variance
