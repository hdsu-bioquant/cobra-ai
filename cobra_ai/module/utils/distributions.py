import torch
import torch.nn as nn


class Reparameterize(nn.Module):
    """
    Class that implements the reparameterization trick.
    Noise is added to the latent space sampling to allow gradient backpropagation.
    """
    def __init__(self):
        super().__init__()

    def forward(self, mu, log_var):
        """
        Parameters
        ----------
        mu
            mean vector
        log_var
            variance vector
        """
        sigma = torch.exp(0.5*log_var) 
        eps = torch.randn_like(sigma) 
        return mu + eps * sigma