import numpy as np
import torch
import torch.nn as nn

"""Encoder module"""

class Encoder(nn.Module):
    """
    This class constructs an Encoder module for a variational autoencoder.

    Parameters
    ----------
    in_features
        # of features that are used as input
    layer_dims
        list giving the dimensions of the hidden layers
    latent_dim 
        latent dimension
    drop
        dropout rate, default is 0.2
    z_drop
        dropout rate for latent space, default is 0.5
    """

    def __init__(self, 
                 in_features: int, 
                 latent_dim: int, 
                 layer_dims: list = [512], 
                 drop: float = 0.2, 
                 z_drop: float = 0.5):
        super(Encoder, self).__init__()

        self.in_features = in_features
        self.layer_dims = layer_dims
        self.layer_nums = [layer_dims[i:i+2] for i in range(len(layer_dims)-1)]
        self.latent_dim = latent_dim
        self.drop = drop
        self.z_drop = z_drop

        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.in_features, self.layer_dims[0]),
                    nn.BatchNorm1d(self.layer_dims[0]),
                    nn.Dropout(p=self.drop),
                    nn.ReLU()
                )
            ] +

            [self.build_block(x[0], x[1]) for x in self.layer_nums] 
        )

        self.mu = nn.Sequential(
            nn.Linear(self.layer_dims[-1], self.latent_dim),
            nn.Dropout(p=self.z_drop)
        )

        self.logvar = nn.Sequential(
            nn.Linear(self.layer_dims[-1], self.latent_dim),
            nn.Dropout(p=self.z_drop)
        )

    def build_block(self, ins, outs):
        return nn.Sequential(
            nn.Linear(ins, outs),
            nn.BatchNorm1d(outs),
            nn.Dropout(p=self.drop),
            nn.ReLU()
        )

    def forward(self, x):

        # encoding
        c = x
        for layer in self.encoder:
            c = layer(c)

        mu = self.mu(c)
        log_var = self.logvar(c)

        return mu, log_var


"""Ontology guided decoder module"""


class OntoDecoder(nn.Module):
    """
    This class constructs an ontology structured Decoder module.
  
    Parameters
    ----------
    in_features
        # of features that are used as input
    layer_dims
        list of tuples that define in and out for each layer
    mask_list
        matrix for each layer transition, that determines which weights to zero out
    latent_dim
        latent dimension
    """ 

    def __init__(self, 
                 in_features: int, 
                 layer_dims: list, 
                 mask_list: list, 
                 latent_dim: int, 
                 neuronnum: int = 3):
        super(OntoDecoder, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.in_features = in_features
        self.layer_dims = np.hstack([layer_dims[:-1] * neuronnum, layer_dims[-1]])
        self.layer_shapes = [(np.sum(self.layer_dims[:i+1]), self.layer_dims[i+1]) for i in range(len(self.layer_dims)-1)]
        self.masks = []
        for m in mask_list[0:-1]:
            m = m.repeat_interleave(neuronnum, dim=0)
            m = m.repeat_interleave(neuronnum, dim=1)
            self.masks.append(m.to(self.device))
        self.masks.append(mask_list[-1].repeat_interleave(neuronnum, dim=1).to(self.device))
        self.latent_dim = latent_dim

        # Decoder
        self.decoder = nn.ModuleList(

            [self.build_block(x[0], x[1]) for x in self.layer_shapes[:-1]] +

            [
                nn.Sequential(
                    nn.Linear(self.layer_shapes[-1][0], self.in_features)
                )
            ]
            ).to(self.device)
        
        # apply masks to zero out weights of non-existent connections
        for i in range(len(self.decoder)):
            self.decoder[i][0].weight.data = torch.mul(self.decoder[i][0].weight.data, self.masks[i])

        # make all weights in decoder positive
        for i in range(len(self.decoder)):
            self.decoder[i][0].weight.data = self.decoder[i][0].weight.data.clamp(0)

    def build_block(self, ins, outs):
        return nn.Sequential(
            nn.Linear(ins, outs)
        )

    def forward(self, z):

        # decoding
        out = z

        for layer in self.decoder[:-1]:
            c = layer(out)
            out = torch.cat((c, out), dim=1)
        reconstruction = self.decoder[-1](out)
        
        return reconstruction