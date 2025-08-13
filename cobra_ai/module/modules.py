import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from typing import Iterable, Literal
from torch.autograd import Function


"""Encoder module"""

class Encoder(nn.Module):
    """
    This class constructs an Encoder module for a variational autoencoder.

    Parameters
    ----------
    n_features
        # of features that are used as input
    hidden_dims
        A list of integers indicating the number of nodes in each hidden layer
    latent_dim 
        latent dimension
    batch_names
        A list of strings indicating the batch names
    normalisation
        Which normalisation to use, either "batch" or "layer"
    activation_fn
        Which activation function to use
    bias
        Whether to learn bias in linear layers or not
    dropout_rate
        dropout rate
    """

    def __init__(
            self, 
            n_features: int, 
            hidden_dims: Iterable[int],
            latent_dim: int, 
            batch_names: Iterable[str] = [],
            normalisation: Literal["batch", "layer"] = "batch",
            activation_fn: nn.Module = nn.ReLU,
            bias: bool = True,
            dropout_rate: float = 0.2
            ):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.encoder = FCLayers(
            layer_dims = [n_features] + hidden_dims,
            batch_names = batch_names,
            normalisation = normalisation,
            activation_fn = activation_fn,
            bias = bias,
            dropout_rate = dropout_rate
        )

        self.mu = nn.Sequential(
            nn.Linear(hidden_dims[-1], latent_dim),
        ).to(self.device)

        self.logvar = nn.Sequential(
            nn.Linear(hidden_dims[-1], latent_dim),
        ).to(self.device)


    def forward(self, x: torch.tensor):
        """
        Forward computation on minibatch of samples.
        
        Parameters
        ----------
        x
            torch.tensor of shape (minibatch, n_features + n_batches)
        """

        c = self.encoder(x)

        mu = self.mu(c)
        log_var = self.logvar(c)

        return mu, log_var


"""Decoder module"""

class Decoder(nn.Module):
    """
    This class constructs a Decoder module for a variational autoencoder.
   
    Parameters
    ----------
    n_features
        # of features that will be reconstructed
    hidden_dims
        A list of integers indicating the number of nodes in each hidden layer
    latent_dim
        input dimension
    batch_names
        A list of strings indicating the batch names
    normalisation
        Which normalisation to use, either "batch" or "layer"
    activation_fn
        Which activation function to use
    bias
        Whether to learn bias in linear layers or not
    dropout_rate
        dropout rate
    """

    def __init__(
            self, 
            n_features: int, 
            hidden_dims: Iterable[int],
            latent_dim: int, 
            batch_names: Iterable[str] = [],
            normalisation: Literal["batch", "layer"] = "batch",
            activation_fn: nn.Module = nn.ReLU,
            bias: bool = True,
            dropout_rate: float = 0.2,
            ):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.decoder = FCLayers(
            layer_dims = [latent_dim] + hidden_dims,
            batch_names = batch_names,
            normalisation = normalisation,
            activation_fn = activation_fn,
            bias = bias,
            dropout_rate = dropout_rate
        )

        self.reconstruction = nn.Sequential(
            nn.Linear(hidden_dims[-1], n_features)
        ).to(self.device)


    def forward(self, x: torch.tensor):
        """
        Forward computation on minibatch of samples.
        
        Parameters
        ----------
        x
            torch.tensor of shape (minibatch, n_features + n_batches)
        """

        c = self.decoder(x)
        out = self.reconstruction(c)
        
        return out




"""Ontology guided decoder module"""


class OntoDecoder(nn.Module):
    """
    This class constructs an ontology structured Decoder module.
  
    Parameters
    ----------
    n_features
        # of features that are used as input
    layer_dims
        list of tuples that define in and out for each layer
    mask_list
        matrix for each layer transition, that determines which weights to zero out
    root_layer_latent
        whether latent space layer is set as first ontology layer (True, default) or first decoder layer (False)
    latent_dim
        latent dimension
    batch_names
        A list of strings indicating the batch names
    neuronnum
        number of neurons to use per term
    normalisation
        which normalisation to use, either "batch" or "layer"
    activation_fn
        Which activation function to use
    bias
        whether to learn bias in linear layers
    dropout_rate
        dropout rate
    pos_weights
        whether to make all decoder weights positive
    """ 

    def __init__(self, 
                 n_features: int, 
                 layer_dims: list, 
                 mask_list: list, 
                 root_layer_latent: bool = True,
                 latent_dim: int = 128, 
                 batch_names: Iterable[str] = [],
                 neuronnum: int = 3,
                 normalisation: Literal["batch", "layer"] = "batch",
                 activation_fn: nn.Module = nn.ReLU,
                 bias: bool = True,
                 dropout_rate: float = 0.0,
                 pos_weights: bool = True,
                 linear_decoder: bool = False):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.root_layer_latent = root_layer_latent
        self.start_point = 0 if root_layer_latent else 1
        self.layer_dims = np.hstack([layer_dims[:-1] * neuronnum, layer_dims[-1]])
        self.masks = []
        for m in mask_list[0:-1]:
            m = m.repeat_interleave(neuronnum, dim=0)
            m = m.repeat_interleave(neuronnum, dim=1)
            self.masks.append(m.to(self.device))
        self.masks.append(mask_list[-1].repeat_interleave(neuronnum, dim=1).to(self.device))
        self.pos_weights = pos_weights

        if not root_layer_latent:
            self.layer_dims = np.insert(self.layer_dims, 0, latent_dim)

        self.decoder = FCLayers(
            layer_dims = self.layer_dims,
            skip_connections = True,
            root_layer_latent = root_layer_latent,
            batch_names= batch_names,
            normalisation = normalisation,
            activation_fn = activation_fn,
            bias = bias,
            dropout_rate = dropout_rate,
        )
        
        # if OntoDecoder should be linear, strip down FCLayers
        if linear_decoder == True:
            for idx, block in enumerate(self.decoder.fc_layers):
                self.decoder.fc_layers[idx] = nn.Sequential(block[0])

        self.decoder.to(self.device)

        self.reconstruction = nn.Sequential(
            nn.Linear(self.decoder.fc_layers[-1][0].in_features + self.decoder.fc_layers[-1][0].out_features, n_features)
        ).to(self.device)

        # apply masks to zero out weights of non-existent connections
        for i in range(self.start_point,len(self.decoder.fc_layers)):
            self.decoder.fc_layers[i][0].weight.data = torch.mul(self.decoder.fc_layers[i][0].weight.data, self.masks[i-self.start_point])
            self.reconstruction[0].weight.data = torch.mul(self.reconstruction[0].weight.data, self.masks[-1])

        # make all weights in decoder positive
        if self.pos_weights:
            for i in range(self.start_point, len(self.decoder.fc_layers)):
                self.decoder.fc_layers[i][0].weight.data = self.decoder.fc_layers[i][0].weight.data.clamp(0)
                self.reconstruction[0].weight.data = self.reconstruction[0].weight.data.clamp(0)

    def forward(self, z: torch.tensor):
        """
        Forward computation on minibatch of samples.
        
        Parameters
        ----------
        z
            torch.tensor of shape (minibatch, in_features)
        """

        z_features, x_batch = torch.split(z, [self.layer_dims[0], z.shape[1]-self.layer_dims[0]], dim=1)

        # if ontology does not start in latent space layer, run data through first linear layer
        if not self.root_layer_latent:
            out = self.decoder.fc_layers[0][0](z_features)
            # and if batch information is provided, pass through batch layers and add to output
            if x_batch.sum() > 0:
                for idx, layer in enumerate(self.decoder.batch_layers.values()):
                    out = out + layer(x_batch[:, idx].unsqueeze(1))
            # pass through the remaining layers of the first block
            for i, layer in enumerate(self.decoder.fc_layers[0]):
                if i != 0 and layer is not None:
                    out = layer(out)
        else:
            out = z_features.clone()

        # pass through the ontology blocks
        z = out.clone()
        for block in self.decoder.fc_layers[self.start_point:]:
            for layer in block:
                if layer is not None:
                    z = layer(z)
            out = torch.cat((z, out), dim=1)
            z = out.clone()
        
        # pass through reconstruction layer
        out = self.reconstruction(z)
        # add batch information if provided and not added before
        if self.root_layer_latent and x_batch.sum() > 0:
            for idx, layer in enumerate(self.decoder.batch_layers.values()):
                out = out + layer(x_batch[:, idx].unsqueeze(1))
        
        return out
    

"""Classifier"""

class Classifier(nn.Module):
    def __init__(
            self, 
            n_features: int, 
            hidden_dims: Iterable[int],
            n_classes: Iterable[str] = [],
            normalisation: Literal["batch", "layer"] = "batch",
            activation_fn: nn.Module = nn.ReLU,
            bias: bool = True,
            dropout_rate: float = 0.2
            ):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.layer_dims = [n_features] + hidden_dims
        self.layer_nums = [self.layer_dims[i:i+2] for i in range(len(self.layer_dims)-1)] # split into pairs of in and out for each layer

        self.classifier = []
        [self.classifier.extend(
            build_block(
                ins = x[0],
                outs = x[1],
                normalisation = normalisation,
                activation_fn = activation_fn,
                bias = bias,
                dropout_rate = dropout_rate,
             )) 
        for x in self.layer_nums
        ] 
        self.classifier.append(nn.Linear(self.layer_dims[-1], n_classes, bias=bias))
        self.classifier = nn.Sequential(*self.classifier).to(self.device)  
    
    def forward(self, z_input: torch.tensor):
        return self.classifier(z_input)




class FCLayers(nn.Module):
    """
    A class to build fully connected layers with the possibility of including batch information.

    Parameters
    ----------
    layer_dims
        A list of integers indicating the number of nodes in each layer
    skip_connections
        Whether to use skip connections between layers (for OntoVAE, default: False)
    root_layer_latent
        whether the first layer should already implement skip connections (True) or not (False) (for OntoVAE, default: False)
    batch_names
        A list of strings indicating the batch names
    normalisation
        Which normalisation to use, either "batch" or "layer"
    activation_fn
        Which activation function to use
    bias
        Whether to learn bias in linear layers or not
    dropout_rate
        dropout rate
    """

    def __init__(
            self, 
            layer_dims: Iterable[int],
            skip_connections: bool = False,
            root_layer_latent: bool = False,
            batch_names: Iterable[str] = [],
            normalisation: Literal["batch", "layer"] = "batch",
            activation_fn: nn.Module = nn.ReLU,
            bias: bool = True,
            dropout_rate: float = 0.2,
            ):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.layer_dims = layer_dims
        if skip_connections:
            if root_layer_latent:
                self.layer_nums = [(np.sum(self.layer_dims[:i+1]), self.layer_dims[i+1]) for i in range(len(self.layer_dims)-2)]
            else:
                self.layer_nums = [(layer_dims[0], layer_dims[1])] + [(np.sum(self.layer_dims[1:i+1]), self.layer_dims[i+1]) for i in range(1, len(self.layer_dims)-2)]
        else:
            self.layer_nums = [self.layer_dims[i:i+2] for i in range(len(self.layer_dims)-1)] # split into pairs of in and out for each layer

        # create the fully connected layers for the input features if any
        if self.layer_dims[0] > 0:
            self.fc_layers = nn.Sequential(
                *[
                    nn.Sequential(*build_block(
                        ins=x[0],
                        outs=x[1],
                        normalisation=normalisation,
                        activation_fn=activation_fn,
                        bias=bias,
                        dropout_rate=dropout_rate
                    )) for x in self.layer_nums
                ]
            ).to(self.device) 

        # create the additional batch layers if batch information is provided
        self.batch_names = batch_names
        self.batch_layers = nn.ModuleDict()
        if len(self.batch_names) > 0:
            for batch in batch_names:
                if root_layer_latent:
                    self.batch_layers[batch] = nn.Linear(1, layer_dims[-1]).to(self.device) # batch is input to reconstruction layer
                else:
                    self.batch_layers[batch] = nn.Linear(1, layer_dims[1]).to(self.device) # batch is input to first hidden layer


    def forward(self, x: torch.tensor):
        """
        Forward computation on minibatch of samples.
        
        Parameters
        ----------
        x
           torch.tensor of shape (minibatch, n_features + n_batches), 
        """

        x_features, x_batch = torch.split(x, [self.layer_dims[0], x.shape[1]-self.layer_dims[0]], dim=1)

        # pass through the first linear layer of the first block if there are input features
        if self.layer_dims[0] > 0:
            out = self.fc_layers[0][0](x_features)
        else: 
            out = torch.zeros(x_batch.size(0), self.layer_dims[1], device=self.device)

        # pass through batch layers if batch information is provided
        if x_batch.sum() > 0:
            for idx, layer in enumerate(self.batch_layers.values()):
                out = out + layer(x_batch[:, idx].unsqueeze(1))

        # if there are input features
        if self.layer_dims[0] > 0:
            # pass through remaning layers of first block
            for i, layer in enumerate(self.fc_layers[0]):
                if i != 0 and layer is not None:
                    out = layer(out)
            # pass through remaining blocks
            for block in self.fc_layers[1:]:
                for layer in block:
                        out = layer(out)
        
        return out
    

    def add_batches(self, batch_names: Iterable[str]):
        """
        Add batches to the model.
        
        Parameters
        ----------
        batch_names
            A list of strings indicating the batch names
        """

        # add the batch names to the model
        self.batch_names.extend(batch_names)

        # update the batch_layers ModuleDict
        for batch in batch_names:
            self.batch_layers[batch] = nn.Linear(1, self.layer_dims[1]).to(self.device)




"""Function to build NN blocks"""

def build_block(ins: int,
                outs: int,
                normalisation: Literal["batch", "layer"] = "batch",
                activation_fn: nn.Module = nn.ReLU,
                bias: bool = True,
                dropout_rate: float = 0.2, 
                ):
    block = [
            nn.Linear(ins, outs, bias=bias),
            nn.BatchNorm1d(outs) if normalisation == "batch" else nn.LayerNorm(outs),
            activation_fn(),
            nn.Dropout(p=dropout_rate),
    ]
    return block


"""GradientReversal"""

class GradientReversalFunction(Function):
    """
    Class to reverse the gradient during adversarial training
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)  # Identity in forward

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None  # Reverses gradient
    


class GradientReversalLayer(nn.Module):
    """
    Class that implements a Gradient Reversal Layer (GRL)
    """
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
