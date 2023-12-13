import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from typing import Iterable

"""Encoder module"""

class Encoder(nn.Module):
    """
    This class constructs an Encoder module for a variational autoencoder.
    Inspired by SCVI FCLayers class.

    Parameters
    ----------
    in_features
        # of features that are used as input
    latent_dim 
        latent dimension
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    hidden_layers
        number of hidden layers
    neurons_per_layer
        number of neurons per hidden layer
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    activation_fn
        Which activation function to use
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer (True), or just the first (False).
    drop
        dropout rate
    z_drop
        dropout rate for latent space
    """

    def __init__(self, 
                 in_features: int, 
                 latent_dim: int, 
                 n_cat_list: Iterable[int] = None,
                 hidden_layers: int = 1,
                 neurons_per_layer: int = 512,
                 use_batch_norm: bool = True,
                 use_layer_norm: bool = False,
                 use_activation: bool = True,
                 activation_fn: nn.Module = nn.ReLU,
                 bias: bool = True,
                 inject_covariates: bool = True,
                 drop: float = 0.2, 
                 z_drop: float = 0.5):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.in_features = in_features
        self.layer_dims = [neurons_per_layer] * hidden_layers
        self.layer_nums = [self.layer_dims[i:i+2] for i in range(len(self.layer_dims)-1)]
        self.latent_dim = latent_dim
        self.drop = drop
        self.z_drop = z_drop

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        self.inject_covariates = inject_covariates
        self.cat_dim = sum(self.n_cat_list)

        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.in_features + self.cat_dim, self.layer_dims[0], bias=bias),
                    nn.BatchNorm1d(self.layer_dims[0]) if use_batch_norm else None,
                    nn.LayerNorm(self.layer_dims[0]) if use_layer_norm else None,
                    activation_fn() if use_activation else None,
                    nn.Dropout(p=self.drop) if self.drop > 0 else None
                )
            ] +

            [build_block(ins = x[0],
                outs = x[1],
                cat_dim = self.cat_dim,
                use_batch_norm = use_batch_norm,
                use_layer_norm = use_layer_norm,
                use_activation = use_activation,
                activation_fn = activation_fn,
                bias = bias,
                inject_covariates = inject_covariates,
                drop = self.drop
            ) for x in self.layer_nums] 
        ).to(self.device)

        self.mu = nn.Sequential(
            nn.Linear(self.layer_dims[-1] + self.cat_dim * inject_covariates, self.latent_dim),
            nn.Dropout(p=self.z_drop) if self.z_drop > 0 else None
        ).to(self.device)

        self.logvar = nn.Sequential(
            nn.Linear(self.layer_dims[-1] + self.cat_dim * inject_covariates, self.latent_dim),
            nn.Dropout(p=self.z_drop) if self.z_drop > 0 else None
        ).to(self.device)


    def forward(self, x: torch.tensor, cat_list: Iterable[torch.tensor]):
        """
        Forward computation on minibatch of samples.
        
        Parameters
        ----------
        x
            torch.tensor of shape (minibatch, in_features)
        cat_list
            Iterable of torch.tensors containing the category memberships
            shape of each tensor is (minibatch, 1)
        """

        if self.cat_dim > 0:
            categs = []
            for n_cat, cat in zip(self.n_cat_list, cat_list):
                if n_cat > 1:
                    categs.append(one_hot(cat.long(), n_cat).squeeze())
            categs = torch.hstack(categs)
            c = torch.hstack((x, categs))
        else:
            c = x

        for i, block in enumerate(self.encoder):
            if i == 0:
                for layer in block:
                    if layer is not None:
                        c = layer(c)
            else:
                for layer in block:
                    if layer is not None:
                        if self.cat_dim > 0 and self.inject_covariates and isinstance(layer, nn.Linear):
                            c = layer(torch.hstack((c, categs)))
                        else:
                            c = layer(c)
    
        if self.cat_dim > 0 and self.inject_covariates :
            c = torch.hstack((c, categs))

        mu = self.mu(c)
        log_var = self.logvar(c)

        return mu, log_var



class Decoder(nn.Module):
    """
    This class constructs a Decoder module for a variational autoencoder.
    Inspired by SCVI FCLayers class.

    Parameters
    ----------
    in_features
        # of features that will be reconstructed
    latent_dim
        input dimension
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    hidden_layers
        number of hidden layers
    neurons_per_layer
        number of neurons per hidden layer
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    activation_fn
        Which activation function to use
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer (True), or just the first (False).
    drop
        dropout rate
    """

    def __init__(self, 
                 in_features: int, 
                 latent_dim: int, 
                 n_cat_list: Iterable[int] = None,
                 hidden_layers: int = 1,
                 neurons_per_layer: int = 512,
                 use_batch_norm: bool = True,
                 use_layer_norm: bool = False,
                 use_activation: bool = True,
                 activation_fn: nn.Module = nn.ReLU,
                 bias: bool = True,
                 inject_covariates: bool = True,
                 drop: float = 0.2):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.layer_dims = [neurons_per_layer] * hidden_layers
        self.layer_nums = [self.layer_dims[i:i+2] for i in range(len(self.layer_dims)-1)]
        self.drop = drop

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        self.inject_covariates = inject_covariates
        self.cat_dim = sum(self.n_cat_list)

        self.decoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.latent_dim + self.cat_dim, self.layer_dims[0], bias=bias),
                    nn.BatchNorm1d(self.layer_dims[0]) if use_batch_norm else None,
                    nn.LayerNorm(self.layer_dims[0]) if use_layer_norm else None,
                    activation_fn() if use_activation else None,
                    nn.Dropout(p=self.drop) if self.drop > 0 else None
                )
            ] +

            [build_block(ins = x[0],
                outs = x[1],
                cat_dim = self.cat_dim,
                use_batch_norm = use_batch_norm,
                use_layer_norm = use_layer_norm,
                use_activation = use_activation,
                activation_fn = activation_fn,
                bias = bias,
                inject_covariates = inject_covariates,
                drop = self.drop
            ) for x in self.layer_nums[:-1]] +

            [
                nn.Sequential(
                    nn.Linear(self.layer_nums[-1][0] + self.cat_dim * self.inject_covariates, self.in_features)
                )
            ]
            ).to(self.device)
       

    def forward(self, x: torch.tensor, cat_list: Iterable[torch.tensor]):
        """
        Forward computation on minibatch of samples.
        
        Parameters
        ----------
        x
            torch.tensor of shape (minibatch, in_features)
        cat_list
            Iterable of torch.tensors containing the category memberships
            shape of each tensor is (minibatch, 1)
        """

        if self.cat_dim > 0:
            categs = []
            for n_cat, cat in zip(self.n_cat_list, cat_list):
                if n_cat > 1:
                    categs.append(one_hot(cat.long(), n_cat).squeeze())
            categs = torch.hstack(categs)
            c = torch.hstack((x, categs))
        else:
            c = x

        for i, block in enumerate(self.decoder[:-1]):
            if i == 0:
                for layer in block:
                    if layer is not None:
                        c = layer(c)
            else:
                for layer in block:
                    if layer is not None:
                        if self.cat_dim > 0 and self.inject_covariates and isinstance(layer, nn.Linear):
                            c = layer(torch.hstack((c, categs)))
                        else:
                            c = layer(c)
    
        if self.cat_dim > 0 and self.inject_covariates :
            out = torch.hstack((c, categs))
        else:
            out = c

        for layer in self.decoder[-1]:
            if layer is not None:
                out = layer(out)
        
        return out




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
    neuronnum
        number of neurons to use per term
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    activation_fn
        Which activation function to use
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer (True), or just the last (False).
    drop
        dropout rate
    """ 

    def __init__(self, 
                 in_features: int, 
                 layer_dims: list, 
                 mask_list: list, 
                 latent_dim: int, 
                 neuronnum: int = 3,
                 n_cat_list: Iterable[int] = None,
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 use_activation: bool = False,
                 activation_fn: nn.Module = nn.ReLU,
                 bias: bool = True,
                 inject_covariates: bool = False,
                 drop: float = 0):
        super().__init__()

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
        self.drop = drop

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        self.inject_covariates = inject_covariates
        self.cat_dim = sum(self.n_cat_list)

        self.decoder = nn.ModuleList(

            [build_block(ins = x[0],
                outs = x[1],
                cat_dim = self.cat_dim,
                use_batch_norm = use_batch_norm,
                use_layer_norm = use_layer_norm,
                use_activation = use_activation,
                activation_fn = activation_fn,
                bias = bias,
                inject_covariates = inject_covariates,
                drop = self.drop
            ) for x in self.layer_shapes[:-1]] +

            [
                nn.Sequential(
                    nn.Linear(self.layer_shapes[-1][0] + self.cat_dim, self.in_features)
                )
            ]
            ).to(self.device)
        
        # attach covs to masks (set to 1s)
        if len(self.n_cat_list) > 0:
            if inject_covariates:
                self.layer_shapes = [(lshape[0] + self.cat_dim, lshape[1]) for lshape in self.layer_shapes]
                self.masks = [torch.hstack((mask, torch.ones(mask.shape[0], self.cat_dim).to(self.device))) for mask in self.masks]
            else:
                self.layer_shapes[-1] = (self.layer_shapes[-1][0] + self.cat_dim, self.layer_shapes[-1][1]) 
                self.masks[-1] = torch.hstack((self.masks[-1], torch.ones(self.masks[-1].shape[0], self.cat_dim).to(self.device))) 

        # apply masks to zero out weights of non-existent connections
        for i in range(len(self.decoder)):
            self.decoder[i][0].weight.data = torch.mul(self.decoder[i][0].weight.data, self.masks[i])

        # make all weights in decoder positive
        for i in range(len(self.decoder)):
            self.decoder[i][0].weight.data = self.decoder[i][0].weight.data.clamp(0)


    def forward(self, z: torch.tensor, cat_list: Iterable[torch.tensor]):
        """
        Forward computation on minibatch of samples.
        
        Parameters
        ----------
        z
            torch.tensor of shape (minibatch, in_features)
        cat_list
            Iterable of torch.tensors containing the category memerships
            shape of each tensor is (minibatch, 1)
        """

        if self.cat_dim > 0:
            categs = []
            for n_cat, cat in zip(self.n_cat_list, cat_list):
                if n_cat > 1:
                    categs.append(one_hot(cat.long(), n_cat).squeeze())
            categs = torch.hstack(categs)

        out = z.clone()

        for block in self.decoder[:-1]:
            for layer in block:
                if layer is not None:
                    if self.cat_dim > 0 and self.inject_covariates and isinstance(layer, nn.Linear):
                        z = layer(torch.hstack((z, categs)))
                    else:
                        z = layer(z)
            out = torch.cat((z, out), dim=1)
            z = out.clone()
        
        if self.cat_dim > 0:
            out = torch.hstack((out, categs))

        for layer in self.decoder[-1]:
            if layer is not None:
                out = layer(out)
        
        return out


"""Classifier module"""
class Classifier(nn.Module):
    """
    Classifier module that can do binary or multi-class classification
    Parameters
    -------------
    in_features
        # of features that are used as input
    n_classes 
        number of classes
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    hidden_layers
        number of hidden layers
    neurons_per_layer
        number of neurons in a hidden layer
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    activation_fn
        Which activation function to use
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer (True), or just the first (False).
    drop
        dropout rate
    """
    def __init__(self, 
                 in_features: int, 
                 n_classes: int, 
                 n_cat_list: Iterable[int] = None,
                 hidden_layers: int = 1,
                 neurons_per_layer: int = 64,
                 use_batch_norm: bool = True,
                 use_layer_norm: bool = False,
                 use_activation: bool = True,
                 activation_fn: nn.Module = nn.ReLU,
                 bias: bool = True,
                 inject_covariates: bool = True,
                 drop: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.layer_dims = [neurons_per_layer] * hidden_layers
        self.layer_nums = [self.layer_dims[i:i+2] for i in range(len(self.layer_dims)-1)]
        self.n_classes = n_classes
        self.drop = drop
        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []
        self.inject_covariates = inject_covariates
        self.cat_dim = sum(self.n_cat_list)

        self.classifier = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.in_features + self.cat_dim, self.layer_dims[0], bias=bias),
                    nn.BatchNorm1d(self.layer_dims[0]) if use_batch_norm else None,
                    nn.LayerNorm(self.layer_dims[0]) if use_layer_norm else None,
                    activation_fn() if use_activation else None,
                    nn.Dropout(p=self.drop) if self.drop > 0 else None
                )
            ] +
            [build_block(ins = x[0],
                outs = x[1],
                cat_dim = self.cat_dim,
                use_batch_norm = use_batch_norm,
                use_layer_norm = use_layer_norm,
                use_activation = use_activation,
                activation_fn = activation_fn,
                bias = bias,
                inject_covariates = inject_covariates,
                drop = self.drop
            ) for x in self.layer_nums] +
            [
                nn.Sequential(
                    nn.Linear(self.layer_dims[-1] + self.cat_dim * inject_covariates, self.n_classes, bias=bias),
                    nn.BatchNorm1d(self.n_classes) if use_batch_norm else None,
                    nn.LayerNorm(self.n_classes) if use_layer_norm else None,
                    nn.Softmax() if self.n_classes > 2 else nn.Sigmoid()
                )
            ]

        )
    
    def forward(self, x: torch.tensor, cat_list: Iterable[torch.tensor]):
        """
        Forward computation on minibatch of samples.
        
        Parameters
        ----------
        x
            torch.tensor of shape (minibatch, in_features)
        cat_list
            Iterable of torch.tensors containing the category memberships
            shape of each tensor is (minibatch, 1)
        """
        if self.cat_dim > 0:
            categs = []
            for n_cat, cat in zip(self.n_cat_list, cat_list):
                if n_cat > 1:
                    categs.append(one_hot(cat.long(), n_cat).squeeze())
            categs = torch.hstack(categs)
            c = torch.hstack((x, categs))
        else:
            c = x
        for i, block in enumerate(self.classifier):
            if i == 0:
                for layer in block:
                    if layer is not None:
                        c = layer(c)
            else:
                for layer in block:
                    if layer is not None:
                        if self.cat_dim > 0 and self.inject_covariates and isinstance(layer, nn.Linear):
                            c = layer(torch.hstack((c, categs)))
                        else:
                            c = layer(c)

        return c
    
"""Function to build NN blocks"""

def build_block(ins: int,
                outs: int,
                cat_dim: int,
                use_batch_norm: bool = True,
                use_layer_norm: bool = False,
                use_activation: bool = True,
                activation_fn: nn.Module = nn.ReLU,
                bias: bool = True,
                inject_covariates: bool = True,
                drop: float = 0.2, 
                ):
    return nn.Sequential(
            nn.Linear(ins + cat_dim * inject_covariates, outs, bias=bias),
            nn.BatchNorm1d(outs) if use_batch_norm else None,
            nn.LayerNorm(outs) if use_layer_norm else None,
            activation_fn() if use_activation else None,
            nn.Dropout(p=drop) if drop > 0 else None
        )