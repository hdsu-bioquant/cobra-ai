#!/usr/bin/env python3
import json
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal

import anndata as ad
from anndata import AnnData


from cobra_ai.module.modules import Encoder, OntoDecoder, GradientReversalLayer, Classifier
from cobra_ai.module.base_vae import BaseVAE
from cobra_ai.module.ontobj import Ontobj

"""VAE with ontology in decoder"""

class OntoVAE(BaseVAE):
    """
    This class combines a normal encoder with an ontology structured decoder.
    The input should be log-transformed normalized data. 
    Mainly for single-cell, but also works with bulk data if stored in adata.
    """

    @classmethod
    def _read_params(
        cls,
        modelpath
        ):
        with open(modelpath + '/model_hyperparams.json', 'r') as fp:
            params = json.load(fp)
        if params['activation_fn'] is not None:
            params['activation_fn'] = eval(params['activation_fn'])
        params['onto_annot'] = pd.DataFrame(params['onto_annot'])
        params['masks'] = [np.array(m) for m in params['masks']]
        params['trained'] = True
        return params

    def __init__(self):
        super().__init__()

    def construct(
            self, 
            ontobj: Ontobj = None, 
            adata: AnnData = None, 
            batch_key: str = None,
            top_thresh: int = None,
            bottom_thresh: int = None,
            keep_genes: bool = True,
            latent_dim: int = 128,
            root_layer_latent: bool = False,
            neuronnum: int = 3,
            hidden_dims: list = [256,256],
            hidden_dims_classifier: list = [64],
            normalisation: Literal["batch", "layer"] = "batch",
            activation_fn: nn.Module = nn.ReLU,
            bias: bool = True,
            dropout_rate: float = 0.2,
            z_dropout: float = 0.5,
            pos_weights: bool = True,
            linear_decoder: bool = False,
            input_features: list = [],
            onto_features: list = [],
            masks: list = [],
            onto_annot: dict = {},
            batch_idx: dict = {},
            trained: bool = False
            ):
        """
        Function to construct the OntoVAE model based on an Ontobj.

        Parameters
        ----------
        ontobj
            Ontobj object containing the ontology information
        adata
            anndata object containing the RNA data 
        batch_key
            column in adata.obs that contains batch information (optional)
        latent_dim
            latent dimension for RNA modality
        root_layer_latent
            whether the first ontology layer should be located in the latent space (True) or in the first decoder layer (False)
        neuronnum
            number of neurons per ontology term
        hidden_dims
            A list of integers indicating the number of nodes in each hidden layer of the RNA encoder
        hidden_dims_dec
            if none, rna_hidden_dims_dec = rna_hidden_dims[::-1]
        hidden_dims_classifier
            A list of integers indicating the number of nodes in each hidden layer of the batch classifier
        normalisation
            which normalisation to use, either "batch" or "layer"
        activation_fn
            Which activation function to use
        bias 
            whether to learn bias in linear layers
        dropout_rate
            dropout rate
        z_dropout
            dropout rate for the latent space
        pos_weights
            whether to use positive weights in the decoder
        input_features
            list of input features (used internally when loading a trained model)
        onto_features
            list of ontology features (used internally when loading a trained model)
        masks
            list of decoder masks (used internally when loading a trained model)
        onto_annot
            dataframe with ontology annotation (used internally when loading a trained model)
        batch_idx
            dictionary of batch to index mapping (used internally when loading a trained model)
        trained
            whether the model is already trained or not (used internally when loading a trained model)
        """

        self.model = 'ontovae'
        self.trained = trained
        if not self.trained:
            self.val_loss_min = float('inf')

        # general parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hyperparams = {}
        self.adata = adata
        self.input_features = input_features
        self.z_dropout = z_dropout

        # parse batch information
        self.batch_key = batch_key
        self.batch_names = []
        self.new_batch_names = []
        self.batch_idx = batch_idx
        if self.batch_idx is not None:
            self.batch_names = list(self.batch_idx.keys())
        
        # OntoVAE specific
        self.keep_genes = keep_genes
        self.neuronnum = neuronnum
        self.root_layer_latent = root_layer_latent
        self.start_point = 0 if self.root_layer_latent else 1
        self.pos_weights = pos_weights
        if self.trained:   
            self.onto_features = onto_features
            self.masks = masks
            self.onto_annot = onto_annot
        
        # parse ontology information
        self._parse_ontobj(
                ontobj=ontobj,
                top_thresh=top_thresh,
                bottom_thresh=bottom_thresh,
                latent_dim = latent_dim
        )
        
        # parse adata and input features
        if adata is not None:
            self._match_adata(
                adata,
                input_features = input_features,
            )

        # update batch information if adata is provided
        if adata is not None and batch_key is not None:
            self._update_batch(
                adata,
                batch_key,
                add_layers=False
            )

        # store model construction hyperparams in dict
        self.hyperparams.update({
            'input_features': self.input_features,
            'keep_genes': keep_genes,
            'latent_dim': latent_dim,
            'root_layer_latent': root_layer_latent,
            'neuronnum': neuronnum,
            'hidden_dims': hidden_dims,
            'normalisation': normalisation,
            'activation_fn': str(activation_fn).split("'")[1],
            'bias': bias,
            'dropout_rate': dropout_rate,
            'z_dropout': z_dropout,
            'pos_weights': pos_weights, 
            'linear_decoder': linear_decoder
        })

        # Encoder
        self.encoder = Encoder(
            n_features = len(self.input_features),
            hidden_dims = hidden_dims,
            latent_dim = self.latent_dim,
            batch_names = self.batch_names,
            normalisation = normalisation,
            activation_fn = activation_fn,
            bias = bias,
            dropout_rate = dropout_rate 
        )

        # Decoder
        self.decoder = OntoDecoder(
            n_features = len(self.onto_features),
            layer_dims = self.layer_dims_dec,
            mask_list = self.mask_list,
            root_layer_latent = self.root_layer_latent,
            latent_dim = self.latent_dim,
            batch_names = self.batch_names,
            neuronnum = self.neuronnum,
            normalisation = normalisation,
            activation_fn = activation_fn,
            bias = bias,
            dropout_rate = dropout_rate,
            pos_weights = self.pos_weights,
            linear_decoder = linear_decoder 
        )

        # optional: batch alignment
        if self.batch_key or len(self.batch_idx) > 0:
            self.grl = GradientReversalLayer()
            self.batch_classifier = Classifier(
                n_features = self.latent_dim,
                hidden_dims = hidden_dims_classifier,
                n_classes = len(self.batch_idx),
                normalisation = normalisation,
                activation_fn = activation_fn,
                bias = bias,
                dropout_rate = dropout_rate
            )

        self.to(self.device)

        self.initialized = True


    def _parse_ontobj(
            self, 
            ontobj: Ontobj = None,
            top_thresh: int = None,
            bottom_thresh: int = None,
            latent_dim: int = 128,
            ):
        """
        Helper function to parse the Ontobj and set up the model accordingly.
        """
        # repopulate the slots from ontobj
        if ontobj is not None:
            if not isinstance(ontobj, Ontobj):
                raise ValueError("ontobj must be an instance of Ontobj")
            
            # retrive trimming thresholds or extract from ontobj if not provided
            if top_thresh is not None and bottom_thresh is not None:
                if not str(top_thresh) + '_' + str(bottom_thresh) in ontobj.annot.keys():
                    raise ValueError('Available trimming thresholds are: ' + ', '.join(list(ontobj.annot.keys())))
            else:
                top_thresh = list(ontobj.annot.keys())[0].split('_')[0]
                bottom_thresh = list(ontobj.annot.keys())[0].split('_')[1]

            # retrieve features
            self.onto_features = ontobj.extract_genes(
                top_thresh = top_thresh,
            )

            # retrieve ontology annotations
            self.onto_annot = ontobj.extract_annot(
                top_thresh = top_thresh,
                bottom_thresh = bottom_thresh
            )

            # retrieve decoder masks
            self.masks = ontobj.extract_masks(
                top_thresh = top_thresh,
                bottom_thresh = bottom_thresh
            )

        # set masks and layer dimensions
        self.mask_list = [torch.tensor(m, dtype=torch.float32) for m in self.masks]
        self.layer_dims_dec =  np.array([self.mask_list[0].shape[1]] + [m.shape[0] for m in self.mask_list])
        self.latent_dim = self.layer_dims_dec[0] * self.neuronnum if self.root_layer_latent else latent_dim

        # update hyperparams with ontology information
        self.hyperparams.update({
            'onto_features': self.onto_features,
            'onto_annot': self.onto_annot.to_dict(orient='records'),
            'masks': [m.tolist() for m in self.masks]
        })

        if len(self.onto_features) == 0:
            raise ValueError("No ontology features found. Please check the Ontobj or input parameters.")
        

    def _match_adata(
            self,
            adata: AnnData,
            input_features: list,
            return_adata: bool = False
            ):
        """
        Helper function to match the adata to the ontology and input features.
        """
        adata = adata.copy()

        # necessary cleanup of adata
        if len(list(adata.layers.keys())) > 0:
            for k in list(adata.layers.keys()):
                del adata.layers[k]

        adata.varm = ""
        adata_genes = list(adata.var_names)

        # define input features based on adata and param
        if len(input_features) > 0 and self.keep_genes:
            genes = input_features
        else:
            genes = self.onto_features

        # raise error if no input genes are found
        if len([g for g in adata.var_names if g in genes]) == 0:
            raise ValueError("No input gene could be mapped to the ontology. Please check if you are using the correct gene identifiers.")

        # split adata into target and additional genes
        adata_target = adata[:,adata.var_names.isin(genes)].copy()
        adata_add = adata[:, ~adata.var_names.isin(genes)].copy()

        # look for missing genes and create dummy adata
        missing_genes = [g for g in genes if g not in adata_genes]
        counts = csr_matrix(np.zeros((adata.shape[0], len(missing_genes)), dtype=np.float32))
        ddata = AnnData(counts)
        ddata.obs_names = adata.obs_names
        ddata.var_names = missing_genes

        # concatenate everything
        ndata = ad.concat([adata_target, ddata], join="outer", axis=1)
        ndata = ndata[:,genes]
        if self.keep_genes:
            ndata = ad.concat([ndata, adata_add], join="outer", axis=1)

        # repopulate the adata object
        ndata.obs = adata.obs
        ndata.obsm = adata.obsm

        # redefine the input features
        self.input_features = ndata.var_names.tolist()

        if return_adata:
            return ndata
        else:
            self.adata = ndata


    def _average_neuronnum(self, act: torch.tensor):
        """
        Helper function to calculate the average value of multiple neurons.
        """
        chunks = act.split(self.neuronnum, dim=1)  # splits into chunks of size neuronnum along dim=1
        act = torch.stack(chunks, dim=1).mean(dim=2) # takes the average across the neurons
        
        return act
    
    def _get_activation(self, index, activation={}):
        def hook(model, input, output):
            activation[index] = output
        return hook 
    
    def _attach_hooks(self, lin_layer=True, activation={}, hooks={}):
        """helper function to attach hooks to the decoder"""
        for i in range(len(self.decoder.decoder.fc_layers)): #range(self.start_point, len(self.decoder.decoder.fc_layers)
            key = str(i)
            hook_ind=0 if lin_layer else np.where(np.array(self.decoder.decoder.fc_layers[i]) != None)[0][-1]
            value = self.decoder.decoder.fc_layers[i][hook_ind].register_forward_hook(self._get_activation(i, activation))
            hooks[key] = value

    @torch.no_grad()
    def _hook_activities(self, x, lin_layer=True):
        """
        Attaches hooks and retrieves pathway activities.

        Parameters
        ----------
        x
            torch.tensor of shape (minibatch, in_features)
        lin_layer:
            whether hooks should be attached to linear layer of the model
        """

        # set to eval mode
        self.eval()

        # initialize activations and hooks
        activation = {}
        hooks = {}

        # attach the hooks
        self._attach_hooks(lin_layer=lin_layer, activation=activation, hooks=hooks)

        # pass data through model
        out = self.forward(x)

        act = torch.cat(list(activation.values()), dim=1)
        
        # remove hooks
        for h in hooks:
            hooks[h].remove()

        # return pathway activities or reconstructed gene values
        if self.root_layer_latent:
            return torch.hstack((out['z'], act))
        else:
            return act
        
    @torch.no_grad()
    def get_pathway_activities(
        self, 
        adata: AnnData=None, 
        batch_key: str = None,
        lin_layer: bool=True,
        output_numpy: bool=True,
        return_adata: bool = True
        ):
        """
        Wrapper around _run_batches to retrieve pathway activities.

        Parameters
        ----------
        adata
            AnnData object that was processed with setup_anndata
        lin_layer
            whether linear layer should be used for calculation
        """
        self.eval()

        if return_adata:
            output_numpy = True

        self.batch_key = batch_key

        if self.adata is None and adata is None:
            raise ValueError('Please provide adata.')
        
        if adata is None:
            adata = self.adata

        res = self._run_minibatches(
            adata, 
            retrieve='act', 
            lin_layer=lin_layer,
            output_numpy = output_numpy
            )
        
        if return_adata:
            adata = self._return_adata(
                adata,
                res,
                'pathway_activities'
            )

        return adata if return_adata else res
    
   