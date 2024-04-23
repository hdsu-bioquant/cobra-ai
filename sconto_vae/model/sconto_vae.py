#!/usr/bin/env python3
import json
import os

import numpy as np
import pandas as pd
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from typing import Iterable, Literal

from anndata import AnnData

from sconto_vae.module.modules import Encoder, OntoDecoder
from sconto_vae.module.utils import split_adata, FastTensorDataLoader, EarlyStopper, update_bn

from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

# imports for autotuning
from sconto_vae.module.decorators import classproperty
from sconto_vae.module.autotune import Tunable
from ray import train

"""VAE with ontology in decoder"""

class scOntoVAE(nn.Module):
    """
    This class combines a normal encoder with an ontology structured decoder.
    For now, works with log-normalized data,

    Parameters
    ----------
    adata
        anndata object that has been preprocessed with setup_anndata function
    use_batch_norm_enc
        Whether to have `BatchNorm` layers or not in encoder
    use_layer_norm_enc
        Whether to have `LayerNorm` layers or not in encoder
    use_activation_enc
        Whether to have layer activation or not in encoder
    activation_fn_enc
        Which activation function to use in encoder
    bias_enc
        Whether to learn bias in linear layers or not in encoder
    hidden_layers_enc
        number of hidden layers in encoder (number of nodes is determined by neuronnum)
    inject_covariates_enc
        Whether to inject covariates in each layer (True), or just the first (False) of encoder
    drop_enc
        dropout rate in encoder
    z_drop
        dropout rate for latent space 
    root_layer_latent
        whether latent space layer is set as first ontology layer (True, default) or first decoder layer (False)
    latent_dim
        latent space dimension if root_layer_latent is False
    neuronnum
        number of neurons per term in decoder
    use_batch_norm_dec
        Whether to have `BatchNorm` layers or not in decoder
    use_layer_norm_dec
        Whether to have `LayerNorm` layers or not in decoder
    use_activation_dec
        Whether to have layer activation or not in decoder
    use_activation_lat
        Whether to use the decoder activation function after latent space sampling (not recommended)
    activation_fn_dec
        Which activation function to use in decoder
    rec_activation
        activation function for the reconstruction layer, e.g. nn.Sigmoid
    bias_dec
        Whether to learn bias in linear layers or not in decoder
    inject_covariates_dec
        Whether to inject covariates in each layer (True), or just the last (False) of decoder
    drop_dec
        dropout rate in decoder
    """

    @classmethod
    def load(cls, adata: AnnData, modelpath: str):
        with open(modelpath + '/model_params.json', 'r') as fp:
            params = json.load(fp)
        if params['activation_fn_enc'] is not None:
            params['activation_fn_enc'] = eval(params['activation_fn_enc'])
        if params['activation_fn_dec'] is not None:
            params['activation_fn_dec'] = eval(params['activation_fn_dec'])
        if params['rec_activation'] is not None:
            params['rec_activation'] = eval(params['rec_activation'])
        model = cls(adata, **params) 
        checkpoint = torch.load(modelpath + '/best_model.pt',
                            map_location = torch.device(model.device))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        return model

    def __init__(self, 
                 adata: AnnData, 
                 use_batch_norm_enc: Tunable[bool] = True,
                 use_layer_norm_enc: Tunable[bool] = False,
                 use_activation_enc: Tunable[bool] = True,
                 activation_fn_enc: Tunable[nn.Module] = nn.ReLU,
                 bias_enc: Tunable[bool] = True,
                 hidden_layers_enc: Tunable[int]=3, 
                 inject_covariates_enc: Tunable[bool] = False,
                 drop_enc: Tunable[float] = 0.2, 
                 z_drop: Tunable[float] = 0.5,
                 root_layer_latent: Tunable[bool] = False,
                 latent_dim: Tunable[int] = 128,
                 neuronnum: Tunable[int] = 3,
                 use_batch_norm_dec: Tunable[bool] = True,
                 use_layer_norm_dec: Tunable[bool] = False,
                 use_activation_dec: Tunable[bool] = True,
                 use_activation_lat: Tunable[bool] = False,
                 activation_fn_dec: Tunable[nn.Module] = nn.Tanh,
                 rec_activation: nn.Module = None,
                 bias_dec: Tunable[bool] = True,
                 inject_covariates_dec: Tunable[bool] = False,
                 drop_dec: Tunable[float] = 0):
        super().__init__()

        # store init params in dict
        self.params = {'use_batch_norm_enc': use_batch_norm_enc,
                          'use_layer_norm_enc': use_layer_norm_enc,
                          'use_activation_enc': use_activation_enc,
                          'activation_fn_enc': str(activation_fn_enc).split("'")[1] if activation_fn_enc is not None else activation_fn_enc,
                          'bias_enc': bias_enc,
                          'hidden_layers_enc': hidden_layers_enc,
                          'inject_covariates_enc': inject_covariates_enc,
                          'drop_enc': drop_enc,
                          'z_drop': z_drop,
                          'root_layer_latent': root_layer_latent,
                          'latent_dim': latent_dim,
                          'neuronnum': neuronnum,
                          'use_batch_norm_dec': use_batch_norm_dec,
                          'use_layer_norm_dec': use_layer_norm_dec,
                          'use_activation_dec': use_activation_dec,
                          'use_activation_lat': use_activation_lat,
                          'activation_fn_dec': str(activation_fn_dec).split("'")[1] if activation_fn_dec is not None else activation_fn_dec,
                          'rec_activation': str(rec_activation).split("'")[1] if rec_activation is not None else rec_activation,
                          'bias_dec': bias_dec,
                          'inject_covariates_dec': inject_covariates_dec,
                          'drop_dec': drop_dec}


        self.adata = adata

        if '_ontovae' not in self.adata.uns.keys():
            raise ValueError('Please run sconto_vae.module.utils.setup_anndata_ontovae first.')

        # parse OntoVAE information
        self.thresholds = adata.uns['_ontovae']['thresholds']
        self.in_features = len(self.adata.uns['_ontovae']['genes'])
        self.mask_list = adata.uns['_ontovae']['masks']
        self.mask_list = [torch.tensor(m, dtype=torch.float32) for m in self.mask_list]
        self.layer_dims_dec =  np.array([self.mask_list[0].shape[1]] + [m.shape[0] for m in self.mask_list])
        self.root_layer_latent = root_layer_latent
        self.start_point = 0 if self.root_layer_latent else 1
        self.latent_dim = self.layer_dims_dec[0] * neuronnum if self.root_layer_latent else latent_dim
        self.neurons_per_layer_enc = self.latent_dim
        self.z_drop = z_drop

        # additional info
        self.neuronnum = neuronnum
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_activation_dec = use_activation_dec
        self.use_activation_lat = use_activation_lat
        self.activation_fn_dec = activation_fn_dec
        self.rec_activation = rec_activation

        # parse covariate information
        self.batch = adata.obs['_ontovae_batch']
        self.labels = adata.obs['_ontovae_labels']
        self.covs = adata.obsm['_ontovae_categorical_covs'] if '_ontovae_categorical_covs' in adata.obsm.keys() else None

        self.n_cat_list = [len(self.batch.unique()), len(self.labels.unique())]
        if self.covs is not None:
            self.n_cat_list.extend([len(self.covs[c].unique()) for c in self.covs.columns])

        self.rec_weights = None

        # Encoder
        self.encoder = Encoder(in_features = self.in_features,
                                latent_dim = self.latent_dim,
                                n_cat_list = self.n_cat_list,
                                hidden_layers = hidden_layers_enc,
                                neurons_per_layer = self.neurons_per_layer_enc, 
                                use_batch_norm = use_batch_norm_enc,
                                use_layer_norm = use_layer_norm_enc,
                                use_activation = use_activation_enc,
                                activation_fn = activation_fn_enc,
                                bias = bias_enc,
                                inject_covariates = inject_covariates_enc,
                                drop = drop_enc)

        # Decoder
        self.decoder = OntoDecoder(in_features = self.in_features,
                                    layer_dims = self.layer_dims_dec,
                                    mask_list = self.mask_list,
                                    root_layer_latent = self.root_layer_latent,
                                    latent_dim = self.latent_dim,
                                    neuronnum = self.neuronnum,
                                    n_cat_list = self.n_cat_list,
                                    use_batch_norm = use_batch_norm_dec,
                                    use_layer_norm = use_layer_norm_dec,
                                    use_activation = use_activation_dec,
                                    activation_fn = activation_fn_dec,
                                    rec_activation = rec_activation,
                                    bias = bias_dec,
                                    inject_covariates = inject_covariates_dec,
                                    drop = drop_dec)

        self.to(self.device)

    def _cov_tensor(self, adata):
        """
        Helper function to aggregate information from adata to use as input for dataloader.
        """
        covs = adata.obs[['_ontovae_batch', '_ontovae_labels']]
        if '_ontovae_categorical_covs' in adata.obsm.keys():
            covs = pd.concat([covs, adata.obsm['_ontovae_categorical_covs']], axis=1)
        return torch.tensor(np.array(covs))

    def reparameterize(self, mu, log_var):
        """
        Performs the reparameterization trick.

        Parameters
        ----------
        mu
            mean from the encoder's latent space
        log_var
            log variance from the encoder's latent space
        mode
            train: training mode
            val: validation mode
        """
        sigma = torch.exp(0.5*log_var) 
        eps = torch.randn_like(sigma) 
        z = mu + eps * sigma
        if self.z_drop > 0:
            if self.training:
                z = nn.Dropout(p=self.z_drop)(z)
        return z
        
    def _get_embedding(self, x: torch.tensor, cat_list: Iterable[torch.tensor]):
        """
        Generates latent space embedding.

        Parameters
        ----------
        x
            torch.tensor of shape (minibatch, in_features)
        cat_list
            Iterable of torch.tensors containing the category memberships
            shape of each tensor is (minibatch, 1)
        """
        mu, log_var = self.encoder(x, cat_list)
        z = self.reparameterize(mu, log_var)
        if self.use_activation_lat:
            z = self.activation_fn_dec()(z)
        return z, mu, log_var


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

        z, mu, log_var = self._get_embedding(x, cat_list)
        reconstruction = self.decoder(z, cat_list)
        return z, mu, log_var, reconstruction

    def vae_loss(self, reconstruction, mu, log_var, data, kl_coeff, run=None):
        """
        Calculates VAE loss as combination of reconstruction loss and weighted Kullback-Leibler loss.
        """
        kl_loss = -0.5 * torch.sum(1. + log_var - mu.pow(2) - log_var.exp(), )
        if self.rec_weights is not None:
            rec_loss = torch.sum(torch.matmul(input=(data - reconstruction)**2, other=self.rec_weights))
        else:
            rec_loss = F.mse_loss(reconstruction, data, reduction="sum")
        if run is not None:
            mode = 'train' if self.training else 'val'
            run["metrics/" + mode + "/kl_loss"].log(kl_loss)
            run["metrics/" + mode + "/rec_loss"].log(rec_loss)
        return torch.mean(rec_loss + kl_coeff*kl_loss)
    
    def train_round(self, 
                    dataloader: FastTensorDataLoader, 
                    kl_coeff: float, 
                    optimizer: optim.Optimizer, 
                    pos_weights: bool,
                    run=None):
        """
        Parameters
        ----------
        dataloader
            pytorch dataloader instance with training data
        kl_coeff 
            coefficient for weighting Kullback-Leibler loss
        optimizer
            optimizer for training
        run
            Neptune run if training is to be logged
        """
        # set to train mode
        self.train()

        # initialize running loss
        running_loss = 0.0

        # iterate over dataloader for training
        for i, minibatch in tqdm(enumerate(dataloader), total=len(dataloader)):

            # move minibatch to device
            data = torch.tensor(minibatch[0].todense(), dtype=torch.float32).to(self.device)
            cat_list = torch.split(minibatch[1].T.to(self.device), 1)
            optimizer.zero_grad()

            # forward step
            _, mu, log_var, reconstruction = self.forward(data, cat_list)
            loss = self.vae_loss(reconstruction, mu, log_var, data, kl_coeff, run=run)
            running_loss += loss.item()

            # backward propagation
            loss.backward()

            # zero out gradients from non-existent connections
            for i in range(self.start_point, len(self.decoder.decoder)):
                self.decoder.decoder[i][0].weight.grad = torch.mul(self.decoder.decoder[i][0].weight.grad, self.decoder.masks[i-self.start_point])

            # perform optimizer step
            optimizer.step()

            # make weights in Onto module positive
            if pos_weights:
                for i in range(self.start_point, len(self.decoder.decoder)):
                    self.decoder.decoder[i][0].weight.data = self.decoder.decoder[i][0].weight.data.clamp(0)

        # compute avg training loss
        train_loss = running_loss/len(dataloader)
        return train_loss

    @torch.no_grad()
    def val_round(self, 
                  dataloader: FastTensorDataLoader, 
                  kl_coeff: float, 
                  run=None):
        """
        Parameters
        ----------
        dataloader
            pytorch dataloader instance with training data
        kl_coeff
            coefficient for weighting Kullback-Leibler loss
        run
            Neptune run if training is to be logged
        """
        # set to eval mode
        self.eval()

        # initialize running loss
        running_loss = 0.0

        # iterate over dataloader for validation
        for i, minibatch in tqdm(enumerate(dataloader), total=len(dataloader)):

            # move minibatch to device
            data = torch.tensor(minibatch[0].todense(), dtype=torch.float32).to(self.device)
            cat_list = torch.split(minibatch[1].T.to(self.device), 1)

            # forward step
            _, mu, log_var, reconstruction = self.forward(data, cat_list)
            loss = self.vae_loss(reconstruction, mu, log_var,data, kl_coeff, run=run)
            running_loss += loss.item()

        # compute avg val loss
        val_loss = running_loss/len(dataloader)
        return val_loss

    def train_model(self, 
                    modelpath: str, 
                    save: bool = True,
                    train_size: float = 0.9,
                    seed: int = 42,
                    lr: Tunable[float]=1e-4, 
                    kl_coeff: Tunable[float]=1e-4, 
                    batch_size: Tunable[int]=128, 
                    optimizer: Tunable[optim.Optimizer] = optim.AdamW,
                    pos_weights: Tunable[bool] = True,
                    use_rec_weights: bool = False,
                    epochs: int=300, 
                    early_stopping: bool=True,
                    patience: int=10,
                    swa_model=None,
                    swa_start: int=25,
                    run=None):
        """
        Parameters
        ----------
        modelpath
            path to a folder where to store the params and the best model 
        save
            if the params and the best model should be saved, if save is False the modelpath parameter could only be an empty string
        train_size
            which percentage of samples to use for training
        seed
            seed for the train-val split
        lr
            learning rate
        kl_coeff
            Kullback Leibler loss coefficient
        batch_size
            size of minibatches
        optimizer
            which optimizer to use
        pos_weights
            whether to make weights in decoder positive
        epochs
            over how many epochs to train
        run
            passed here if logging to Neptune should be carried out
        """

        if os.path.isfile(modelpath + '/best_model.pt'):
            print("A model already exists in the specified directory and will be overwritten.")

        if save:
            # save train params
            train_params = {'train_size': train_size,
                            'seed': seed,
                            'lr': lr,
                            'kl_coeff': kl_coeff,
                            'batch_size': batch_size,
                            'optimizer': str(optimizer).split("'")[1],
                            'pos_weights': pos_weights,
                            'use_rec_weights': use_rec_weights,
                            'epochs': epochs,
                            'early_stopping': early_stopping,
                            'patience': patience
                            }
            with open(modelpath + '/train_params.json', 'w') as fp:
                json.dump(train_params, fp, indent=4)
            
            if run is not None:
                run["train_parameters"] = train_params

            # save model params
            with open(modelpath + '/model_params.json', 'w') as fp:
                json.dump(self.params, fp, indent=4)
            
            if run is not None:
                run["model_parameters"] = self.params

        # train-val split
        train_adata, val_adata = split_adata(self.adata, 
                                             train_size = train_size,
                                             seed = seed)

        train_covs = self._cov_tensor(train_adata)
        val_covs = self._cov_tensor(val_adata)

        # generate dataloaders
        trainloader = FastTensorDataLoader(train_adata.X, 
                                           train_covs,
                                         batch_size=batch_size, 
                                         shuffle=True)
        valloader = FastTensorDataLoader(val_adata.X, 
                                         val_covs,
                                        batch_size=batch_size, 
                                        shuffle=False)
        
        # compute reconstruction weights
        if use_rec_weights:
            weights = torch.tensor(np.var(np.array(self.adata.X.todense()), axis=0), dtype=torch.float32)
            self.rec_weights = torch.mul(weights, torch.div(weights[weights != 0].size(dim=0), torch.sum(weights,))).to(self.device)
        else:
            self.rec_weights = None

        val_loss_min = float('inf')
        optimizer = optimizer(self.parameters(), lr = lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=100)

        if swa_model is not None:
            swa_scheduler = SWALR(optimizer, swa_lr = 0.05)

        if early_stopping:
                early_stopper = EarlyStopper(patience=patience)

        for epoch in range(epochs):
            print(f"Epoch {epoch+1} of {epochs}")
            train_epoch_loss = self.train_round(trainloader, kl_coeff, optimizer, pos_weights, run)
            
            if swa_model is not None:
                if epoch >= swa_start:
                    swa_model.update_parameters(self)
                    swa_scheduler.step()
                else:
                    scheduler.step()

            val_epoch_loss = self.val_round(valloader, kl_coeff, run)

            if early_stopping:
                if early_stopper.early_stop(val_epoch_loss):
                    break

            train.report({"validation_loss": val_epoch_loss})

            if run is not None:
                run["metrics/train/loss"].log(train_epoch_loss)
                run["metrics/val/loss"].log(val_epoch_loss)
                
            if val_epoch_loss < val_loss_min and save:
                print('New best model!')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_epoch_loss,
                }, modelpath + '/best_model.pt')
                val_loss_min = val_epoch_loss
                
            print(f"Train Loss: {train_epoch_loss:.4f}")
            print(f"Val Loss: {val_epoch_loss:.4f}")

        # update swa_model
        if swa_model is not None:
            update_bn(trainloader, swa_model, use_cobra=False)

    def _get_activation(self, index, activation={}):
        def hook(model, input, output):
            activation[index] = output
        return hook 
    
    def _attach_hooks(self, lin_layer=True, activation={}, hooks={}):
        """helper function to attach hooks to the decoder"""
        for i in range(len(self.decoder.decoder)-1):
            key = str(i)
            hook_ind=0 if lin_layer else np.where(np.array(self.decoder.decoder[i]) != None)[0][-1]
            value = self.decoder.decoder[i][hook_ind].register_forward_hook(self._get_activation(i, activation))
            hooks[key] = value


    @torch.no_grad()
    def _hook_activities(self, x, cat_list, lin_layer=True):
        """
        Attaches hooks and retrieves pathway activities.

        Parameters
        ----------
        x
            torch.tensor of shape (minibatch, in_features)
        cat_list
            Iterable of torch.tensors containing the category memberships
            shape of each tensor is (minibatch, 1)
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
        z, _, _, _ = self.forward(x, cat_list)

        act = torch.cat(list(activation.values()), dim=1)
        
        # remove hooks
        for h in hooks:
            hooks[h].remove()

        # return pathway activities or reconstructed gene values
        if self.root_layer_latent:
            return torch.hstack((z,act))
        else:
            return act

    def _average_neuronnum(self, act: np.array):
        """
        Helper function to calculate the average value of multiple neurons.
        """
        act = np.array(np.split(act, act.shape[1]/self.neuronnum, axis=1)).mean(axis=2).T
        return act

    @torch.no_grad()
    def _run_batches(self, adata: AnnData, retrieve: Literal['latent', 'act', 'rec'], lin_layer: bool=True):
        """
        Runs batches of a dataloader through encoder or complete VAE and collects results.

        Parameters
        ----------
        latent
            whether to retrieve latent space embedding (True) or reconstructed values (False)
        """
        self.eval()

        if adata is not None:
            if '_ontovae' not in adata.uns.keys():
                raise ValueError('Please run sconto_vae.module.utils.setup_anndata first.')
        else:
            adata = self.adata

        covs = self._cov_tensor(adata)

        dataloader = FastTensorDataLoader(adata.X, 
                                          covs,
                                         batch_size=128, 
                                         shuffle=False)

        res = []
        for minibatch in dataloader:
            x = torch.tensor(minibatch[0].todense(), dtype=torch.float32).to(self.device)
            cat_list = torch.split(minibatch[1].T.to(self.device), 1)
            if retrieve == 'latent':
                result, _, _ = self._get_embedding(x, cat_list)
            elif retrieve == 'act':
                result = self._hook_activities(x, cat_list, lin_layer)
            else:
                _, _, _, result = self.forward(x, cat_list)
            result = result.to('cpu').detach().numpy()
            if retrieve in ['latent', 'act']:
                result = self._average_neuronnum(result)
            res.append(result)
        res = np.vstack(res)

        return res
    
    @torch.no_grad()
    def to_latent(self, adata: AnnData=None):
        """
        Wrapper around _run_batches to retrieve latent space embedding.

        Parameters
        ----------
        adata
            AnnData object that was processed with setup_anndata_vanillavae
        """
        self.eval()
        res = self._run_batches(adata, retrieve='latent')
        return res
    
    @torch.no_grad()
    def get_pathway_activities(self, adata: AnnData=None, lin_layer=True):
        """
        Wrapper around _run_batches to retrieve pathway activities.

        Parameters
        ----------
        adata
            AnnData object that was processed with setup_anndata
        lin_layer
            whether linear layer should be used for calculation
        """
        if len(self.decoder.decoder) == 1:
            raise ValueError('Pathway activities cannot be computed for a one-layer network.')

        self.eval()
        res = self._run_batches(adata, 'act', lin_layer)
        return res

    @torch.no_grad()
    def get_reconstructed_values(self, adata: AnnData=None):
        """
        Wrapper around _run_batches to retrieve output layer.

        Parameters
        ----------
        adata
            AnnData object that was processed with setup_anndata_vanillavae
        """
        self.eval()
        res = self._run_batches(adata, retrieve='rec')
        return res
    
    @torch.no_grad()
    def perturbation(self, adata: AnnData=None, genes: list=[], values: list=[], output=Literal['latent','act','rec'], lin_layer=True):
        """
        Retrieves pathway activities or reconstructed gene values after performing in silico perturbation.

        Parameters
        ----------
        adata
            AnnData object that was processed with setup_anndata
        genes
            a list of genes to perturb
        values
            list with new values, same length as genes
        output
            whether to retrieve latent space ('latent'), pathway activities ('act') or reconstructed gene values ('rec')
        lin_layer
            whether linear layer should be used for pathway activity retrieval
        """
        self.eval()

        if adata is not None:
            if '_ontovae' not in adata.uns.keys():
                raise ValueError('Please run sconto_vae.module.utils.setup_anndata first.')
            pdata = adata.copy()
        else:
            pdata = self.adata.copy()

        # get indices of the genes in list
        gindices = [pdata.uns['_ontovae']['genes'].index(g) for g in genes]

        # replace their values
        for i in range(len(genes)):
            pdata.X[:,gindices[i]] = values[i]

        # run perturbed data through network
        if output == 'latent':
            res = self._run_batches(pdata, 'latent')
        elif output == 'act':
            res = self._run_batches(pdata, 'act', lin_layer)
        else:
            res = self._run_batches(pdata, retrieve='rec')

        return res


    @classproperty
    def _tunables(cls):
        return [cls.__init__, cls.train_model]
    
    @classproperty
    def _metrics(cls):
        ''' Maybe should provide the metric in the manner ["name", "mode"]'''
        return ["validation_loss"]

