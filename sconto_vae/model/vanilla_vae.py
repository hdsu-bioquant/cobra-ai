#!/usr/bin/env python3
import json

import numpy as np
import pandas as pd
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from typing import Iterable

from anndata import AnnData

from sconto_vae.module.modules import Encoder, Decoder
from sconto_vae.module.utils import split_adata, FastTensorDataLoader



"""vanilla VAE"""

class vanillaVAE(nn.Module):
    """
    vanilla VAE that works with anndata Object

    Parameters
    ----------
    adata
        anndata object that has been preprocessed with setup_anndata function
    latent_dim
        latent dimension
    hidden_layers_enc
        number of hidden layers in encoder
    neurons_per_layer_enc
        neurons per hidden layer in encoder
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
    inject_covariates_enc
        Whether to inject covariates in each layer (True), or just the first (False) of encoder
    drop_enc
        dropout rate in encoder
    z_drop
        dropout rate for latent space 
    hidden_layers_dec
        number of hidden layers in decoder
    neurons_per_layer_dec
        neurons per hidden layer in decoder
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
        model = cls(adata, **params) 
        checkpoint = torch.load(modelpath + '/best_model.pt',
                            map_location = torch.device(model.device))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        return model

    def __init__(self, 
                 adata: AnnData, 
                 latent_dim: int = 128,
                 hidden_layers_enc: int = 2,
                 neurons_per_layer_enc: int = 256,
                 use_batch_norm_enc: bool = True,
                 use_layer_norm_enc: bool = False,
                 use_activation_enc: bool = True,
                 activation_fn_enc: nn.Module = nn.ReLU,
                 bias_enc: bool = True,
                 inject_covariates_enc: bool = False,
                 drop_enc: float = 0.2, 
                 z_drop: float = 0.5,
                 hidden_layers_dec: int = 2,
                 neurons_per_layer_dec: int = 256,
                 use_batch_norm_dec: bool = True,
                 use_layer_norm_dec: bool = False,
                 use_activation_dec: bool = True,
                 use_activation_lat: bool = False,
                 activation_fn_dec: nn.Module = nn.ReLU,
                 bias_dec: bool = True,
                 inject_covariates_dec: bool = False,
                 drop_dec: float = 0):
        super().__init__()

        # store init params in dict
        self.params = {'latent_dim': latent_dim,
                       'hidden_layers_enc': hidden_layers_enc,
                       'neurons_per_layer_enc': neurons_per_layer_enc,
                          'use_batch_norm_enc': use_batch_norm_enc,
                          'use_layer_norm_enc': use_layer_norm_enc,
                          'use_activation_enc': use_activation_enc,
                          'activation_fn_enc': str(activation_fn_enc).split("'")[1] if activation_fn_enc is not None else activation_fn_enc,
                          'bias_enc': bias_enc,
                          'inject_covariates_enc': inject_covariates_enc,
                          'drop_enc': drop_enc,
                          'z_drop': z_drop,
                          'hidden_layers_dec': hidden_layers_dec,
                          'neurons_per_layer_dec': neurons_per_layer_dec,
                          'use_batch_norm_dec': use_batch_norm_dec,
                          'use_layer_norm_dec': use_layer_norm_dec,
                          'use_activation_dec': use_activation_dec,
                          'use_activation_lat': use_activation_lat,
                          'activation_fn_dec': str(activation_fn_dec).split("'")[1] if activation_fn_dec is not None else activation_fn_dec,
                          'bias_dec': bias_dec,
                          'inject_covariates_dec': inject_covariates_dec,
                          'drop_dec': drop_dec}


        self.adata = adata

        # parse information
        self.in_features = self.adata.shape[1]
        self.latent_dim = latent_dim
        self.use_activation_lat = use_activation_lat
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # parse SCVI information
        self.batch = adata.obs['_scvi_batch']
        self.labels = adata.obs['_scvi_labels']
        self.covs = adata.obsm['_scvi_extra_categorical_covs'] if '_scvi_extra_categorical_covs' in adata.obsm.keys() else None

        self.n_cat_list = [len(self.batch.unique()), len(self.labels.unique())]
        if self.covs is not None:
            self.n_cat_list.extend([len(self.covs[c].unique()) for c in self.covs.columns])

        # Encoder
        self.encoder = Encoder(in_features = self.in_features,
                                latent_dim = self.latent_dim,
                                n_cat_list = self.n_cat_list,
                                hidden_layers = hidden_layers_enc,
                                neurons_per_layer = neurons_per_layer_enc, 
                                use_batch_norm = use_batch_norm_enc,
                                use_layer_norm = use_layer_norm_enc,
                                use_activation = use_activation_enc,
                                activation_fn = activation_fn_enc,
                                bias = bias_enc,
                                inject_covariates = inject_covariates_enc,
                                drop = drop_enc,
                                z_drop = z_drop)

        # Decoder
        self.decoder = Decoder(in_features = self.in_features,
                                latent_dim = self.latent_dim,
                                n_cat_list = self.n_cat_list,
                                hidden_layers = hidden_layers_dec,
                                neurons_per_layer = neurons_per_layer_dec,
                                use_batch_norm = use_batch_norm_dec,
                                use_layer_norm = use_layer_norm_dec,
                                use_activation = use_activation_dec,
                                activation_fn = activation_fn_dec,
                                bias = bias_dec,
                                inject_covariates = inject_covariates_dec,
                                drop = drop_dec)

        self.to(self.device)

    def _cov_tensor(self, adata):
        """
        Helper function to aggregate information from adata to use as input for dataloader.
        """
        covs = adata.obs[['_scvi_batch', '_scvi_labels']]
        if '_scvi_extra_categorical_covs' in adata.obsm.keys():
            covs = pd.concat([covs, adata.obsm['_scvi_extra_categorical_covs']], axis=1)
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
        """
        sigma = torch.exp(0.5*log_var) 
        eps = torch.randn_like(sigma) 
        return mu + eps * sigma
        
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
        embedding = self.reparameterize(mu, log_var)
        if self.use_activation_lat:
            embedding = self.activation_fn_dec()(embedding)
        return embedding


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

        mu, log_var = self.encoder(x, cat_list)
        z = self.reparameterize(mu, log_var)
        if self.use_activation_lat:
            z = self.activation_fn_dec()(z)
        reconstruction = self.decoder(z, cat_list)
        return reconstruction, mu, log_var

    def vae_loss(self, reconstruction, mu, log_var, data, kl_coeff, mode='train', run=None):
        """
        Calculates VAE loss as combination of reconstruction loss and weighted Kullback-Leibler loss.
        """
        kl_loss = -0.5 * torch.sum(1. + log_var - mu.pow(2) - log_var.exp(), )
        rec_loss = F.mse_loss(reconstruction, data, reduction="sum")
        if run is not None:
            run["metrics/" + mode + "/kl_loss"].log(kl_loss)
            run["metrics/" + mode + "/rec_loss"].log(rec_loss)
        return torch.mean(rec_loss + kl_coeff*kl_loss)

    def train_round(self, 
                    dataloader: FastTensorDataLoader, 
                    kl_coeff: float, 
                    optimizer: optim.Optimizer, 
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
            reconstruction, mu, log_var = self.forward(data, cat_list)
            loss = self.vae_loss(reconstruction, mu, log_var, data, kl_coeff, mode='train', run=run)
            running_loss += loss.item()

            # backward propagation
            loss.backward()

            # perform optimizer step
            optimizer.step()

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
            reconstruction, mu, log_var = self.forward(data, cat_list)
            loss = self.vae_loss(reconstruction, mu, log_var,data, kl_coeff, mode='val', run=run)
            running_loss += loss.item()

        # compute avg val loss
        val_loss = running_loss/len(dataloader)
        return val_loss

    def train_model(self, 
                    modelpath: str, 
                    train_size: float = 0.9,
                    seed: int = 42,
                    lr: float=1e-4, 
                    kl_coeff: float=1e-4, 
                    batch_size: int=128, 
                    optimizer: optim.Optimizer = optim.AdamW,
                    epochs: int=300, 
                    run=None):
        """
        Parameters
        ----------
        modelpath
            path to a folder where to store the params and the best model 
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
        epochs
            over how many epochs to train
        run
            passed here if logging to Neptune should be carried out
        """
        # save train params
        train_params = {'train_size': train_size,
                        'seed': seed,
                        'lr': lr,
                        'kl_coeff': kl_coeff,
                        'batch_size': batch_size,
                        'optimizer': str(optimizer).split("'")[1],
                        'epochs': epochs
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

        val_loss_min = float('inf')
        optimizer = optimizer(self.parameters(), lr = lr)



        for epoch in range(epochs):
            print(f"Epoch {epoch+1} of {epochs}")
            train_epoch_loss = self.train_round(trainloader, kl_coeff, optimizer, run)
            val_epoch_loss = self.val_round(valloader, kl_coeff, run)
            
            if run is not None:
                run["metrics/train/loss"].log(train_epoch_loss)
                run["metrics/val/loss"].log(val_epoch_loss)
                
            if val_epoch_loss < val_loss_min:
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


    @torch.no_grad()
    def to_latent(self, adata: AnnData=None):
        """
        Retrieves reconstructed values from output layer.

        Parameters
        ----------
        adata
            AnnData object that was processed with setup_anndata_vanillavae
        """
        self.eval()

        if adata is None:
            adata = self.adata

        covs = self._cov_tensor(adata)

        # generate dataloaders
        dataloader = FastTensorDataLoader(adata.X, 
                                          covs,
                                         batch_size=128, 
                                         shuffle=False)

        embed = []
        for minibatch in dataloader:
            x = torch.tensor(minibatch[0].todense(), dtype=torch.float32).to(self.device)
            cat_list = torch.split(minibatch[1].T.to(self.device), 1)
            embedding = self._get_embedding(x, cat_list)
            embed.append(embedding.to('cpu').detach().numpy())
        embed = np.vstack(embed)

        return embed


    @torch.no_grad()
    def get_reconstructed_values(self, adata: AnnData=None):
        """
        Retrieves reconstructed values from output layer.

        Parameters
        ----------
        adata
            AnnData object that was processed with setup_anndata_vanillavae
        """
        self.eval()

        if adata is None:
            adata = self.adata

        covs = self._cov_tensor(adata)

        # generate dataloaders
        dataloader = FastTensorDataLoader(adata.X, 
                                          covs,
                                         batch_size=128, 
                                         shuffle=False)

        rec = []
        for minibatch in dataloader:
            x = torch.tensor(minibatch[0].todense(), dtype=torch.float32).to(self.device)
            cat_list = torch.split(minibatch[1].T.to(self.device), 1)
            reconstruction, _, _ = self.forward(x, cat_list)
            rec.append(reconstruction.to('cpu').detach().numpy())
        rec = np.vstack(rec)

        return rec

    
    @torch.no_grad()
    def perturbation(self, adata: AnnData=None, genes: list=[], values: list=[]):
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
        """
        self.eval()

        if adata is None:
            adata = self.adata

        # get indices of the genes in list
        gindices = [list(adata.var_names).index(g) for g in genes]

        # replace their values
        for i in range(len(genes)):
            adata.X[:,gindices[i]] = values[i]

        covs = self._cov_tensor(adata)

        # generate dataloader
        dataloader = FastTensorDataLoader(adata.X, 
                                          covs,
                                         batch_size=128, 
                                         shuffle=False)

        rec = []
        for minibatch in dataloader:
            x = torch.tensor(minibatch[0].todense(), dtype=torch.float32).to(self.device)
            cat_list = torch.split(minibatch[1].T.to(self.device), 1)
            reconstruction, _, _ = self.forward(x, cat_list)
            rec.append(reconstruction.to('cpu').detach().numpy())
        rec = np.vstack(rec)

        return rec
        


    





