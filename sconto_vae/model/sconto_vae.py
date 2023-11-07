#!/usr/bin/env python3

import sys
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from anndata import AnnData
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, LayerField
from scvi.dataloaders import AnnDataLoader

from sconto_vae.module.modules import Encoder, OntoDecoder
from sconto_vae.module.utils import FastTensorDataLoader



"""VAE with ontology in decoder"""

class scOntoVAE(nn.Module):
    """
    This class combines a normal encoder with an ontology structured decoder.
    For now, works with log-normalized data,

    Parameters
    ----------
    adata
        anndata object that has been preprocessed with setup_anndata function
    neuronnum
        number of neurons per term
    drop
        dropout rate, default is 0.2
    z_drop
        dropout rate for latent space, default is 0.5
    """

    def __init__(self, 
                 adata: AnnData, 
                 neuronnum: int = 3, 
                 drop: float = 0.2, 
                 z_drop: float = 0.5):
        super(scOntoVAE, self).__init__()

        self.adata = adata

        if '_ontovae' not in self.adata.uns.keys():
            raise ValueError('Please run sconto_vae.module.utils.setup_anndata_ontovae first.')

        self.thresholds = adata.uns['_ontovae']['thresholds']
        self.in_features = len(self.adata.uns['_ontovae']['genes'])
        self.mask_list = adata.uns['_ontovae']['masks']
        self.mask_list = [torch.tensor(m, dtype=torch.float32) for m in self.mask_list]
        self.layer_dims_dec =  np.array([self.mask_list[0].shape[1]] + [m.shape[0] for m in self.mask_list])
        self.latent_dim = self.layer_dims_dec[0] * neuronnum
        self.layer_dims_enc = [self.latent_dim]
        self.neuronnum = neuronnum
        self.drop = drop
        self.z_drop = z_drop
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Encoder
        self.encoder = Encoder(self.in_features,
                                self.latent_dim,
                                self.layer_dims_enc,
                                self.drop,
                                self.z_drop)

        # Decoder
        self.decoder = OntoDecoder(self.in_features,
                                    self.layer_dims_dec,
                                    self.mask_list,
                                    self.latent_dim,
                                    self.neuronnum)

        self.X = adata.X

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
        
    def _get_embedding(self, x):
        """
        Generates latent space embedding.

        Parameters
        ----------
        x
            dataset of which embedding should be generated
        """
        mu, log_var = self.encoder(x)
        embedding = self.reparameterize(mu, log_var)
        return embedding

    def forward(self, x):
        # encoding
        mu, log_var = self.encoder(x)
            
        # sample from latent space
        z = self.reparameterize(mu, log_var)
        
        # decoding
        reconstruction = self.decoder(z)
            
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

    def train_round(self, dataloader, kl_coeff, optimizer, run=None):
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
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):

            # move batch to device
            data = torch.tensor(data[0].todense(), dtype=torch.float32).to(self.device)
            optimizer.zero_grad()

            # forward step
            reconstruction, mu, log_var = self.forward(data)
            loss = self.vae_loss(reconstruction, mu, log_var, data, kl_coeff, mode='train', run=run)
            running_loss += loss.item()

            # backward propagation
            loss.backward()

            # zero out gradients from non-existent connections
            for i in range(len(self.decoder.decoder)):
                self.decoder.decoder[i][0].weight.grad = torch.mul(self.decoder.decoder[i][0].weight.grad, self.decoder.masks[i])

            # perform optimizer step
            optimizer.step()

            # make weights in Onto module positive
            for i in range(len(self.decoder.decoder)):
                self.decoder.decoder[i][0].weight.data = self.decoder.decoder[i][0].weight.data.clamp(0)

        # compute avg training loss
        train_loss = running_loss/len(dataloader)
        return train_loss

    @torch.no_grad()
    def val_round(self, dataloader, kl_coeff, run=None):
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
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):

            # move batch to device
            data = torch.tensor(data[0].todense(), dtype=torch.float32).to(self.device)

            # forward step
            reconstruction, mu, log_var = self.forward(data)
            loss = self.vae_loss(reconstruction, mu, log_var,data, kl_coeff, mode='val', run=run)
            running_loss += loss.item()

        # compute avg val loss
        val_loss = running_loss/len(dataloader)
        return val_loss

    def train_model(self, 
                    modelpath: str, 
                    lr: float=1e-4, 
                    kl_coeff: float=1e-4, 
                    batch_size: int=128, 
                    epochs: int=300, 
                    run=None):
        """
        Parameters
        ----------
        modelpath
            where to store the best model (full path with filename)
        lr
            learning rate
        kl_coeff
            Kullback Leibler loss coefficient
        batch_size
            size of minibatches
        epochs
            over how many epochs to train
        run
            passed here if logging to Neptune should be carried out
        """
        # train-test split
        indices = np.random.RandomState(seed=42).permutation(self.X.shape[0])
        X_train_ind = indices[:round(len(indices)*0.8)]
        X_val_ind = indices[round(len(indices)*0.8):]
        X_train, X_val = self.X[X_train_ind,:], self.X[X_val_ind,:]

        # generate dataloaders
        trainloader = FastTensorDataLoader(X_train, 
                                       batch_size=batch_size, 
                                       shuffle=True)
        valloader = FastTensorDataLoader(X_val, 
                                        batch_size=batch_size, 
                                        shuffle=False)

        val_loss_min = float('inf')
        optimizer = optim.AdamW(self.parameters(), lr = lr)

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
                }, modelpath)
                val_loss_min = val_epoch_loss
                
            print(f"Train Loss: {train_epoch_loss:.4f}")
            print(f"Val Loss: {val_epoch_loss:.4f}")


    def load(self, modelpath: str):
        """
        Helper function to load a model
        
        Parameters
        ----------
        Path where a trained model was saved.
        """
        checkpoint = torch.load(modelpath + '/best_model.pt',
                            map_location = torch.device(self.device))
        self.load_state_dict(checkpoint['model_state_dict'], strict=False)        


    def _get_activation(self, index, activation={}):
        def hook(model, input, output):
            activation[index] = output
        return hook 
    
    def _attach_hooks(self, activation={}, hooks={}):
        """helper function to attach hooks to the decoder"""
        for i in range(len(self.decoder.decoder)-1):
            key = str(i)
            value = self.decoder.decoder[i][0].register_forward_hook(self._get_activation(i, activation))
            hooks[key] = value


    @torch.no_grad()
    def _pass_data(self, data, output):
        """
        Passes data through the model.

        Parameters
        ----------
        data
            data to be passed
        output
            'act': return pathway activities
            'rec': return reconstructed values
        """

        # set to eval mode
        self.eval()

        # move data to device
        data = data.to(self.device)

        # get latent space embedding
        z = self._get_embedding(data)
        z = z.to('cpu').detach().numpy()
        
        z = np.array(np.split(z, z.shape[1]/self.neuronnum, axis=1)).mean(axis=2).T

        # initialize activations and hooks
        activation = {}
        hooks = {}

        # attach the hooks
        self._attach_hooks(activation=activation, hooks=hooks)
        
        # pass data through model
        reconstruction, _, _ = self.forward(data)

        act = torch.cat(list(activation.values()), dim=1).to('cpu').detach().numpy()
        act = np.array(np.split(act, act.shape[1]/self.neuronnum, axis=1)).mean(axis=2).T
        
        # remove hooks
        for h in hooks:
            hooks[h].remove()

        # return pathway activities or reconstructed gene values
        if output == 'act':
            return np.hstack((z,act))
        if output == 'rec':
            return reconstruction.to('cpu').detach().numpy()


    @torch.no_grad()
    def get_pathway_activities(self, adata: AnnData=None, indices=None, terms=None):
        """
        Retrieves pathway activities from latent space and decoder.

        Parameters
        ----------
        adata
            AnnData object that was processed with setup_anndata
        indices
            if activities are to be retrieved for a subset of samples
        terms
            list of ontology term ids whose activities should be retrieved
        """
        if adata is not None:
            if '_ontovae' not in adata.uns.keys():
                raise ValueError('Please run sconto_vae.module.utils.setup_anndata first.')
        else:
            adata = self.adata

        anndata_fields = [
            LayerField(registry_key="x", layer=None, is_count_data=False)
        ]
        adata_manager = AnnDataManager(fields=anndata_fields)
        adata_manager.register_fields(adata)
        dl = AnnDataLoader(adata_manager, indices=indices, batch_size=128)

        act = []
        for tensors in dl:
            act.append(self._pass_data(tensors['x'], 'act'))
        act = np.vstack(act)

        # if term was specified, subset
        if terms is not None:
            annot = adata.uns['_ontovae']['annot']
            term_ind = annot[annot.ID.isin(terms)].index.to_numpy()

            act = act[:,term_ind]

        return act


    @torch.no_grad()
    def get_reconstructed_values(self, adata: AnnData=None, indices=None, rec_genes=None):
        """
        Retrieves reconstructed values from output layer.

        Parameters
        ----------
        adata
            AnnData object that was processed with setup_anndata
        indices
            if activities are to be retrieved for a subset of samples
        rec_genes
            list of genes whose reconstructed values should be retrieved
        """

        if adata is not None:
            if '_ontovae' not in adata.uns.keys():
                raise ValueError('Please run sconto_vae.module.utils.setup_anndata first.')
        else:
            adata = self.adata

        anndata_fields = [
            LayerField(registry_key="x", layer=None, is_count_data=False)
        ]
        adata_manager = AnnDataManager(fields=anndata_fields)
        adata_manager.register_fields(adata)
        dl = AnnDataLoader(adata_manager, indices=indices, batch_size=128)

        rec = []
        for tensors in dl:
            rec.append(self._pass_data(tensors['x'], 'rec'))
        rec = np.vstack(rec)

        # if genes were passed, subset
        if rec_genes is not None:
            onto_genes = adata.uns['_ontovae']['genes']
            gene_ind = np.array([onto_genes.index(g) for g in rec_genes])

            rec = rec[:,gene_ind]

        return rec

    
    @torch.no_grad()
    def perturbation(self, adata: AnnData=None, indices=None, genes: list=[], values: list=[], output='terms', terms=None, rec_genes=None):
        """
        Retrieves pathway activities or reconstructed gene values after performing in silico perturbation.

        Parameters
        ----------
        adata
            AnnData object that was processed with setup_anndata
        indices
            if activities are to be retrieved for a subset of samples
        genes
            a list of genes to perturb
        values
            list with new values, same length as genes
        output
            - 'terms': retrieve pathway activities
            - 'genes': retrieve reconstructed values

        terms
            list of ontology term ids whose values should be retrieved
        rec_genes
            list of genes whose values should be retrieved
        """

        if adata is not None:
            if '_ontovae' not in adata.uns.keys():
                raise ValueError('Please run sconto_vae.module.utils.setup_anndata first.')
        else:
            adata = self.adata

        # get indices of the genes in list
        gindices = [self.adata.uns['_ontovae']['genes'].index(g) for g in genes]

        # replace their values
        for i in range(len(genes)):
            adata.X[:,gindices[i]] = values[i]

        anndata_fields = [
            LayerField(registry_key="x", layer=None, is_count_data=False)
        ]
        adata_manager = AnnDataManager(fields=anndata_fields)
        adata_manager.register_fields(adata)
        dl = AnnDataLoader(adata_manager, indices=indices, batch_size=128)

        # get pathway activities or reconstructed values after perturbation
        res = []
        for tensors in dl:
            if output == 'terms':
                res.append(self._pass_data(tensors['x'], 'act'))
            if output == 'genes':
                res.append(self._pass_data(tensors['x'], 'rec'))

        # if term was specified, subset
        if terms is not None:
            annot = adata.uns['_ontovae']['annot']
            term_ind = annot[annot.ID.isin(terms)].index.to_numpy()

            res = res[:,term_ind]
        
        if rec_genes is not None:
            onto_genes = adata.uns['_ontovae']['genes']
            gene_ind = np.array([onto_genes.index(g) for g in rec_genes])

            res = res[:,gene_ind]

        return res
        


    




