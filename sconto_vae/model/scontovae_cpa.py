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

from sconto_vae.model.sconto_vae import scOntoVAE
from sconto_vae.module.modules import Classifier
from sconto_vae.module.utils import split_adata, FastTensorDataLoader



"""VAE with ontology in decoder"""

class OntoVAEcpa(scOntoVAE):
    """
    This class extends scOntoVAE with a CPA-like approach of disentangling covariate effects in the latent space in a linear fashion.

    Parameters
    ----------
    adata
        anndata object that has been preprocessed with setup_anndata function
    layer_dims_class
        list containing the hidden layer dimensions of classifier
    use_batch_norm_class
        Whether to have `BatchNorm` layers or not in encoder
    use_layer_norm_class
        Whether to have `LayerNorm` layers or not in encoder
    use_activation_class
        Whether to have layer activation or not in encoder
    activation_fn_class
        Which activation function to use in encoder
    bias_class
        Whether to learn bias in linear layers or not in encoder
    inject_covariates_class
        Whether to inject covariates in each layer (True), or just the first (False) of encoder
    drop_class
        dropout rate in encoder
    average_neurons
        whether to average by neuronnum before passing terms to classifier
    
    Inherited Parameters (should be passed as dictionary to **kwargs)
    --------------------
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
    neuronnum
        number of neurons per term in decoder
    use_batch_norm_dec
        Whether to have `BatchNorm` layers or not in decoder
    use_layer_norm_dec
        Whether to have `LayerNorm` layers or not in decoder
    use_activation_dec
        Whether to have layer activation or not in decoder
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
        if params['activation_fn_class'] is not None:
            params['activation_fn_class'] = eval(params['activation_fn_class'])
        model = cls(adata, **params) 
        checkpoint = torch.load(modelpath + '/best_model.pt',
                            map_location = torch.device(model.device))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        return model
    
    def __init__(self, 
                 adata: AnnData, 
                 layer_dims_class: list = [64],
                 use_batch_norm_class: bool = True,
                 use_layer_norm_class: bool = False,
                 use_activation_class: bool = True,
                 activation_fn_class: nn.Module = nn.ReLU,
                 bias_class: bool = True,
                 inject_covariates_class: bool = False,
                 drop_class: float = 0.2,
                 average_neurons: bool = False,
                 **kwargs):
        super().__init__(adata, **kwargs)

        class_params = {'layer_dims_class': layer_dims_class,
                        'use_batch_norm_class': use_batch_norm_class,
                        'use_layer_norm_class': use_layer_norm_class,
                        'use_activation_class': use_activation_class,
                        'activation_fn_class': str(activation_fn_class).split("'")[1] if activation_fn_class is not None else activation_fn_class,
                        'bias_class': bias_class,
                        'inject_covariates_class': inject_covariates_class,
                        'drop_class': drop_class,
                        'average_neurons': average_neurons}
        self.params.update(class_params)

        
        # set up covariates
        self.cpa_covs = adata.obsm['_cpa_categorical_covs'] if '_cpa_categorical_covs' in adata.obsm.keys() else None
        if self.cpa_covs is None:
            raise ValueError('Please specify cpa_keys in setup_anndata_ontovae to run the model.')

        self.cov_dict = {}
        for cov in self.cpa_covs.columns:
            self.cov_dict[cov] = dict(zip(adata.obs.loc[:,cov].tolist(), self.cpa_covs.loc[:,cov].tolist()))

        # embedding of covars
        self.covars_embeddings = nn.ModuleDict(
            {
                key: torch.nn.Embedding(len(self.cov_dict[key]), self.latent_dim)
                for key in self.cov_dict.keys()
            }
        )
        
        # covars classifiers
        self.covars_classifiers = nn.ModuleDict(
            {
                key: Classifier(in_features = self.latent_dim,
                                n_classes = len(self.cov_dict[key]),
                                n_cat_list = self.n_cat_list,
                                layer_dims = layer_dims_class,
                                use_batch_norm = use_batch_norm_class,
                                use_layer_norm = use_layer_norm_class,
                                use_activation = use_activation_class,
                                activation_fn = activation_fn_class,
                                bias = bias_class,
                                inject_covariates = inject_covariates_class,
                                drop = drop_class)
                for key in self.cov_dict.keys()
            }
        )

        self.to(self.device)

    def _get_embedding(self, x: torch.tensor, cat_list: Iterable[torch.tensor], cov_list: Iterable[torch.tensor]):
        """
        Generates latent space embedding.

        Parameters
        ----------
        x
            torch.tensor of shape (minibatch, in_features)
        cat_list
            Iterable of torch.tensors containing the category memberships
            shape of each tensor is (minibatch, 1)
        cov_list
            Iterable of torch.tensors containing the covs to delineate in 
            latent space 
        """
        # encoding
        mu, log_var = self.encoder(x, cat_list)
            
        # sample from latent space
        z_basal = self.reparameterize(mu, log_var)
        if self.use_activation_lat and self.use_activation_dec:
            z_basal = self.activation_fn_dec()(z_basal)

        # covariate encoding
        covars_embeddings = {}
        for i, key in enumerate(self.covars_embeddings.keys()):
            covars_embeddings[key] = self.covars_embeddings[key](cov_list[i].long().squeeze())

        # create different z's
        z_cov = {}
        z_total = z_basal.clone()
        for key in covars_embeddings.keys():
            z_cov['z_' + key] = (z_basal + covars_embeddings[key]).to('cpu').detach().numpy()
            z_total += covars_embeddings[key]

        z_dict = dict(z_basal=z_basal.to('cpu').detach().numpy())
        z_dict.update(z_cov, z_total=z_total.to('cpu').detach().numpy())

        return z_dict
  
    def forward(self, x: torch.tensor, cat_list: Iterable[torch.tensor], cov_list: Iterable[torch.tensor], mixup_lambda: float):
        """
        Forward computation on minibatch of samples.
        
        Parameters
        ----------
        x
            torch.tensor of shape (minibatch, in_features)
        cat_list
            Iterable of torch.tensors containing the category memberships
            shape of each tensor is (minibatch, 1)
        cov_list
            Iterable of torch.tensors containing the covs to delineate in 
            latent space 
        mixup_lambda
            coefficient for adversarial training
        """
        # data mixup for adversarial training
        batch_size = x.size()[0]
        if mixup_lambda > 0:
            index = torch.randperm(batch_size).to(x.device)
            x = mixup_lambda * x + (1. - mixup_lambda) * x[index, :]
        else:
            index = torch.arange(0,batch_size).to(x.device)

        # encoding
        mu, log_var = self.encoder(x, cat_list)
            
        # sample from latent space
        z_basal = self.reparameterize(mu, log_var)
        if self.use_activation_lat and self.use_activation_dec:
            z_basal = self.activation_fn_dec()(z_basal)

        # covariate classifiers on z_basal
        covars_pred = {}
        for key in self.covars_classifiers.keys():
            covar_pred = self.covars_classifiers[key](z_basal, cat_list)
            covars_pred[key] = covar_pred

        # covariate encoding
        covars_embeddings = {}
        for i, key in enumerate(self.covars_embeddings.keys()):
            cov_embed = self.covars_embeddings[key](cov_list[i].long().squeeze())
            cov_mix_embed = self.covars_embeddings[key](cov_list[i].long().squeeze()[index])
            covars_embeddings[key] = mixup_lambda * cov_embed + (1. - mixup_lambda) * cov_mix_embed
  
        # add covariates to z_basal
        z_total = z_basal.clone()
        for key in covars_embeddings.keys():
            z_total += covars_embeddings[key]

        # decoding
        reconstruction = self.decoder(z_total, cat_list)
            
        return reconstruction, mu, log_var, covars_pred

    def clf_loss(self, class_output, y, cov: str, mode='train', run=None):
        """
        Calculates loss of a covariate classifier
        """
        class_loss = nn.CrossEntropyLoss()
        clf_loss = class_loss(class_output, y)
        if run is not None:
            run["metrics/" + mode + "/" + cov + "_clf_loss"].log(clf_loss)
        return clf_loss

    def train_round(self, 
                    dataloader: FastTensorDataLoader, 
                    kl_coeff: float, 
                    adv_coeff: float,
                    mixup_lambda: float,
                    optimizer: optim.Optimizer, 
                    run=None):
        """
        Parameters
        ----------
        dataloader
            pytorch dataloader instance with training data
        kl_coeff 
            coefficient for weighting Kullback-Leibler loss
        adv_coeff 
            coefficient for weighting classifier
        mixup_lambda
            coefficient for adversarial training
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
            cov_list = torch.split(minibatch[2].T.to(self.device), 1)
            optimizer.zero_grad()

            # forward step
            reconstruction, mu, log_var, covars_pred = self.forward(data, cat_list, cov_list, mixup_lambda)
            vae_loss = self.vae_loss(reconstruction, mu, log_var, data, kl_coeff, mode='train', run=run)
            clf_loss = 0.0
            for i, vals in enumerate(cov_list):
                cov = list(self.cov_dict.keys())[i]
                cov_loss = self.clf_loss(covars_pred[cov], vals.long().squeeze(), cov=cov, mode='train', run=run)
                clf_loss += cov_loss
            loss = vae_loss - adv_coeff * clf_loss
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
    def val_round(self, 
                  dataloader: FastTensorDataLoader, 
                  kl_coeff: float, 
                  adv_coeff: float,
                  mixup_lambda: float,
                  run=None):
        """
        Parameters
        ----------
        dataloader
            pytorch dataloader instance with training data
        kl_coeff
            coefficient for weighting Kullback-Leibler loss
        adv_coeff 
            coefficient for weighting classifier
        mixup_lambda
            coefficient for adversarial training
        run
            Neptune run if training is to be logged
        """
        # set to eval mode
        self.eval()
        #mixup_lambda=0

        # initialize running loss
        running_loss = 0.0

        # iterate over dataloader for validation
        for i, minibatch in tqdm(enumerate(dataloader), total=len(dataloader)):

            # move minibatch to device
            data = torch.tensor(minibatch[0].todense(), dtype=torch.float32).to(self.device)
            cat_list = torch.split(minibatch[1].T.to(self.device), 1)
            cov_list = torch.split(minibatch[2].T.to(self.device), 1)

            # forward step
            reconstruction, mu, log_var, covars_pred = self.forward(data, cat_list, cov_list, mixup_lambda)
            vae_loss = self.vae_loss(reconstruction, mu, log_var, data, kl_coeff, mode='val', run=run)
            clf_loss = 0.0
            for i, vals in enumerate(cov_list):
                cov = list(self.cov_dict.keys())[i]
                cov_loss = self.clf_loss(covars_pred[cov], vals.long().squeeze(), cov=cov, mode='val', run=run)
                clf_loss += cov_loss
            loss = vae_loss - adv_coeff * clf_loss
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
                    adv_coeff: float=1e3,
                    mixup_lambda: float=0.2,
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
        adv_coeff 
            coefficient for weighting classifier
        mixup_lambda
            coefficient for adversarial training
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
                        'adv_coeff': adv_coeff,
                        'mixup_lambda': mixup_lambda,
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

        train_batch = self._cov_tensor(train_adata)
        val_batch = self._cov_tensor(val_adata)

        train_covs = torch.tensor(train_adata.obsm['_cpa_categorical_covs'].to_numpy())
        val_covs = torch.tensor(val_adata.obsm['_cpa_categorical_covs'].to_numpy())

        # generate dataloaders
        trainloader = FastTensorDataLoader(train_adata.X, 
                                           train_batch,
                                           train_covs,
                                         batch_size=batch_size, 
                                         shuffle=True)
        valloader = FastTensorDataLoader(val_adata.X, 
                                         val_batch,
                                         val_covs,
                                        batch_size=batch_size, 
                                        shuffle=False)

        val_loss_min = float('inf')
        optimizer = optimizer(self.parameters(), lr = lr)

        for epoch in range(epochs):
            print(f"Epoch {epoch+1} of {epochs}")
            train_epoch_loss = self.train_round(trainloader, kl_coeff, adv_coeff, mixup_lambda, optimizer, run)
            val_epoch_loss = self.val_round(valloader, kl_coeff, adv_coeff, mixup_lambda, run)
            
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
    def _pass_data(self, x, cat_list, cov_list, output, lin_layer=True):
        """
        Passes data through the model.

        Parameters
        ----------
        x
            torch.tensor of shape (minibatch, in_features)
        cat_list
            Iterable of torch.tensors containing the category memberships
            shape of each tensor is (minibatch, 1)
        cov_list
            Iterable of torch.tensors containing the covs to delineate in 
            latent space 
        output
            'act': return pathway activities
            'rec': return reconstructed values
        lin_layer:
            whether hooks should be attached to linear layer of the model
        """

        # set to eval mode
        self.eval()

        # get latent space embedding
        z = self._get_embedding(x, cat_list)
        z = z.to('cpu').detach().numpy()

        # initialize activations and hooks
        activation = {}
        hooks = {}

        # attach the hooks
        self._attach_hooks(lin_layer=lin_layer, activation=activation, hooks=hooks)

        # pass data through model
        reconstruction, _, _, _ = self.forward(x, cat_list, cov_list, mixup_lambda=0)

        act = torch.cat(list(activation.values()), dim=1).to('cpu').detach().numpy()
        
        # remove hooks
        for h in hooks:
            hooks[h].remove()

        # return pathway activities or reconstructed gene values
        if output == 'act':
            return np.hstack((z,act))
        if output == 'rec':
            return reconstruction.to('cpu').detach().numpy()

    @torch.no_grad()
    def to_latent(self, adata: AnnData=None):
        """
        Retrieves different representations of z.

        Parameters
        ----------
        adata
            AnnData object that was processed with setup_anndata
        """
        if adata is not None:
            if '_ontovae' not in adata.uns.keys():
                raise ValueError('Please run sconto_vae.module.utils.setup_anndata first.')
        else:
            adata = self.adata

        batch = self._cov_tensor(adata)
        covs = torch.tensor(adata.obsm['_cpa_categorical_covs'].to_numpy())

        # generate dataloaders
        dataloader = FastTensorDataLoader(adata.X, 
                                          batch,
                                          covs,
                                         batch_size=128, 
                                         shuffle=False)

        embed = []
        for minibatch in dataloader:
            x = torch.tensor(minibatch[0].todense(), dtype=torch.float32).to(self.device)
            cat_list = torch.split(minibatch[1].T.to(self.device), 1)
            cov_list = torch.split(minibatch[2].T.to(self.device), 1)
            embed_dict = self._get_embedding(x, cat_list, cov_list)
            embed_dict_avg = {k: self._average_neuronnum(v) for k, v in embed_dict.items()}
            embed.append(embed_dict_avg)
        key_list = list(embed[0].keys())
        embed_dict = {}
        for key in key_list:
            embed_key = np.vstack([e[key] for e in embed])
            embed_dict[key] = embed_key

        return embed_dict
    
    @torch.no_grad()
    def get_pathway_activities(self, adata: AnnData=None, terms=None, lin_layer=True):
        """
        Retrieves pathway activities from latent space and decoder.

        Parameters
        ----------
        adata
            AnnData object that was processed with setup_anndata
        terms
            list of ontology term ids whose activities should be retrieved
        lin_layer
            whether linear layer should be used for calculation
        """
        if adata is not None:
            if '_ontovae' not in adata.uns.keys():
                raise ValueError('Please run sconto_vae.module.utils.setup_anndata first.')
        else:
            adata = self.adata

        batch = self._cov_tensor(adata)
        covs = torch.tensor(adata.obsm['_cpa_categorical_covs'].to_numpy())

        # generate dataloaders
        dataloader = FastTensorDataLoader(adata.X, 
                                          batch,
                                          covs,
                                         batch_size=128, 
                                         shuffle=False)

        act = []
        for minibatch in dataloader:
            x = torch.tensor(minibatch[0].todense(), dtype=torch.float32).to(self.device)
            cat_list = torch.split(minibatch[1].T.to(self.device), 1)
            cov_list = torch.split(minibatch[2].T.to(self.device), 1)

            a = self._pass_data(x, cat_list, 'act', lin_layer)
            aavg = self._average_neuronnum(a)
            act.append(aavg)
        act = np.vstack(act)

        # if term was specified, subset
        if terms is not None:
            annot = adata.uns['_ontovae']['annot']
            term_ind = annot[annot.ID.isin(terms)].index.to_numpy()

            act = act[:,term_ind]

        return act


    @torch.no_grad()
    def get_reconstructed_values(self, adata: AnnData=None, rec_genes=None):
        """
        Retrieves reconstructed values from output layer.

        Parameters
        ----------
        adata
            AnnData object that was processed with setup_anndata
        rec_genes
            list of genes whose reconstructed values should be retrieved
        """

        if adata is not None:
            if '_ontovae' not in adata.uns.keys():
                raise ValueError('Please run sconto_vae.module.utils.setup_anndata first.')
        else:
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
            r = self._pass_data(x, cat_list, 'rec')
            ravg = self._average_neuronnum(r)
            rec.append(ravg)
        rec = np.vstack(rec)

        # if genes were passed, subset
        if rec_genes is not None:
            onto_genes = adata.uns['_ontovae']['genes']
            gene_ind = np.array([onto_genes.index(g) for g in rec_genes])

            rec = rec[:,gene_ind]

        return rec

    
    @torch.no_grad()
    def perturbation(self, adata: AnnData=None, genes: list=[], values: list=[], output='terms', terms=None, rec_genes=None, lin_layer=True):
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
            - 'terms': retrieve pathway activities
            - 'genes': retrieve reconstructed values

        terms
            list of ontology term ids whose values should be retrieved
        rec_genes
            list of genes whose values should be retrieved
        lin_layer
            whether linear layer should be used for pathway activity retrieval
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

        covs = self._cov_tensor(adata)

        # generate dataloader
        dataloader = FastTensorDataLoader(adata.X, 
                                          covs,
                                         batch_size=128, 
                                         shuffle=False)

        # get pathway activities or reconstructed values after perturbation
        res = []
        for minibatch in dataloader:
            x = torch.tensor(minibatch[0].todense(), dtype=torch.float32).to(self.device)
            cat_list = torch.split(minibatch[1].T.to(self.device), 1)
            if output == 'terms':
                r = self._pass_data(x, cat_list, 'act', lin_layer)
                ravg = self._average_neuronnum(r)
                res.append(ravg)
            if output == 'genes':
                r = self._pass_data(x, cat_list, 'rec')
                ravg = self._average_neuronnum(r)
                res.append(ravg)

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

     
    
  
        


    




