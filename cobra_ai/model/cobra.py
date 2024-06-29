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

from cobra_ai.model.onto_vae import OntoVAE
from cobra_ai.module.modules import Classifier
from cobra_ai.module.metrics import knn_purity
from cobra_ai.module.utils import split_adata, FastTensorDataLoader, EarlyStopper, update_bn

from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

# imports for autotuning
from cobra_ai.module.decorators import classproperty
from cobra_ai.module.autotune import Tunable
from ray import train


class COBRA(OntoVAE):
    """
    This class extends OntoVAE with a CPA-like approach of disentangling covariate effects in the latent space in a linear fashion.

    Parameters
    ----------
    adata
        anndata object that has been preprocessed with setup_anndata function
    hidden_layers_class
        number of hidden layers
    neurons_per_class_layer
        number of neurons in a hidden layer
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
        if os.path.isfile(modelpath + '/covariate_mapping.json'):
            with open(modelpath + '/covariate_mapping.json', 'r') as fp:
                model.cov_dict = json.load(fp)
        return model
    
    def __init__(self, 
                 adata: AnnData, 
                 hidden_layers_class: Tunable[int] = 2,
                 neurons_per_class_layer: Tunable[int] = 64,
                 use_batch_norm_class: Tunable[bool] = True,
                 use_layer_norm_class: Tunable[bool] = False,
                 use_activation_class: Tunable[bool] = True,
                 activation_fn_class: Tunable[nn.Module] = nn.ReLU,
                 bias_class: Tunable[bool] = True,
                 inject_covariates_class: Tunable[bool] = False,
                 drop_class: Tunable[float] = 0.2,
                 average_neurons: Tunable[bool] = False,
                 extend_embedding: int=0,
                 **kwargs):
        super().__init__(adata, **kwargs)

        class_params = {'hidden_layers_class': hidden_layers_class,
                        'neurons_per_class_layer': neurons_per_class_layer,
                        'use_batch_norm_class': use_batch_norm_class,
                        'use_layer_norm_class': use_layer_norm_class,
                        'use_activation_class': use_activation_class,
                        'activation_fn_class': str(activation_fn_class).split("'")[1] if activation_fn_class is not None else activation_fn_class,
                        'bias_class': bias_class,
                        'inject_covariates_class': inject_covariates_class,
                        'drop_class': drop_class,
                        'average_neurons': average_neurons,
                        'extend_embedding': extend_embedding}
        self.params.update(class_params)

        
        # set up covariates
        self.cpa_covs = adata.obsm['_cobra_categorical_covs'] if '_cobra_categorical_covs' in adata.obsm.keys() else None
        if self.cpa_covs is None:
            raise ValueError('Please specify cpa_keys in setup_anndata_ontovae to run the model.')

        self.cov_dict = adata.uns['cov_dict']
        self.inject = {}
        for key in self.cov_dict.keys():
            self.inject[key] = False
        self.inject_cov_dict = {}

        # embedding of covars
        self.covars_embeddings = nn.ModuleDict(
            {
                key: nn.ModuleList((torch.nn.Embedding(len(self.cov_dict[key]), self.latent_dim),
                                   torch.nn.Embedding(extend_embedding, self.latent_dim)))
                for key in self.cov_dict.keys()
            }
        )
        
        # covars classifiers
        self.covars_classifiers = nn.ModuleDict(
            {
                key: Classifier(in_features = self.latent_dim,
                                n_classes = len(self.cov_dict[key]),
                                n_cat_list = self.n_cat_list,
                                hidden_layers = hidden_layers_class,
                                neurons_per_layer = neurons_per_class_layer,
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

    def configure_adata(self, adata: AnnData):
        """
        Configures anndata with previously unseen categories for the embedding layers.
        
        """
        self.adata = adata

        for cov in list(self.cov_dict.keys()):
            new_cats = [cat for cat in adata.obs.loc[:,cov].unique() if cat not in self.cov_dict[cov].keys()]
            if len(new_cats) > 0:
                self.inject[cov] = True
                self.inject_cov_dict[cov] = adata.uns['cov_dict'][cov]
            else:
                self.inject[cov] = False

    def _get_embedding(self, x: torch.tensor, cat_list: Iterable[torch.tensor], cov_list: Iterable[torch.tensor], mixup_lambda=1):
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
        mixup_lambda
            coefficient for adversarial training
        """
        # data mixup for adversarial training
        batch_size = x.size()[0]
        if mixup_lambda  < 1:
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

        # covariate encoding
        covars_embeddings = {}
        for i, key in enumerate(self.covars_embeddings.keys()):
            covs = cov_list[i].long().squeeze()
            ind = 0 if not self.inject[key] else 1
            x = self.covars_embeddings[key][ind](covs)
            x_mix = self.covars_embeddings[key][ind](covs[index])
            x_new = mixup_lambda * x + (1. - mixup_lambda) * x_mix
            covars_embeddings[key] = x_new

        # create different z's
        z_cov = {}
        z_total = z_basal.clone()
        for key in covars_embeddings.keys():
            z_cov['z_' + key] = (z_basal + covars_embeddings[key])
            z_total += covars_embeddings[key]

        z_dict = dict(z_basal=z_basal)
        z_dict.update(z_cov, z_total=z_total)

        return z_dict, mu, log_var
  
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
        # inference
        zdict, mu, log_var = self._get_embedding(x, cat_list, cov_list, mixup_lambda)

        # decoding
        reconstruction = self.decoder(zdict['z_total'], cat_list)
            
        return zdict, mu, log_var, reconstruction

    def adv_forward(self, z: torch.tensor, cat_list: Iterable[torch.tensor], compute_penalty=False):
        """
        Forward computation on minibatch of samples for z_basal.
        
        Parameters
        ----------
        z
            torch.tensor of shape (minibatch, in_features)
        cat_list
            Iterable of torch.tensors containing the category memberships
            shape of each tensor is (minibatch, 1)
        """
        if compute_penalty:
            z = z.requires_grad_(True)

        # covariate classifiers on z_basal
        covars_pred = {}
        for key in self.covars_classifiers.keys():
            covar_pred = self.covars_classifiers[key](z, cat_list)
            covars_pred[key] = covar_pred
        
        if compute_penalty:
            penalty = 0.0
            # Penalty losses
            for key in self.covars_classifiers.keys():
                pen = (
                    torch.autograd.grad(
                        covars_pred[key].sum(),
                        z,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                    )[0].pow(2).mean()
                )
                penalty += pen
            covars_pred['penalty'] = penalty
        
        return covars_pred

    def clf_loss(self, class_output, y, cov: str, run=None):
        """
        Calculates loss of a covariate classifier
        """
        class_loss = nn.CrossEntropyLoss()
        clf_loss = class_loss(class_output, y)
        if run is not None:
            mode = "train" if self.training else "val"
            run["metrics/" + mode + "/" + cov + "_clf_loss"].log(clf_loss)
        return clf_loss

    def train_round(self, 
                    dataloader: FastTensorDataLoader, 
                    kl_coeff: float, 
                    adv_coeff: float,
                    pen_coeff: float,
                    mixup_lambda: float,
                    adv_step: int,
                    optimizer_vae: optim.Optimizer, 
                    optimizer_adv: optim.Optimizer,
                    pos_weights: bool,
                    swa_model,
                    swa_per_epoch: bool,
                    epoch: int,
                    swa_start: int,
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
        pen_coeff
            coefficient for weighting gradient penalty
        mixup_lambda
            coefficient for adversarial training
        adv_step:
            after how many minibatches the discriminators should be updated
        optimizer_vae
            optimizer for training the VAE
        optimizer_adv
            optimizer for training the adversarial component
        pos_weights:
            whether to make weights in decoder positive
        run
            Neptune run if training is to be logged
        """
        # set to train mode
        self.train()

        # initialize running losses
        running_loss_vae = 0.0
        running_loss_adv = 0.0

        # init purity
        purity = 0.0

        # iterate over dataloader for training
        for i, minibatch in tqdm(enumerate(dataloader), total=len(dataloader)):

            # move minibatch to device
            data = torch.tensor(minibatch[0].todense(), dtype=torch.float32).to(self.device)
            cat_list = torch.split(minibatch[1].T.to(self.device), 1)
            cov_list = torch.split(minibatch[2].T.to(self.device), 1)
            
            # VAE optimizer
            optimizer_vae.zero_grad()

            # forward step generator
            z_dict, mu, logvar, reconstruction = self.forward(data, cat_list, cov_list, mixup_lambda)
            z_basal = z_dict["z_basal"]
            covars_pred = self.adv_forward(z_basal, cat_list)
            vae_loss = self.vae_loss(reconstruction, mu, logvar, data, kl_coeff, run=run)
            adv_loss = 0.0
            for i, vals in enumerate(cov_list):
                cov = list(self.cov_dict.keys())[i]
                cov_loss = self.clf_loss(covars_pred[cov], vals.long().squeeze(), cov=cov, run=run)
                adv_loss += cov_loss
            loss = vae_loss - adv_coeff * adv_loss
            running_loss_vae += loss.item()

            # backward propagation
            loss.backward()

            # zero out gradients from non-existent connections
            for i in range(self.start_point, len(self.decoder.decoder)):
                self.decoder.decoder[i][0].weight.grad = torch.mul(self.decoder.decoder[i][0].weight.grad, self.decoder.masks[i-self.start_point])

            # perform optimizer step
            optimizer_vae.step()

            # make weights in Onto module positive
            if pos_weights:
                for i in range(self.start_point, len(self.decoder.decoder)):
                    self.decoder.decoder[i][0].weight.data = self.decoder.decoder[i][0].weight.data.clamp(0)

            # adversarial training
            if i % adv_step == 0:
                # adversarial optimizer
                optimizer_adv.zero_grad()

                # forward step discriminator
                covars_pred = self.adv_forward(z_basal.detach(), cat_list, compute_penalty=True)
                adv_loss = 0.0
                for i, vals in enumerate(cov_list):
                    cov = list(self.cov_dict.keys())[i]
                    cov_loss = self.clf_loss(covars_pred[cov], vals.long().squeeze(), cov=cov, run=run)
                    adv_loss += cov_loss
                loss = adv_loss + pen_coeff * covars_pred['penalty']
                running_loss_adv += loss

                # backward propagation
                loss.backward()
                optimizer_adv.step()
            
            # compute KNN purity
            cov_purity = []
            for i, vals in enumerate(cov_list):
                cov = list(self.cov_dict.keys())[i]
                cov_purity.append(knn_purity(z_basal.to('cpu').detach().numpy(), vals.long().squeeze().to('cpu').detach().numpy()))
                cov_purity.append(-knn_purity(z_dict['z_' + cov].to('cpu').detach().numpy(), vals.long().squeeze().to('cpu').detach().numpy()))
            purity += np.sum(cov_purity)

            # swa averaging
            if swa_model is not None:
                if not swa_per_epoch:
                    if epoch >= swa_start:
                        swa_model.update_parameters(self)



        # compute avg training loss
        train_loss_vae = running_loss_vae/len(dataloader)
        train_loss_adv = running_loss_adv/len(dataloader)

        # compute average purity
        avg_purity = purity/len(dataloader)

        return train_loss_vae, train_loss_adv, avg_purity

    @torch.no_grad()
    def val_round(self, 
                  dataloader: FastTensorDataLoader, 
                  kl_coeff: float, 
                  adv_coeff: float,
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
        run
            Neptune run if training is to be logged
        """
        # set to eval mode
        self.eval()
        #mixup_lambda=0

        # initialize running losses
        running_loss_vae = 0.0

        # init purity
        purity = 0.0

        # iterate over dataloader for validation
        for i, minibatch in tqdm(enumerate(dataloader), total=len(dataloader)):

            # move minibatch to device
            data = torch.tensor(minibatch[0].todense(), dtype=torch.float32).to(self.device)
            cat_list = torch.split(minibatch[1].T.to(self.device), 1)
            cov_list = torch.split(minibatch[2].T.to(self.device), 1)

            # forward step generator
            z_dict, mu, logvar, reconstruction = self.forward(data, cat_list, cov_list, mixup_lambda=1)
            z_basal = z_dict["z_basal"]
            covars_pred = self.adv_forward(z_basal, cat_list)
            vae_loss = self.vae_loss(reconstruction, mu, logvar, data, kl_coeff, run=run)
            adv_loss = 0.0
            for i, vals in enumerate(cov_list):
                cov = list(self.cov_dict.keys())[i]
                cov_loss = self.clf_loss(covars_pred[cov], vals.long().squeeze(), cov=cov, run=run)
                adv_loss += cov_loss
            loss = vae_loss - adv_coeff * adv_loss
            running_loss_vae += loss.item()

            # compute KNN purity
            cov_purity = []
            for i, vals in enumerate(cov_list):
                cov = list(self.cov_dict.keys())[i]
                cov_purity.append(knn_purity(z_basal.to('cpu').detach().numpy(), vals.long().squeeze().to('cpu').detach().numpy()))
                cov_purity.append(-knn_purity(z_dict['z_' + cov].to('cpu').detach().numpy(), vals.long().squeeze().to('cpu').detach().numpy()))
            purity += np.sum(cov_purity)

        # compute avg training loss
        val_loss_vae = running_loss_vae/len(dataloader)

        # compute average purity
        avg_purity = purity/len(dataloader)

        return val_loss_vae, avg_purity

    def train_model(self, 
                    modelpath: str, 
                    save: bool = True,
                    train_size: float = 0.85,
                    seed: int = 42,
                    lr_vae: Tunable[float]=1e-4, 
                    lr_adv: Tunable[float]=1e-3,
                    kl_coeff: Tunable[float]=1e-3, 
                    adv_coeff: Tunable[float]=1e3,
                    pen_coeff: Tunable[float]=2.0,
                    mixup_lambda: Tunable[float]=1,
                    adv_step: Tunable[int]=1,
                    batch_size: Tunable[int]=128, 
                    optimizer: Tunable[optim.Optimizer] = optim.AdamW,
                    pos_weights: Tunable[bool] = True,
                    use_rec_weights: bool = False,
                    epochs: int=300, 
                    early_stopping: bool=True,
                    patience: int=10,
                    perform_swa: bool=False,
                    swa_per_epoch: bool=True,
                    swa_start: int=25,
                    swa_epochs: int=10,
                    run=None):
        """
        Parameters
        ----------
        modelpath
            path to a folder where to store the params and the best model 
        save
            boolean value, if best model should be saved or not
        train_size
            which percentage of samples to use for training
        seed
            seed for the train-val split
        lr_vae
            learning rate for the VAE optimizer
        lr_adv
            learning rate for the adversarial optimizer
        kl_coeff
            Kullback Leibler loss coefficient
        adv_coeff 
            coefficient for weighting classifier
        pen_coeff
            coefficient for weighting gradient penalty
        mixup_lambda
            coefficient for adversarial training
        adv_step
            after how many minibatches the discriminators should be updated
        batch_size
            size of minibatches
        optimizer
            which optimizer to use
        pos_weights
            whether to make weights in decoder positive
        epochs
            over how many epochs to train
        early_stopping
            if early stopping should be used during training
        patience
            number of epochs after which training stops if loss is not improved
        run
            passed here if logging to Neptune should be carried out
        """

        if save:
            if os.path.isfile(modelpath + '/best_model.pt'):
                print("A model already exists in the specified directory and will be overwritten.")

            # save train params
            train_params = {'train_size': train_size,
                            'seed': seed,
                            'lr_vae': lr_vae,
                            'lr_adv': lr_adv,
                            'kl_coeff': kl_coeff,
                            'adv_coeff': adv_coeff,
                            'pen_coeff': pen_coeff,
                            'mixup_lambda': mixup_lambda,
                            'adv_step': adv_step,
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

            # save covariate dictionary
            with open(modelpath + '/covariate_mapping.json', 'w') as fp:
                json.dump(self.cov_dict, fp, indent=4)
            if np.any(np.array(list(self.inject.values()))):
                with open(modelpath + '/injected_covariate_mapping.json', 'w') as fp:
                    json.dump(self.inject_cov_dict, fp, indent=4)

        # train-val split
        train_adata, val_adata = split_adata(self.adata, 
                                             train_size = train_size,
                                             seed = seed)

        train_batch = self._cov_tensor(train_adata)
        val_batch = self._cov_tensor(val_adata)

        train_covs = torch.tensor(np.array(train_adata.obsm['_cobra_categorical_covs'], dtype='int64'))
        val_covs = torch.tensor(np.array(val_adata.obsm['_cobra_categorical_covs'], dtype='int64'))

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

        # compute reconstruction weights
        if use_rec_weights:
            weights = torch.tensor(np.var(np.array(self.adata.X.todense()), axis=0), dtype=torch.float32)
            self.rec_weights = torch.mul(weights, torch.div(weights[weights != 0].size(dim=0), torch.sum(weights,))).to(self.device)
        else:
            self.rec_weights = None

        val_loss_min = float('inf')
        val_purity_min = float('inf')

        optimizer_vae = optimizer(list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.covars_embeddings.parameters()), lr = lr_vae)
        scheduler_vae = CosineAnnealingLR(optimizer_vae, T_max=100)
        
        optimizer_adv = optimizer(self.covars_classifiers.parameters(), lr = lr_adv)
        scheduler_adv = CosineAnnealingLR(optimizer_adv, T_max=100)

        if perform_swa:
            swa_model = AveragedModel(self)
            swa_scheduler_vae = SWALR(optimizer_vae, swa_lr = 0.05)
            swa_scheduler_adv = SWALR(optimizer_adv, swa_lr = 0.05)
        else:
            swa_model = None
        
        if early_stopping:
                early_stopper = EarlyStopper(patience=patience)
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1} of {epochs}")

            if perform_swa and epoch > swa_start + swa_epochs:
                break

            train_epoch_loss_vae, train_epoch_loss_adv, train_knn_purity = self.train_round(trainloader, 
                                                                                            kl_coeff, 
                                                                                            adv_coeff, 
                                                                                            pen_coeff, 
                                                                                            mixup_lambda, 
                                                                                            adv_step, 
                                                                                            optimizer_vae, 
                                                                                            optimizer_adv,
                                                                                            pos_weights,
                                                                                            swa_model,
                                                                                            swa_per_epoch,
                                                                                            epoch,
                                                                                            swa_start,
                                                                                            run)
            
            if perform_swa:
                if epoch >= swa_start:
                    if swa_per_epoch:
                        swa_model.update_parameters(self)
                    else:
                        update_bn(trainloader, swa_model)
                        self.load_state_dict(swa_model.module.state_dict())
                    swa_scheduler_vae.step()
                    swa_scheduler_adv.step()
                else:
                    scheduler_vae.step()
                    scheduler_adv.step()
            else:
                scheduler_vae.step()
                scheduler_adv.step()
                    
            val_epoch_loss_vae, val_knn_purity = self.val_round(valloader, 
                                                                kl_coeff, 
                                                                adv_coeff,
                                                                run)
            
            if early_stopping and not perform_swa:
                if early_stopper.early_stop(val_epoch_loss_vae):
                    break
            if early_stopping and perform_swa:
                if early_stopper.early_stop(val_epoch_loss_vae) and epoch < swa_start:
                    swa_start = epoch

            train.report({"validation_loss": val_epoch_loss_vae})
            
            if run is not None:
                run["metrics/train/loss_vae"].log(train_epoch_loss_vae)
                run["metrics/train/loss_adv"].log(train_epoch_loss_adv)
                run["metrics/train/knn_purity"].log(train_knn_purity)
                run["metrics/val/loss_vae"].log(val_epoch_loss_vae)
                run["metrics/val/knn_purity"].log(val_knn_purity)
            
            if perform_swa and epoch >= swa_start:
                val_loss_min = float('inf')
            if val_epoch_loss_vae < val_loss_min and save:
                print('New best model!')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_vae_state_dict': optimizer_vae.state_dict(),
                    'optimizer_adv_state_dict': optimizer_adv.state_dict(),
                    'knn_purity': val_knn_purity,
                }, modelpath + '/best_model.pt')
                val_loss_min = val_epoch_loss_vae

            if perform_swa and swa_per_epoch:
                update_bn(trainloader, swa_model)
                self.load_state_dict(swa_model.module.state_dict())
                

    @torch.no_grad()
    def _pass_data(self, x, cat_list, cov_list, retrieve: Literal['act', 'rec'], lin_layer=True):
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
        retrieve
            'act': return pathway activities
            'rec': return reconstructed values
        lin_layer:
            whether hooks should be attached to linear layer of the model
        """

        # set to eval mode
        self.eval()

        # get latent space embedding dict
        zdict, _, _ = self._get_embedding(x, cat_list, cov_list)
        dict_keys = list(zdict.keys())

        # pass forward the different z's
        act_dict = {}

        for z_key in dict_keys:
            z = zdict[z_key].clone()

            # attach the hooks
            if retrieve == 'act':
                activation = {}
                hooks = {}
                self._attach_hooks(lin_layer=lin_layer, activation=activation, hooks=hooks)

            # pass data through model
            reconstruction = self.decoder(z, cat_list)

            # return pathway activities or reconstructed gene values
            if retrieve == 'act':
                act = torch.cat(list(activation.values()), dim=1)
                for h in hooks:
                    hooks[h].remove()
                if self.root_layer_latent:
                    act_dict[z_key] = torch.hstack((z,act))
                else:
                    act_dict[z_key] = act
            else:
                act_dict[z_key] = reconstruction
        return act_dict
    
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
                raise ValueError('Please run cobra_ai.module.utils.setup_anndata first.')
        else:
            adata = self.adata

        batch = self._cov_tensor(adata)
        covs = torch.tensor(np.array(adata.obsm['_cobra_categorical_covs'], dtype='int64'))

        dataloader = FastTensorDataLoader(adata.X, 
                                          batch,
                                          covs,
                                         batch_size=128, 
                                         shuffle=False)

        res = []
        for minibatch in dataloader:
            x = torch.tensor(minibatch[0].todense(), dtype=torch.float32).to(self.device)
            cat_list = torch.split(minibatch[1].T.to(self.device), 1)
            cov_list = torch.split(minibatch[2].T.to(self.device), 1)
            if retrieve == 'latent':
                result, _, _ = self._get_embedding(x, cat_list, cov_list)
                if self.root_layer_latent:
                    result_avg = {k: self._average_neuronnum(v.to('cpu').detach().numpy()) for k, v in result.items()}
                else:
                    result_avg = {k: v.to('cpu').detach().numpy() for k, v in result.items()}
                res.append(result_avg)
            else:
                result = self._pass_data(x, cat_list, cov_list, retrieve, lin_layer)
                if retrieve == 'act':
                    result_avg = {k: self._average_neuronnum(v.to('cpu').detach().numpy()) for k, v in result.items()}
                else:
                    result_avg = {k: v.to('cpu').detach().numpy() for k, v in result.items()}
                res.append(result_avg)

        res_out = {}
        key_list = list(res[0].keys())
        for key in key_list:
            res_key = np.vstack([r[key] for r in res])
            res_out[key] = res_key

        return res_out
    
    @torch.no_grad()
    def to_latent(self, adata: AnnData=None):
        """
        Retrieves different representations of z.

        Parameters
        ----------
        adata
            AnnData object that was processed with setup_anndata
        """
        self.eval()
        res = self._run_batches(adata, retrieve='latent')
        return res
    
    @torch.no_grad()
    def get_pathway_activities(self, adata: AnnData=None, lin_layer=True):
        """
        Retrieves pathway activities from latent space and decoder.

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
        res = self._run_batches(adata, retrieve='act', lin_layer=lin_layer)
        return res


    @torch.no_grad()
    def get_reconstructed_values(self, adata: AnnData=None):
        """
        Retrieves reconstructed values from output layer.

        Parameters
        ----------
        adata
            AnnData object that was processed with setup_anndata
        """
        self.eval()
        res = self._run_batches(adata, retrieve='rec')
        return res

    @torch.no_grad()
    def decode(self, adata: AnnData, embedding: np.array, retrieve: Literal['act', 'rec'], lin_layer=True): 
        """
        Passes a user-defined embedding through the decoder.

        Parameters:
        ----------
        adata: An AnnData object processed with setup_anndata_ontovae
        embedding: An embedding (z) from the latent space
        lin_layer: whether hooks should be attached to linear layer of the model
        retrieve: 'act' for pathway activity; 'rec' for reconstructions
        
        Returns:
        ----------
        Pathway activity or reconstruction
        """

        # set to eval mode
        self.eval()
        
        if adata is not None:
            if '_ontovae' not in adata.uns.keys():
                raise ValueError('Please run cobra_ai.module.utils.setup_anndata first.')
        else:
            adata = self.adata 
        
        batch = torch.zeros((embedding.shape[0], self._cov_tensor(adata).shape[1]), dtype=torch.int8)
        
        embedding = torch.tensor(embedding, dtype=torch.float32)
        dataloader = FastTensorDataLoader(embedding, 
                                        batch,
                                        batch_size=128, 
                                        shuffle=False)
        # pass forward the different z's
        if retrieve == 'act':
            
            res = []
            for minibatch in dataloader:
                self.eval()
                z = minibatch[0].to(self.device)
                cat_list = torch.split(minibatch[1].T.to(self.device), 1)
                
                activation = {}
                hooks = {}
                self._attach_hooks(lin_layer=lin_layer, activation=activation, hooks=hooks)
                
                reconstruction = self.decoder(z, cat_list)
                
                act = torch.cat(list(activation.values()), dim=1)
                for h in hooks:
                    hooks[h].remove()
            
                result_avg = self._average_neuronnum(act.to('cpu').detach().numpy())
                res.append(result_avg)
            res_out = np.vstack([r for r in res])
        else:
            res = []
            for minibatch in dataloader:
                self.eval()
                z = minibatch[0].to(self.device)
                cat_list = torch.split(minibatch[1].T.to(self.device), 1)
                
                activation = {}
                hooks = {}
                self._attach_hooks(lin_layer=lin_layer, activation=activation, hooks=hooks)
                
                reconstruction = self.decoder(z, cat_list)
                res.append(reconstruction.to('cpu').detach().numpy())
            res_out = np.vstack([r for r in res])
                
                
        return res_out

    @classproperty
    def _tunables(cls):
        return [cls.__init__, cls.train_model]
    
    @classproperty
    def _metrics(cls):
        ''' Maybe should provide the metric in the manner ["name", "mode"]'''
        return ["validation_loss"]
