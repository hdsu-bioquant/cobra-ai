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

class OntoVAEclass(scOntoVAE):
    """
    This class extends scOntoVAE with a classifier

    Parameters
    ----------
    adata
        anndata object that has been preprocessed with setup_anndata function
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
    classify_latent
        whether to only use latent space nodes as input to classifier
    
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
                 classify_latent: bool = True,
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
                        'average_neurons': average_neurons,
                        'classify_latent': classify_latent}
        self.params.update(class_params)

        if '_ontovae_class' not in self.adata.obs:
            raise ValueError('Please specify the class_key in sconto_vae.module.utils.setup_anndata_ontovae.')
        
        self.classify_latent = classify_latent
        self.n_terms = len(adata.uns['_ontovae']['annot'])
        self.n_classes = len(np.unique(adata.obs['_ontovae_class']))
        self.average_neurons = average_neurons
        self.class_features = self.n_terms if self.average_neurons else self.n_terms * self.neuronnum

        # Classifier
        self.classifier = Classifier(in_features = self.latent_dim if self.classify_latent else self.class_features,
                                     n_classes = self.n_classes,
                                     n_cat_list = self.n_cat_list,
                                     layer_dims = layer_dims_class,
                                     use_batch_norm = use_batch_norm_class,
                                     use_layer_norm = use_layer_norm_class,
                                     use_activation = use_activation_class,
                                     activation_fn = activation_fn_class,
                                     bias = bias_class,
                                     inject_covariates = inject_covariates_class,
                                     drop = drop_class
                                     )

        self.to(self.device)

  
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
        # encoding
        mu, log_var = self.encoder(x, cat_list)
            
        # sample from latent space
        z = self.reparameterize(mu, log_var)
        if self.use_activation_lat and self.use_activation_dec:
            z = self.activation_fn_dec()(z)

        if self.classify_latent:
            class_output = self.classifier(z, cat_list)
            reconstruction = self.decoder(z, cat_list)
        else:
            # attach hooks for classification
            activation = {}
            hooks = {}
            self._attach_hooks(activation=activation, hooks=hooks)

            # decoding
            reconstruction = self.decoder(z, cat_list)

            # classification
            act = torch.cat(list(activation.values()), dim=1)
            act = torch.hstack((z,act))
            if self.average_neurons:
                act = torch.stack(list(torch.split(act, self.neuronnum, dim=1)), dim=0).mean(dim=2).T
            class_output = self.classifier(act, cat_list)

            # removal of hooks
            for h in hooks:
                hooks[h].remove()
            
        return reconstruction, mu, log_var, class_output

    def clf_loss(self, class_output, y, mode='train', run=None):
        """
        Calculates loss of the classifier
        """
        class_loss = nn.CrossEntropyLoss()
        clf_loss = class_loss(class_output, y)
        if run is not None:
            run["metrics/" + mode + "/clf_loss"].log(clf_loss)
        return clf_loss

    def train_round(self, 
                    dataloader: FastTensorDataLoader, 
                    kl_coeff: float, 
                    clf_coeff: float,
                    optimizer: optim.Optimizer, 
                    run=None):
        """
        Parameters
        ----------
        dataloader
            pytorch dataloader instance with training data
        kl_coeff 
            coefficient for weighting Kullback-Leibler loss
        clf_coeff 
            coefficient for weighting classifier
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
            labels = torch.tensor(minibatch[2]).to(self.device)
            optimizer.zero_grad()

            # forward step
            reconstruction, mu, log_var, pred_labels = self.forward(data, cat_list)
            vae_loss = self.vae_loss(reconstruction, mu, log_var, data, kl_coeff, mode='train', run=run)
            clf_loss = self.clf_loss(pred_labels, labels, mode='train', run=run)
            loss = vae_loss + clf_coeff * clf_loss
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
                  clf_coeff: float,
                  run=None):
        """
        Parameters
        ----------
        dataloader
            pytorch dataloader instance with training data
        kl_coeff
            coefficient for weighting Kullback-Leibler loss
        clf_coeff 
            coefficient for weighting classifier
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
            labels = torch.tensor(minibatch[2]).to(self.device)

            # forward step
            reconstruction, mu, log_var, pred_labels = self.forward(data, cat_list)
            vae_loss = self.vae_loss(reconstruction, mu, log_var, data, kl_coeff, mode='val', run=run)
            clf_loss = self.clf_loss(pred_labels, labels, mode='val', run=run)
            loss = vae_loss + clf_coeff * clf_loss
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
                    clf_coeff: float=1e4,
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
        clf_coeff 
            coefficient for weighting classifier
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
                        'clf_coeff': clf_coeff,
                        'batch_size': batch_size,
                        'optimizer': str(optimizer).split("'")[1],
                        'epochs': epochs
                        }
        with open(modelpath + '/train_params.json', 'w') as fp:
            json.dump(train_params, fp, indent=4)

        # save model params
        with open(modelpath + '/model_params.json', 'w') as fp:
            json.dump(self.params, fp, indent=4)

        # train-val split
        train_adata, val_adata = split_adata(self.adata, 
                                             train_size = train_size,
                                             seed = seed)

        train_covs = self._cov_tensor(train_adata)
        val_covs = self._cov_tensor(val_adata)
        train_labels = torch.tensor(train_adata.obs['_ontovae_class'])
        val_labels = torch.tensor(val_adata.obs['_ontovae_class'])

        # generate dataloaders
        trainloader = FastTensorDataLoader(train_adata.X, 
                                           train_covs,
                                           train_labels,
                                         batch_size=batch_size, 
                                         shuffle=True)
        valloader = FastTensorDataLoader(val_adata.X, 
                                         val_covs,
                                         val_labels,
                                        batch_size=batch_size, 
                                        shuffle=False)

        val_loss_min = float('inf')
        optimizer = optimizer(self.parameters(), lr = lr)

        for epoch in range(epochs):
            print(f"Epoch {epoch+1} of {epochs}")
            train_epoch_loss = self.train_round(trainloader, kl_coeff, clf_coeff, optimizer, run)
            val_epoch_loss = self.val_round(valloader, kl_coeff, clf_coeff, run)
            
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
    def _pass_data(self, x, cat_list, output, lin_layer=True):
        """
        Passes data through the model.

        Parameters
        ----------
        x
            torch.tensor of shape (minibatch, in_features)
        cat_list
            Iterable of torch.tensors containing the category memberships
            shape of each tensor is (minibatch, 1)
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
        reconstruction, _, _, _ = self.forward(x, cat_list)

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
    def get_classification(self, adata: AnnData = None):
        """
        Gets classifications for a dataset.

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
        
        covs = self._cov_tensor(adata)

        # generate dataloaders
        dataloader = FastTensorDataLoader(adata.X, 
                                          covs,
                                         batch_size=128, 
                                         shuffle=False)

        self.eval()

        pred_classes = []
        for minibatch in dataloader:
            x = torch.tensor(minibatch[0].todense(), dtype=torch.float32).to(self.device)
            cat_list = torch.split(minibatch[1].T.to(self.device), 1)
            _, _, _, class_output = self.forward(x, cat_list)
            pred_classes.extend(class_output.to('cpu').detach().numpy())
        y_pred = np.argmax(np.array(pred_classes),axis=1)

        return y_pred

     
    
  
        


    




