import json
import os
import numpy as np
import pandas as pd
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from typing import Literal, List, Union
import warnings

from scipy.sparse import csr_matrix
import scanpy as sc
import anndata as ad
from anndata import AnnData

import matplotlib.pyplot as plt

from abc import ABC, abstractmethod

from cobra_ai.module.modules import Encoder, Decoder, GradientReversalLayer, Classifier
from cobra_ai.module.utils.distributions import Reparameterize
from cobra_ai.module.utils.utils import split_adata, FastTensorDataLoader, EarlyStopper
from cobra_ai.module.utils.losses import KL_Divergence, MSE_Loss, ContrastiveLoss, CrossEntropyLoss, MMDLoss
from cobra_ai.module.ontobj import Ontobj


class BaseVAE(nn.Module, ABC):
    """
    Backbone class for the different VAE models.
    """

    @classmethod
    def load(
        cls, 
        modelpath: str,
        ):
        """
        This function allows loading of a trained model.

        Parameters
        ----------
        modelpath: str
            path where the model is stored
        """
        params = cls._read_params(modelpath)
        model = cls()
        model.construct(**params) 
        checkpoint = torch.load(modelpath + '/checkpoint.pt',
                            map_location = torch.device(model.device))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.val_loss_min = checkpoint['loss']
        return model
    
    @classmethod
    def _read_params(
        cls,
        modelpath
        ):
        with open(modelpath + '/model_hyperparams.json', 'r') as fp:
            params = json.load(fp)
        if params['activation_fn'] is not None:
            params['activation_fn'] = eval(params['activation_fn'])
        params['trained'] = True
        return params
       
    def __init__(self):
        super().__init__()

    def construct(
            self, 
            adata: AnnData, 
            batch_key: str = None,
            latent_dim: int = 128,
            hidden_dims: list = [256,256],
            hidden_dims_classifier: list = [64],
            normalisation: Literal["batch", "layer"] = "batch",
            activation_fn: nn.Module = nn.ReLU,
            bias: bool = True,
            dropout_rate: float = 0.2,
            z_dropout: float = 0.5,
            input_features: list = [],
            batch_idx: dict = {},
            trained: bool = False,
            ):
        """
        Function to construct the VAE model.

        Backbone Parameters
        -------------------
        adata
            anndata object containing the RNA data (has been setup with setup_anndata_vanillavae)
        batch_key
            column in adata.obs that contains batch information (optional)
        latent_dim
            latent dimension for RNA modality
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
        input_features
            list of input features (used internally when loading a trained model)
        batch_idx
            dictionary of batch to index mapping (used internally when loading a trained model)
        trained
            whether the model is already trained or not (used internally when loading a trained model)
        
        """
        self.model = 'vanilla'
        self.trained = trained
        if not self.trained:
            self.val_loss_min = float('inf')

         # general parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hyperparams = {}
        self.adata = adata
        self.input_features = self.input_features
        self.z_dropout = z_dropout

        # parse batch information
        self.batch_key = batch_key
        self.batch_names = []
        self.new_batch_names = []
        self.batch_idx = batch_idx
        if self.batch_idx is not None:
            self.batch_names = list(self.batch_idx.keys())

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
            'latent_dim': latent_dim,
            'hidden_dims': hidden_dims,
            'normalisation': normalisation,
            'activation_fn': str(activation_fn).split("'")[1],
            'bias': bias,
            'dropout_rate': dropout_rate,
            'z_dropout': z_dropout,
        })

        # Encoder
        self.encoder = Encoder(
            n_features = len(self.input_features),
            hidden_dims = hidden_dims,
            latent_dim = latent_dim,
            batch_names = self.batch_names,
            normalisation = normalisation,
            activation_fn = activation_fn,
            bias = bias,
            dropout_rate = dropout_rate 
        )

        # Decoder
        self.decoder = Decoder(
            n_features = len(self.input_features),
            hidden_dims = hidden_dims,
            latent_dim = latent_dim,
            batch_names = self.batch_names,
            normalisation = normalisation,
            activation_fn = activation_fn,
            bias = bias,
            dropout_rate = dropout_rate
        )

        # optional: batch alignment
        if self.batch_key or len(self.batch_idx) > 0:
            self.grl = GradientReversalLayer()
            self.batch_classifier = Classifier(
                n_features = latent_dim,
                hidden_dims = hidden_dims_classifier,
                n_classes = len(self.batch_idx),
                normalisation = normalisation,
                activation_fn = activation_fn,
                bias = bias,
                dropout_rate = dropout_rate
            )

        self.to(self.device)

        self.initialized = True
   

    def _update_batch(
            self,
            adata: AnnData,
            batch_key: str,
            add_layers: bool = False
            ):
        """
        Updates batch information.

        Parameters
        ----------
        adata
            AnnData object containing the data
        batch_key
            column in adata.obs that contains batch information
        add_layers
            whether to add batch layers to encoder and decoder
        """

        assert batch_key in adata.obs.columns, batch_key + " not found in adata.obs"

        self.batch_key = batch_key
        self.batch_names = list(adata.obs[batch_key].astype(str).unique()) # unique batch names in the batch column

        self.new_batch_names = [n for n in self.batch_names if n not in self.batch_idx.keys()]
        self.batch_idx.update({n: i + len(self.batch_idx) for i, n in enumerate(self.new_batch_names)}) # create mapping from batch to index
        self.hyperparams['batch_idx'] = self.batch_idx

        if add_layers:
            self.encoder.encoder.add_batches(self.new_batch_names)
            self.decoder.decoder.add_batches(self.new_batch_names)

    
    def encode(self, x: torch.tensor):
        """
        Generates latent space embedding.

        Parameters
        ----------
        x
            torch.tensor of shape (minibatch, n_features + n_batches)
        """
        mu, log_var = self.encoder(x)
        z = Reparameterize()(mu, log_var)
        return z, mu, log_var
    
    def decode(self, z: torch.tensor):
        """
        Decodes a latent space embedding.

        Parameters
        ----------
        z
            torch.tensor of shape (minibatch, latent_dim + n_batches)
        """
        out = self.decoder(z)
        return out


    def forward(self, x: torch.tensor):
        """
        Forward computation on minibatch of samples.
        
        Parameters
        ----------
        x
            torch.tensor of shape (minibatch, in_features + n_batches)
        """

        _, x_batch = torch.split(x, [len(self.input_features), x.shape[1]-len(self.input_features)], dim=1)

        z, mu, log_var = self.encode(x)
        if self.z_dropout > 0:
            if self.training:
                z = nn.Dropout(p=self.z_dropout)(z)
        out = self.decoder(torch.cat((z, x_batch), dim=1))
        return {
            'z': z,
            'mu': mu,
            'log_var': log_var,
            'out': out
        }
    
    def training_step(
            self, 
            dataloader: FastTensorDataLoader, 
            kl_weight: float, 
            grl_weight: float,
            contrastive_weight: float,
            mmd_weight: float,
            adv_weight: float,
            gp_weight: float,
            adv_step: int,
            pos_weights: bool,
            optimizers: dict, 
            log_prefix: str,
            run=None
            ):
        """
        Parameters
        ----------
        dataloader
            pytorch dataloader instance with training data
        kl_weight 
            coefficient for weighting Kullback-Leibler loss
        contrastive_weight
            coefficient for weighting contrastive loss for batch integration
        optimizer
            optimizer for training
        log_prefix
            string to be added to the Neptune log path
        run
            Neptune run if training is to be logged
        """

        # set to train mode
        self.train()

        # initialize running loss
        running_loss = 0.0

        # iterate over dataloader for training
        for minibatch in dataloader:
            
            # unpack minibatch
            x_features, x_batches, x_labels, x_cobra_covs, x = self._unpack(minibatch, self.device)
            if x_features.shape[0] == 1:
                break
                
            # forward step
            out_dict = self.forward(x)

            # compute adversarial loss for batch alignment if applicable
            if hasattr(self, 'batch_classifier'):
                batch_logits = self.batch_classifier(out_dict['z'].detach())
                adversarial_loss = CrossEntropyLoss()(
                    batch_logits, 
                    x_batches,
                    mode='train',
                    log_prefix=log_prefix,
                    run=run
                )

                optimizers['classifier'].zero_grad()
                adversarial_loss.backward()
                optimizers['classifier'].step()

            # compute losses
            loss = self.compute_loss(
                x_features,
                x_batches,
                x_labels,
                out_dict, 
                kl_weight, 
                grl_weight,
                contrastive_weight,
                mmd_weight,
                mode='train', 
                log_prefix=log_prefix,
                run=run
                )
            
            running_loss += loss.item()

            # update weights
            optimizers['vae'].zero_grad()
            loss.backward()

            # if applicable, zero out gradients from non-existent connections
            if self.model == 'ontovae':
                for i in range(self.start_point, len(self.decoder.decoder.fc_layers)):
                    self.decoder.decoder.fc_layers[i][0].weight.grad = torch.mul(self.decoder.decoder.fc_layers[i][0].weight.grad, self.decoder.masks[i-self.start_point])
                    self.decoder.reconstruction[0].weight.data = torch.mul(self.decoder.reconstruction[0].weight.data, self.decoder.masks[-1])

            optimizers['vae'].step()

            # if selected, make weights in Ontodecoder positive
            if self.model == 'ontovae' and pos_weights:
                for i in range(self.start_point, len(self.decoder.decoder.fc_layers)):
                    self.decoder.decoder.fc_layers[i][0].weight.data = self.decoder.decoder.fc_layers[i][0].weight.data.clamp(0)
                    self.decoder.reconstruction[0].weight.data = self.decoder.reconstruction[0].weight.data.clamp(0)

            self.pbar['train'].update(1)

        # compute avg training loss
        train_loss = running_loss/len(dataloader)
        return {
            'train_loss_vae': train_loss
        }


    @torch.no_grad()
    def validation_step(
        self, 
        dataloader: FastTensorDataLoader, 
        kl_weight: float, 
        grl_weight: float,
        contrastive_weight: float,
        mmd_weight: float,
        adv_weight: float,
        log_prefix: str,
        run=None
        ):
        """
        Parameters
        ----------
        dataloader
            pytorch dataloader instance with training data
        kl_weight
            coefficient for weighting Kullback-Leibler loss
        contrastive_weight
            coefficient for weighting contrastive loss for batch integration
        log_prefix
            string to be added to the Neptune log path
        run
            Neptune run if training is to be logged
        """
        # set to eval mode
        self.eval()

        # initialize running loss
        running_loss = 0.0

        # iterate over dataloader for validation
        for minibatch in dataloader:

            # unpack minibatch
            x_features, x_batches, x_labels, x_cobra_covs, x = self._unpack(minibatch, self.device)
            if x_features.shape[0] == 1:
                break
                
            # forward step
            out_dict = self.forward(x)

            # compute losses
            loss = self.compute_loss(
                x_features,
                x_batches,
                x_labels,
                out_dict, 
                kl_weight, 
                grl_weight,
                contrastive_weight,
                mmd_weight,
                mode='val', 
                log_prefix=log_prefix,
                run=run
                )
            running_loss += loss.item()
            
            self.pbar['val'].update(1)

        # compute avg val loss
        val_loss = running_loss/len(dataloader)
        return {
            'val_loss_vae': val_loss
        }
    

    def fit(
            self, 
            modelpath: str,
            adata: AnnData = None,
            batch_key: str = None, 
            celltype: str = None,
            train_size: float = 0.9,
            seed: int = 42,
            learning_rate: float=1e-4, 
            learning_rate_adv: float=1e-3,
            betas: tuple = (0.5, 0.9),
            weight_decay: float = 1e-3,
            kl_weight: float=1e-4, 
            grl_weight: float=100.0,
            contrastive_weight: float=0.0,
            mmd_weight: float = 0.0,
            adv_weight: float = 1e-3,
            gp_weight: float = 2.0,
            adv_step: int=1,
            batch_size: int=128, 
            pos_weights: bool = True,
            optimizer: optim.Optimizer = optim.AdamW,
            epochs: int = 500, 
            early_stopping: bool = True,
            patience: int = 30,
            fine_tune: bool = True,
            log_prefix: str = "",
            run=None
            ):
        """
        Parameters
        ----------
        modelpath
            path to a folder where to store the params and the best model 
        adata
            AnnData object contaning the data 
        batch_key 
            column in adata.obs that contains batch information (optional)
        celltype
            column in adata.obs that contains celltype information (optional)
        train_size
            which percentage of samples to use for training
        seed
            seed for the train-val split
        learning_rate
            learning rate for the VAE
        learning_rate_adv
            learning rate for the adversarial part of COBRA
        kl_weight
            Kullback Leibler loss coefficient
        grl_weight
            weight for the Gradient Reversal Layer
        contrastive_weight
            weight for using contrastive loss for batch correction
        mmd_weight
            weight for using MMD loss for batch correction
        adv_weight
            weight for the adversarial part of COBRA
        gp_weight
            gradient penalty weight for COBRA
        adv_step
            adversarial training will be performed every nth step in COBRA
        batch_size
            size of minibatches
        pos_weights
            whether to use positive weights only in the decoder
        optimizer
            which optimizer to use
        epochs
            over how many epochs to train
        early_stopping
            whether to use early stopping
        patience
            how many epochs without loss improvement before stopping
        fine_tune
            if False, only newly added batch layers are updated
        log_prefix
            string to be added to the Neptune log path
        run
            passed here if logging to Neptune should be carried out
        """

        if os.path.isfile(modelpath + '/checkpoint.pt'):
            print("A model already exists in the specified directory and will be overwritten.")

        if self.adata is None and adata is None:
            raise ValueError("No data provided. Please provide an AnnData object.")
        elif adata is not None:
            self._match_adata(adata)
            if batch_key is not None:
                self._update_batch(
                    self.adata,
                    batch_key,
                    add_layers=True
                )

        self.adata.obs[self.model + '_celltype'] = self.adata.obs[celltype] if celltype is not None else 'X'
        celltypes = list(set(self.adata.obs[self.model + '_celltype'].tolist()))
        celltype_map_dict = {celltype: i for i, celltype in enumerate(celltypes)}
        self.adata.obs['cell_type_idx'] = self.adata.obs[self.model + '_celltype'].map(celltype_map_dict)
        
        if len(self.new_batch_names) > 0 and not fine_tune:
            self._partial_freeze()

        # save train params
        self.train_hyperparams = {
            'train_size': train_size,
            'seed': seed,
            'learning_rate': learning_rate,
            'kl_weight': kl_weight,
            'grl_weight': grl_weight if self.batch_key is not None else None,
            'contrastive_weight': contrastive_weight if self.batch_key is not None else None,
            'mmd_weight': mmd_weight if self.batch_key is not None else None,
            'adv_weight': adv_weight if self.model == 'cobra' else None,
            'gp_weight': gp_weight if self.model == 'cobra' else None,
            'adv_step': adv_step if self.model == 'cobra' else None,
            'pos_weights': pos_weights,
            'batch_size': batch_size,
            'optimizer': str(optimizer).split("'")[1],
            'epochs': epochs,
            'early_stopping': early_stopping,
            'patience': patience
            }
        
        self._save_hyperparams(
            modelpath = modelpath,
            log_prefix = log_prefix,
            run = run
           )

        # generate dataloaders
        trainloader, valloader = self._create_dataloaders(
            self.adata, 
            train_size,
            seed,
            batch_size
            )


        # initialize optimizer
        optimizers = self._init_optimizers(
            optimizer = optimizer, 
            learning_rate = learning_rate,
            learning_rate_adv = learning_rate_adv,
            betas = betas,
            weight_decay = weight_decay
            )
        

        if early_stopping:
            early_stopper = EarlyStopper(patience=patience)
        
        with tqdm(total=len(trainloader)) as train_bar:
            with tqdm(total=len(valloader)) as val_bar:
                self.pbar = {'train': train_bar, 'val': val_bar}
                for epoch in range(1, epochs + 1):
                    self.epoch = epoch

                    self.pbar['train'].set_description(f"Epoch {epoch} [Training]")
                    self.pbar['train'].reset()

                    train_epoch_loss = self.training_step(
                        trainloader, 
                        kl_weight, 
                        grl_weight,
                        contrastive_weight,
                        mmd_weight,
                        adv_weight,
                        gp_weight,
                        adv_step,
                        pos_weights,
                        optimizers, 
                        log_prefix,
                        run,
                        )

                    self.pbar['val'].set_description(f"Epoch {epoch} [Validation]")
                    self.pbar['val'].reset()

                    val_epoch_loss = self.validation_step(
                        valloader, 
                        kl_weight, 
                        grl_weight,
                        contrastive_weight,
                        mmd_weight,
                        adv_weight,
                        log_prefix,
                        run
                        )
                    
                    # if early stopping is used, interrupt training if no improvement after certain amount of epochs
                    if early_stopping:
                        if early_stopper.early_stop(val_epoch_loss['val_loss_vae']):
                            print("Early stopping")
                            break
                    
                    # log losses to Neptune
                    if run is not None:
                        run[log_prefix + "/metrics/train/loss_vae"].log(train_epoch_loss['train_loss_vae'])
                        run[log_prefix + "/metrics/val/loss_vae"].log(val_epoch_loss['val_loss_vae'])
                        if self.model == 'cobra':
                            run["metrics/train/loss_adv"].log(train_epoch_loss['train_loss_adv'])
                            run["metrics/train/knn_purity"].log(train_epoch_loss['avg_purity'])
                            run["metrics/val/knn_purity"].log(val_epoch_loss['avg_purity'])

                    # save best model
                    if val_epoch_loss['val_loss_vae'] < self.val_loss_min:
                        self._save_model(
                            modelpath,
                            epoch,
                            optimizers,
                            val_epoch_loss['val_loss_vae']
                        )
                        self.val_loss_min = val_epoch_loss['val_loss_vae']
        
        self.trained = True


    def compute_loss(
            self,
            x_features: torch.tensor,
            x_batches: torch.tensor,
            x_labels: torch.tensor,
            out_dict: dict,
            kl_weight: float,
            grl_weight: float,
            contrastive_weight: float,
            mmd_weight: float,
            mode: Literal["train", "val"],
            log_prefix: str,
            run = None
    ):
        """
        Loss computation for VAE.

        Parameters
        ----------
        x_features
            torch.tensor of shape (minibatch, n_features)
        out_dict
            output of the forward pass
        kl_weight
            coefficient for weighting Kullback-Leibler loss
        contrastive_weight
            coefficient for weighting contrastive loss for batch integration
        grl_weight
            coefficient for weighting adversarial loss for batch integration
        mode
            mode for logging, on of 'train' or 'val'
        log_prefix
            string to be added to the Neptune log path
        run
            Neptune run if training is to be logged
        """
        if self.model in ['ontovae', 'cobra']:
            x_features = x_features[:,0:len(self.onto_features)]

        rec_loss = MSE_Loss()(
            x_features, 
            out_dict['out'], 
            mode=mode, 
            log_prefix=log_prefix,
            run=run
            )
        
        kl_divergence = KL_Divergence()(
            out_dict['mu'], 
            out_dict['log_var'],
            mode=mode, 
            log_prefix=log_prefix,
            run=run
            )
        
        if x_batches.sum() > 0 and grl_weight > 0:
            grl_out = self.grl(out_dict['z'])
            batch_logits = self.batch_classifier(grl_out)
            adversarial_loss = CrossEntropyLoss()(
                batch_logits, 
                x_batches,
                mode,
                log_prefix,
                run
                )
        else:
            adversarial_loss = 0

        if x_batches.sum() > 0 and contrastive_weight > 0:
            contrastive_loss = ContrastiveLoss()(
                out_dict['z'][x_batches==0,:],
                out_dict['z'][x_batches==1,:],
                x_labels[x_batches == 0],
                x_labels[x_batches == 1],
                mode,
                log_prefix,
                run
            )
        else:
            contrastive_loss = 0
        
        if x_batches.sum() > 0 and mmd_weight > 0:
            mmd_loss = MMDLoss()(
                out_dict['z'][x_batches==0,:],
                out_dict['z'][x_batches==1,:],
                mode,
                log_prefix,
                run
            )
        else: 
            mmd_loss = 0
            
        loss = rec_loss + kl_weight * kl_divergence + contrastive_weight * contrastive_loss + grl_weight * adversarial_loss + mmd_weight * mmd_loss
        return loss
    
    def _partial_freeze(self):
        """
        Helper function that freezes the model except for untrained batch layers.
        """
        for param in self.parameters():
            param.requires_grad = False
        for batch in self.new_batch_names:
            for param in self.encoder.encoder.batch_layers[batch].parameters():
                param.requires_grad = True
            for param in self.decoder.decoder.batch_layers[batch].parameters():
                param.requires_grad = True


    def _batch_to_onehot(self, adata: AnnData):
        """
        Extracts batch information from AnnData object and converts to one-hot encoded tensors.

        Parameters
        ----------
        adata: AnnData
            adata containing omics data from a given modality
        """
        if self.batch_key is not None:
            batches = torch.tensor(adata.obs[self.batch_key].astype(str).map(self.batch_idx).to_numpy())
            onehot_batches = F.one_hot(batches, num_classes = len(self.batch_idx))
        else:
            batches = torch.zeros(adata.shape[0])
            onehot_batches = torch.zeros(adata.shape[0],1)
        return batches, onehot_batches

    def _init_optimizers(
            self,
            optimizer: optim.Optimizer,
            learning_rate: float,
            learning_rate_adv: float,
            betas: tuple,
            weight_decay: float
    ) -> dict:
        """
        Parameters
        ----------
        optimizer
            which optimizer to use
        learning_rate
            learning rate
        """
        # VAE optimizer
        vae_params = [
            self.encoder.encoder.parameters(),
            self.decoder.decoder.parameters()
        ]
        if self.model == 'cobra':
            vae_params.append(self.covars_embeddings.parameters())

        vae = optimizer(
            chain(*vae_params),
            lr = learning_rate,
            betas = betas,
            weight_decay = weight_decay
            )
        
        # Batch classifier optimizer
        if hasattr(self, 'batch_classifier'):
            classifier = optimizer(
                self.batch_classifier.parameters(),
                lr = learning_rate,
                betas = betas,
                weight_decay = weight_decay
            )
        else:
            classifier = None

        # COBRA adversarial optimizer
        if self.model == 'cobra':
            adv = optimizer(
                self.covars_classifiers.parameters(), 
                lr = learning_rate_adv,
                betas = betas,
                weight_decay = weight_decay
                )
        else:
            adv = None

        return {
            'vae': vae,
            'classifier': classifier,
            'adv': adv
        }
    


    def _create_dataloaders(
            self,
            adata: AnnData, 
            train_size: float,
            seed: int,
            batch_size: int,
    ):
        """
        Parameters
        ----------
        adata
            adata containing the data
        train_size
            percentage of samples to be used for training
        seed
            seed for splitting adata into training and validations set
        batch_size
            minibatch size for dataloaders
        """
        # train-val split
        adatas = split_adata(
            adata, 
            train_size = train_size,
            seed = seed
            )
        train_batches, train_onehot_batches = self._batch_to_onehot(adatas['train_adata'])
        val_batches, val_onehot_batches = self._batch_to_onehot(adatas['val_adata'])

        trainloader = FastTensorDataLoader(
            torch.tensor(adatas['train_adata'].X.todense(), dtype=torch.float32),
            train_onehot_batches,
            train_batches,
            torch.tensor(adatas['train_adata'].obs['cell_type_idx'].to_numpy(), dtype=torch.float32),
            torch.tensor(np.array(adatas['train_adata'].obsm['cobra_covariates'], dtype='int64')) if self.model == 'cobra' else torch.zeros(adatas['train_adata'].shape[0]), #slot for COBRA covariates
            batch_size=batch_size, 
            shuffle=True
            )
        valloader = FastTensorDataLoader(
            torch.tensor(adatas['val_adata'].X.todense(), dtype=torch.float32),
            val_onehot_batches,
            val_batches,
            torch.tensor(adatas['val_adata'].obs['cell_type_idx'].to_numpy(), dtype=torch.float32),
            torch.tensor(np.array(adatas['val_adata'].obsm['cobra_covariates'], dtype='int64')) if self.model == 'cobra' else torch.zeros(adatas['val_adata'].shape[0]), #slot for COBRA covariates
            batch_size=batch_size, 
            shuffle=False
            )
        
        return trainloader, valloader
    

    def _unpack(self, minibatch, device):
        """
        Helper function to unpack a minibatch from a dataloader.
        """
        x_features = minibatch[0].to(device)
        x_onehot_batch = minibatch[1].to(device)
        x_batches = minibatch[2].to(device)
        x_labels = minibatch[3].to(device)
        x_cobra_covs = minibatch[4].to(device)
        x = torch.cat((x_features, x_onehot_batch), dim=1) if x_onehot_batch.sum() > 0 else x_features
           
        return x_features, x_batches, x_labels, x_cobra_covs, x
    

    def _save_hyperparams(
            self,
            modelpath: str,
            log_prefix: str,
            run=None
            ):
        """
        Stores hyperparams in model directory and also logs them to Neptune if a run is provided.

        Parameters
        ----------
        modelpath
            path to a folder where to store the params and the best model
        run
            passed here if logging to Neptune should be carried out
        """
        with open(modelpath + '/model_hyperparams.json', 'w') as fp:
            json.dump(self.hyperparams, fp, indent=4)

        with open(modelpath + '/train_hyperparams.json', 'w') as fp:
            json.dump(self.train_hyperparams, fp, indent=4)
        
        if run is not None:
            run[log_prefix + "/model_hyperparameters"] = self.hyperparams
            run[log_prefix + "/train_hyperparameters"] = self.train_hyperparams


    def _save_model(
            self,
            modelpath: str,
            epoch: int,
            optimizers: dict,
            val_epoch_loss: float,
    ):
        """
        Helper function to store checkpoint of the model.

        Parameters
        ----------
        modelpath
            path to a folder where to store the params and the best model
        epoch
            current epoch
        optimizer
            optimizer
        val_epoch_loss
            current best loss
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': {k: optimizers[k].state_dict() if optimizers[k] is not None else None for k in optimizers.keys()},
            'loss': val_epoch_loss,
            }, 
            modelpath + '/checkpoint.pt')
    


    @torch.no_grad()
    def _run_minibatches(
        self, 
        adata: AnnData, 
        retrieve: Literal['latent', 'act', 'rec'],
        lin_layer: bool = True,
        output_numpy: bool = True,
        ):
        """
        Runs batches of a dataloader through encoder or complete VAE and collects results.

        Parameters
        ----------
        adata
            AnnData object containing the data
        retrieve
            whether to retrieve latent space, pathway activities (only for OntoVAE) or reconstructed values
        lin_layer
            whether pathway activities should be retrieved from the linear layer (True) or from the output layer (False) of the block
        output_numpy
            whether to output a detached numpy array (True) or a torch.tensor (False)
        """
        self.eval()

        adata = self._match_adata(
            adata,
            self.input_features,
            return_adata = True
            )

        _, onehot_batches = self._batch_to_onehot(adata)
        dataloader = FastTensorDataLoader(
            torch.tensor(adata.X.todense(), dtype=torch.float32), 
            onehot_batches,
            batch_size=128, 
            shuffle=False
            )
        
        res = []
        for minibatch in dataloader:
            x_features = minibatch[0].to(self.device)
            x_batch = minibatch[1].to(self.device)
            x = torch.cat((x_features, x_batch), dim=1) if x_batch.sum() > 0 else x_features
            if retrieve == 'latent':
                result, _, _ = self.encode(x)
                if self.model == 'ontovae' and self.root_layer_latent:
                    result = self._average_neuronnum(result)
            elif retrieve == 'act':
                if self.model != 'ontovae':
                    raise ValueError("Pathway activities can only be retrieved from OntoVAE.")
                result = self._hook_activities(
                    x,
                    lin_layer
                )
                result = self._average_neuronnum(result) 
            else:
                out_dict = self.forward(x)
                result = out_dict['out']
            res.append(result)
        res = torch.vstack(res)
        if output_numpy:
            res = res.to('cpu').detach().numpy()

        return res
    
    @torch.no_grad()
    def to_latent(
        self, 
        adata: AnnData=None,
        batch_key: str = None,
        output_numpy: bool=True,
        return_adata: bool = True
        ):
        """
        Retrieves latent space embedding.

        Parameters
        ----------
        adata
            adata object containing the data
        output_numpy
            whether to output a detached numpy array (True) or a torch.tensor (False)
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
            retrieve='latent',
            output_numpy = output_numpy
            )
        
        if return_adata:
            adata = self._return_adata(
                adata,
                res,
                'latent_space'
            )

        return adata if return_adata else res


    @torch.no_grad()
    def get_reconstructed_values(
        self, 
        adata: AnnData=None,
        batch_key: str = None,
        output_numpy: bool = True,
        return_adata: bool = True
        ):
        """
        Retrieves reconstructed values from output layer.

        Parameters
        ----------
        adata
            adata object containing the data
        output_numpy
            whether to output a detached numpy array (True) or a torch.tensor (False)
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
            retrieve='rec',
            output_numpy = output_numpy
            )
        
        if return_adata:
            adata = self._return_adata(
                adata,
                res,
                'reconstruction'
            )

        return adata if return_adata else res


    @torch.no_grad()
    def perturbation(
        self, 
        adata: AnnData, 
        retrieve: Literal['latent', 'act', 'rec'],
        genes: list=[], 
        values: list=[],
        lin_layer: bool = True,
        output_numpy: bool = True
        ):
        """
        Retrieves reconstructed gene values after performing in silico perturbation.

        Parameters
        ----------
        adata
            AnnData object containing the data
        retrieve
            whether to retrieve latent space or reconstructed values
        genes
            a list of genes to perturb
        values
            list with new values, same length as genes
        output_numpy
            whether to output a detached numpy array (True) or a torch.tensor (False)
        """
        self.eval()

        if adata is None:
            pdata = self.adata.copy()
        else:
            pdata = adata.copy()

        # get indices of the genes in list
        gindices = [list(pdata.var_names).index(g) for g in genes]

        # replace their values
        for i in range(len(genes)):
            pdata.X[:,gindices[i]] = values[i]

        # get results
        res = self._run_minibatches(
            pdata, 
            retrieve=retrieve,
            lin_layer=lin_layer,
            output_numpy=output_numpy
            )
    
        return res

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
        genes = input_features if len(input_features) > 0 else list(adata.var_names)

        # get target genes
        adata_target = adata[:,adata.var_names.isin(genes)].copy()

        # look for missing genes and create dummy adata
        missing_genes = [g for g in genes if g not in adata_genes]
        counts = csr_matrix(np.zeros((adata.shape[0], len(missing_genes)), dtype=np.float32))
        ddata = AnnData(counts)
        ddata.obs_names = adata.obs_names
        ddata.var_names = missing_genes

        # concatenate everything
        ndata = ad.concat([adata_target, ddata], join="outer", axis=1)
        ndata = ndata[:,genes]

        # repopulate the adata object
        ndata.obs = adata.obs
        ndata.obsm = adata.obsm

        # redefine the input features
        self.input_features = ndata.var_names.tolist()

        if return_adata:
            return ndata
        else:
            self.adata = ndata

    
    @torch.no_grad()
    def _return_adata(
        self,
        adata,
        res,
        embedding_type
    ):
        """
        helper function to store model embedding results in adata
        """

        res = pd.DataFrame(res, index=adata.obs_names)
        res.columns = self.onto_annot['ID'].tolist()
        adata.obsm[embedding_type] = res
        return adata

