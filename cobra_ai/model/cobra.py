#!/usr/bin/env python3

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from typing import Iterable, Literal

from anndata import AnnData

from cobra_ai.module.modules import Classifier
from cobra_ai.model.onto_vae import OntoVAE
from cobra_ai.module.ontobj import Ontobj


from cobra_ai.module.utils.metrics import knn_purity
from cobra_ai.module.utils.utils import FastTensorDataLoader
from cobra_ai.module.utils.losses import CrossEntropyLoss



class COBRA(OntoVAE):
    """
    This class extends OntoVAE with a CPA-like approach of disentangling covariate effects in the latent space in a linear fashion.
   
    We can use distinct covariates (e.g. stimulated versus control)
    or combinatorial covariates where the combos should be specified by a plus sign (e.g. Ctrl, IfnA, IfnB, IfnA+IfnB)
    """


    def __init__(self):
        super().__init__()
    
    def construct(
            self, 
            ontobj: Ontobj = None,
            adata: AnnData = None,
            cobra_keys: Iterable[str] = None,
            control_groups: Iterable[str] = None,
            hidden_dims_classifier: list = [64],
            normalisation: Literal["batch", "layer"] = "batch",
            activation_fn: nn.Module = nn.ReLU,
            bias: bool = True,
            dropout_rate: float = 0.2,
            cov_dict: dict = {},
            **kwargs
            ):
        """
        Function to construct the COBRA model based on OntoVAE.

        Parameters
        ----------
        cobra_keys
            columns in adata.obs specifying the covariates that should be disentangled in the latent space
        control_groups
            entries in the cobra_keys columns that serve as reference/control for this covariate
        cov_dict
            dictionary containing all COBRA covariate mappings and info (used internally when loading a trained model)
        """

        # build OntoVAE model
        super().construct(
            ontobj,
            adata,
            hidden_dims_classifier=hidden_dims_classifier,
            normalisation=normalisation,
            activation_fn=activation_fn,
            bias=bias,
            dropout_rate=dropout_rate,
            **kwargs
            )

        if adata is not None and cobra_keys is None:
            raise ValueError("You have to specify which covariates you want to disentangle!")
        else:
            self.cobra_keys = cobra_keys

        # create index information to match samples to the embedding layer and to the classifier
        
        if len(cov_dict) == 0:
            self._parse_covariates(
                cobra_keys = cobra_keys,
                control_groups = control_groups
            )
        else:
            self.cov_dict = cov_dict
            self.cobra_keys = list(cov_dict.keys())

        self.hyperparams.update({'cov_dict': self.cov_dict})
        
        # add cobra covariates to adata
        if self.adata is not None:
            self.adata = self._add_cobra_covs(self.adata)

        # embedding of covars
        self.covars_embeddings = nn.ModuleDict(
            {
                key: torch.nn.Embedding(len(self.cov_dict[key]['embedding']), self.latent_dim, padding_idx=0 if self.cov_dict[key]['control_group'] is not None else None)
                for key in self.cobra_keys
            }
        )
        
        # covars classifiers
        self.covars_classifiers = nn.ModuleDict(
            {
                key: Classifier(
                    n_features = self.latent_dim,
                    hidden_dims = hidden_dims_classifier,
                    n_classes = len(self.cov_dict[key]['classifier']),
                    normalisation = normalisation,
                    activation_fn = activation_fn,
                    bias = bias,
                    dropout_rate = dropout_rate
                    )
                for key in self.cobra_keys
            }
        )

        self.to(self.device)
        self.model = 'cobra'

    def _check_adata(self, adata: AnnData):
        """
        Checks if adata containes previously unseen categories
        """
        
        # Check if adata contains all neccessary covariates
        cobra_keys = list(self.cobra_keys)

        if np.any([c not in adata.obs.columns for c in cobra_keys]):
            raise ValueError('Dataset does not contain all covariates.')
        
        # Check if configure function needs to be run
        configure=False
        for k in cobra_keys:
            if self.cov_dict[k]['cov_type'] == 'combinatorial':
                new_cond = [c for c in adata.obs.loc[:,k].unique() if c not in self.cov_dict[k]['classifier'].keys()]
                if len(new_cond) > 0:
                    configure=True
        
        return configure

    def _configure_adata(self, adata: AnnData):
        """
        Configures anndata to match the existing covariate mappings.
        
        """
        cobra_keys = list(self.cobra_keys)

        mappings = []
        for k in cobra_keys:
            if self.cov_type[k] == 'distinct':
                mappings.append(adata.obs.loc[:,k].map(self.cov_dict[k]))
            else:
                new_cond = [c for c in adata.obs.loc[:,k].unique() if c not in self.cov_dict[k]['classifier'].keys()]
                if len(new_cond) > 0:
                    element_num = len(self.cov_dict[k]['classifier'])
                    self.cov_dict[k]['classifier'].update({new_cond[i]: element_num + i for i in range(len(new_cond))})
                    self.cov_dict[k]['mapping'] = {}
                    combos = list(self.cov_dict[k]['classifier'].keys())
                    for cs in combos:
                        self.cov_dict[k]['mapping'][self.cov_dict[k]['classifier'][cs]] = [self.cov_dict[k]['embedding'][c] for c in cs.split('+')]
                    comb_values = [len(v) for v in self.cov_dict[k]['mapping'].values()]
                    max_comb = np.max(comb_values)
                    to_pad = max_comb - comb_values
                    new_values = [list(self.cov_dict[k]['mapping'].values())[i] + [0] * to_pad[i] for i in np.arange(len(list(self.cov_dict[k]['mapping'].values())))]
                    self.cov_dict[k]['mapping'] = dict(zip(list(self.cov_dict[k]['mapping'].keys()), new_values))
                    adata.uns['cov_dict'][k] = self.cov_dict[k]
                mappings.append(adata.obs.loc[:,k].map(self.cov_dict[k]['classifier']))
        
        adata.obsm['_cobra_categorical_covs'] = pd.concat(mappings, axis=1)
            

    def _get_embeddings(self, x: torch.tensor, cobra_keys: Iterable[torch.tensor]):
        """
        Generates latent space embedding.

        Parameters
        ----------
        x
            torch.tensor of shape (minibatch, in_features)
        cobra_keys
            Iterable of torch.tensors containing the covs to delineate in 
            latent space 
        """
        # encoding
        z_basal, mu, log_var = self.encode(x)

        # covariate encoding
        covars_embeddings = {}
        for i, key in enumerate(self.covars_embeddings.keys()):
            covs = cobra_keys[:,i]
            if self.cov_dict[key]['cov_type'] == 'distinct':
                x = self.covars_embeddings[key](covs)
            else:
                # for combinatorial covariates, we sum up the embeddings of the different categories the minibatch of samples belong to
                mapping = self.cov_dict[key]['mapping']
                num = len(mapping[0])
                x = torch.sum(torch.stack([self.covars_embeddings[key](ten) for ten in [torch.LongTensor([mapping[int(e)][i] for e in covs]).to(self.device) for i in np.arange(num)]]), dim=0)
            covars_embeddings[key] = x

        # create different z's
        z_cov = {}
        z_total = z_basal.clone()
        for key in covars_embeddings.keys():
            z_cov['z_' + key] = (z_basal + covars_embeddings[key])
            z_total += covars_embeddings[key]

        z_dict = dict(z_basal=z_basal)
        z_dict.update(z_cov, z_total=z_total)

        return z_dict, mu, log_var
  
    def forward(self, x: torch.tensor, cobra_keys: Iterable[torch.tensor]):
        """
        Forward computation on minibatch of samples.
        
        Parameters
        ----------
        x
            torch.tensor of shape (minibatch, in_features)
        cat_list
            Iterable of torch.tensors containing the category memberships
            shape of each tensor is (minibatch, 1)
        cobra_keys
            Iterable of torch.tensors containing the covs to delineate in 
            latent space 
        """
        _, x_batch = torch.split(x, [len(self.input_features), x.shape[1]-len(self.input_features)], dim=1)

        # inference
        zdict, mu, log_var = self._get_embeddings(x, cobra_keys)

        # decoding
        out = self.decode(torch.cat((zdict['z_total'], x_batch), dim=1))
            
        return {
            'z_embeddings': zdict, 
            'z': zdict['z_total'],
            'mu': mu, 
            'log_var': log_var, 
            'out': out
            }


    def adv_forward(self, z: torch.tensor, compute_penalty=False):
        """
        Forward computation on minibatch of samples for z_basal.
        
        Parameters
        ----------
        z
            torch.tensor of shape (minibatch, in_features)
        """
        if compute_penalty:
            z = z.requires_grad_(True)

        # covariate classifiers on z_basal
        covars_pred = {}
        for key in self.covars_classifiers.keys():
            covar_pred = self.covars_classifiers[key](z)
            covars_pred[key] = covar_pred
        
        if compute_penalty:
            penalty = 0.0
            # Penalty losses
            for key in self.covars_classifiers.keys():
                grad = torch.autograd.grad(
                        covars_pred[key].sum(),
                        z,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True
                    )[0]
                pen = ((grad.norm(2, dim=1) - 1) ** 2).mean()
                penalty += pen
            covars_pred['penalty'] = penalty
        
        return covars_pred


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
        adv_weight
            coefficient for weighting classifier
        gp_weight
            coefficient for weighting gradient penalty
        adv_step:
            after how many minibatches the discriminators should be updated
        optimizers
            dictionary containing the optimizers
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
        for minibatch in dataloader:

            # move minibatch to device
            x_features, x_batches, x_labels, x_cobra_covs, x = self._unpack(minibatch, self.device)

            # VAE optimizer
            optimizers['vae'].zero_grad()

            # forward step generator
            out_dict = self.forward(x, x_cobra_covs)
            z_basal = out_dict['z_embeddings']["z_basal"]
            covars_pred = self.adv_forward(z_basal)

            # compute the VAE loss
            vae_loss = self.compute_loss(
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

            adv_loss = 0.0
            for i, vals in enumerate(x_cobra_covs.T):
                cov = list(self.cobra_keys)[i]
                cov_loss = CrossEntropyLoss()(
                        covars_pred[cov],
                        vals,
                        mode='train',
                        log_prefix=cov,
                        run = run
                    )
                adv_loss += cov_loss
            loss = vae_loss - adv_weight * adv_loss
            running_loss_vae += loss.item()

            # backward propagation
            loss.backward()

            # zero out gradients from non-existent connections
            for i in range(self.start_point, len(self.decoder.decoder.fc_layers)):
                    self.decoder.decoder.fc_layers[i][0].weight.grad = torch.mul(self.decoder.decoder.fc_layers[i][0].weight.grad, self.decoder.masks[i-self.start_point])
                    self.decoder.reconstruction[0].weight.data = torch.mul(self.decoder.reconstruction[0].weight.data, self.decoder.masks[-1])
            
            # perform optimizer step
            optimizers['vae'].step()

            # make weights in Onto module positive
            if pos_weights:
                for i in range(self.start_point, len(self.decoder.decoder.fc_layers)):
                    self.decoder.decoder.fc_layers[i][0].weight.data = self.decoder.decoder.fc_layers[i][0].weight.data.clamp(0)
                    self.decoder.reconstruction[0].weight.data = self.decoder.reconstruction[0].weight.data.clamp(0)
            
            # adversarial training
            if i % adv_step == 0:
                # adversarial optimizer
                optimizers['adv'].zero_grad()

                # forward step discriminator
                covars_pred = self.adv_forward(z_basal.detach(), compute_penalty=True)
                adv_loss = 0.0
                for i, vals in enumerate(x_cobra_covs.T):
                    cov = list(self.cobra_keys)[i]
                    cov_loss = CrossEntropyLoss()(
                        covars_pred[cov],
                        vals,
                        mode='train',
                        log_prefix=cov,
                        run = run
                    )
                    adv_loss += cov_loss

                loss = adv_loss + gp_weight * covars_pred['penalty']
                running_loss_adv += loss

                # backward propagation
                loss.backward()
                optimizers['adv'].step()
            
            # compute KNN purity
            cov_purity = []
            for i, vals in enumerate(x_cobra_covs.T):
                cov = list(self.cobra_keys)[i]
                cov_purity.append(knn_purity(z_basal.to('cpu').detach().numpy(), vals.long().squeeze().to('cpu').detach().numpy()))
                cov_purity.append(-knn_purity(out_dict['z_embeddings']['z_' + cov].to('cpu').detach().numpy(), vals.long().squeeze().to('cpu').detach().numpy()))
            purity += np.sum(cov_purity)

            self.pbar['train'].update(1)

        # compute avg training loss
        train_loss_vae = running_loss_vae/len(dataloader)
        train_loss_adv = running_loss_adv/len(dataloader)

        # compute average purity
        avg_purity = purity/len(dataloader)

        return {
            'train_loss_vae': train_loss_vae, 
            'train_loss_adv': train_loss_adv, 
            'avg_purity': avg_purity
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
        adv_weight 
            coefficient for weighting classifier
        run
            Neptune run if training is to be logged
        """
        # set to eval mode
        self.eval()

        # initialize running losses
        running_loss_vae = 0.0

        # init purity
        purity = 0.0

        # iterate over dataloader for validation
        for minibatch in dataloader:

            # unpack minibatch
            x_features, x_batches, x_labels, x_cobra_covs, x = self._unpack(minibatch, self.device)
            if x_features.shape[0] == 1:
                break

            # forward step generator
            out_dict = self.forward(x, x_cobra_covs)
            z_basal = out_dict['z_embeddings']["z_basal"]
            covars_pred = self.adv_forward(z_basal)

            # compute the VAE loss
            vae_loss = self.compute_loss(
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
            
            adv_loss = 0.0
            for i, vals in enumerate(x_cobra_covs.T):
                cov = list(self.cobra_keys)[i]
                cov_loss = CrossEntropyLoss()(
                    covars_pred[cov],
                    vals.long().squeeze(),
                    mode='train',
                    log_prefix=cov,
                    run = run
                )
                adv_loss += cov_loss
                
            loss = vae_loss - adv_weight * adv_loss
            running_loss_vae += loss.item()

            # compute KNN purity
            cov_purity = []
            for i, vals in enumerate(x_cobra_covs.T):
                cov = list(self.cobra_keys)[i]
                cov_purity.append(knn_purity(z_basal.to('cpu').detach().numpy(), vals.long().squeeze().to('cpu').detach().numpy()))
                cov_purity.append(-knn_purity(out_dict['z_embeddings']['z_' + cov].to('cpu').detach().numpy(), vals.long().squeeze().to('cpu').detach().numpy()))
            purity += np.sum(cov_purity)

            self.pbar['val'].update(1)

        # compute avg training loss
        val_loss_vae = running_loss_vae/len(dataloader)

        # compute average purity
        avg_purity = purity/len(dataloader)

        return {
            'val_loss_vae': val_loss_vae, 
            'avg_purity': avg_purity
        }              

    @torch.no_grad()
    def _pass_data(
        self, 
        x, 
        cobra_keys,
        retrieve: Literal['act', 'rec'],
        lin_layer=True
        ):
        """
        Passes data through the model.

        Parameters
        ----------
        x
            torch.tensor of shape (minibatch, in_features)
        cobra_keys
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

        _, x_batch = torch.split(x, [len(self.input_features), x.shape[1]-len(self.input_features)], dim=1)

        # get latent space embedding dict
        zdict, _, _ = self._get_embeddings(x, cobra_keys)
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
            out = self.decode(torch.cat((zdict[z_key], x_batch), dim=1))

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
                act_dict[z_key] = out

        return act_dict
        
    
    @torch.no_grad()
    def _run_minibatches(
        self, 
        adata: AnnData, 
        retrieve: Literal['latent', 'act', 'rec'], 
        lin_layer: bool=True,
        output_numpy: bool=True
        ):
        """
        Runs batches of a dataloader through encoder or complete VAE and collects results.
        """
        self.eval()

        adata = self._match_adata(
            adata,
            self.input_features,
            return_adata = True
            )

        if not 'cobra_covariates' in adata.obsm.keys():
            adata = self._add_cobra_covs(adata)

        configure = self._check_adata(adata)
        if configure:
            self._configure_adata(adata)

        _, onehot_batches = self._batch_to_onehot(adata)
        dataloader = FastTensorDataLoader(
            torch.tensor(adata.X.todense(), dtype=torch.float32), 
            onehot_batches,
            torch.tensor(np.array(adata.obsm['cobra_covariates'], dtype='int64')) if self.model == 'cobra' else torch.zeros(adata.shape[0]), #slot for COBRA covariates
            batch_size=128, 
            shuffle=False
            )
        
        res = []
        for minibatch in dataloader:
            x_features = minibatch[0].to(self.device)
            x_batch = minibatch[1].to(self.device)
            x_cobra_covs = minibatch[2].to(self.device)
            x = torch.cat((x_features, x_batch), dim=1) if x_batch.sum() > 0 else x_features
            if retrieve == 'latent':
                result, _, _ = self._get_embeddings(x, x_cobra_covs)
                if self.root_layer_latent:
                    result_avg = {k: self._average_neuronnum(v.to('cpu').detach().numpy()) for k, v in result.items()}
                else:
                    result_avg = {k: v.to('cpu').detach().numpy() for k, v in result.items()}
                res.append(result_avg)
            else:
                result = self._pass_data(x, x_cobra_covs, retrieve, lin_layer)
                if retrieve == 'act':
                    result = {k: self._average_neuronnum(v) for k, v in result.items()}
                res.append(result)

        res_out = {}
        key_list = list(res[0].keys())
        for key in key_list:
            res_key = torch.vstack([r[key] for r in res])
            res_out[key] = res_key

        if output_numpy:
            res_out =  {k: v.to('cpu').detach().numpy() for k, v in res_out.items()}

        return res_out
    

    @torch.no_grad()
    def get_inference(self, adata: AnnData, embedding: np.array, retrieve: Literal['act', 'rec'], lin_layer=True): 
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

    

    def _parse_covariates(
        self,
        cobra_keys: list,
        control_groups: list
    ):
        """
        Helper function to set up indices for classifier and embedding layers
        """
        cov_dict = {}

        for i, k in enumerate(cobra_keys):
            cov_dict[k] = {}

            # first, define which samples go to which classifier index
            cov_dict[k]['cov_type'] = 'distinct'
            classes  = list(self.adata.obs.loc[:,k].unique())
            cov_dict[k]['control_group'] = control_groups[i]
            if control_groups[i] is not None:
                classes.insert(0, classes.pop(classes.index(control_groups[i])))
            cov_dict[k]['classifier'] = dict(zip(classes, range(len(classes))))
                    
            # then define the embedding layer mapping taking into account the ctrl group
            if np.any(['++' in s for s in classes]):
                cov_dict[k]['cov_type'] = 'combinatorial'
                classes = [s.split('++') for s in classes]
                classes = list(set([j for i in classes for j in i]))
            mapping = dict(zip(classes, range(len(classes))))
            cov_dict[k]['embedding'] = mapping 

            # then create sample mapping between embedding layer and classifier
            classes  = list(self.adata.obs.loc[:,k].unique())
            cov_dict[k]['c2e_mapping'] = {}
            for cs in classes:
                cov_dict[k]['c2e_mapping'][cov_dict[k]['classifier'][cs]] = [cov_dict[k]['embedding'][c] for c in cs.split('++')]
            # if covariate is combinatorial, padding might have to be applied to have same value length
            if cov_dict[k]['cov_type'] == 'combinatorial':
                comb_values = [len(v) for v in cov_dict[k]['c2e_mapping'].values()]
                max_comb = np.max(comb_values)
                to_pad = max_comb - comb_values
                new_values = [list(cov_dict[k]['c2e_mapping'].values())[i] + [0] * to_pad[i] for i in range(len(list(cov_dict[k]['c2e_mapping'].values())))]
                cov_dict[k]['c2e_mapping'] = dict(zip(list(cov_dict[k]['c2e_mapping'].keys()), new_values))

            self.cov_dict = cov_dict

    def _add_cobra_covs(
        self,
        adata
    ):
        """
        helper function to add cobra covariates to adata
        """
        mappings = []
        for k in self.cobra_keys:
            mappings.append(adata.obs.loc[:,k].map(self.cov_dict[k]['classifier']))
        mappings = pd.concat(mappings, axis=1)
        adata.obsm['cobra_covariates'] = mappings
        return adata


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

        for view in list(res.keys()):
            res_view = pd.DataFrame(res[view], index=adata.obs_names)
            res_view.columns = self.onto_annot['ID'].tolist()
            adata.obsm[view + '_' + embedding_type] = res_view
        return adata