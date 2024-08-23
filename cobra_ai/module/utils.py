import numpy as np
import pandas as pd
import torch
import pkg_resources
from scipy.sparse import csr_matrix
import anndata as ad
from anndata import AnnData
from cobra_ai.module.ontobj import Ontobj
from typing import Optional
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc


"""AnnData handling"""

def setup_anndata_ontovae(adata: AnnData,
                  ontobj: Ontobj,
                top_thresh: Optional[int]=None,
                bottom_thresh: Optional[int]=None,
                batch_key: Optional[str] = None,
                labels_key: Optional[str] = None,
                categorical_covariate_keys: Optional[list[str]] = None,
                #class_key: Optional[str] = None,
                cobra_keys: Optional[list[str]] = None, 
                control_group: Optional[list[str]] = None,
                layer: Optional[str] = None):
    
    """
    Matches the dataset to the ontology and creates fields for OntoVAE and COBRA.

    Parameters
    ----------
        adata
            Scanpy single-cell AnnData object
        ontobj
            Ontobj containing a preprocessed ontology
        top_thresh
            top threshold for ontology trimming
        bottom_thresh
            bottom threshold for ontology trimming
        batch_key
            Observation to be used as batch
        labels_key
            Observation containing the labels
        categorical_covariate_keys
            Observations to use as covariate keys
        class_key
            Observation to use as class (only for OntoVAE + classifier)
        cobra_keys
            Observations to use for disentanglement of latent space (only for COBRA)
        control_group
            for combinatorial cobra_keys, specifies the control class
        layer
            layer of AnnData containing the data

    Returns
    -------
        ndata
            updated object if copy is True
    """

    if adata.is_view:
        raise ValueError(
            "Current adata object is a View. Please run `adata = adata.copy()`"
        )

    if top_thresh is not None and bottom_thresh is not None:
        if not str(top_thresh) + '_' + str(bottom_thresh) in ontobj.annot.keys():
            raise ValueError('Available trimming thresholds are: ' + ', '.join(list(ontobj.annot.keys())))
    else:
        top_thresh = list(ontobj.annot.keys())[0].split('_')[0]
        bottom_thresh = list(ontobj.annot.keys())[0].split('_')[1]

    
    if layer is not None:
         adata.X = adata.layers[layer].copy()
        
    if len(list(adata.layers.keys())) > 0:
        for k in list(adata.layers.keys()):
            del adata.layers[k]
    
    adata.varm = ""

    genes = ontobj.extract_genes(top_thresh=top_thresh, bottom_thresh=bottom_thresh)
    adata = adata[:,adata.var_names.isin(genes)].copy()

    # create dummy adata for features that were not present in adata
    out_genes = [g for g in genes if g not in adata.var_names.tolist()]
    counts = csr_matrix(np.zeros((adata.shape[0], len(out_genes)), dtype=np.float32))
    ddata = ad.AnnData(counts)
    ddata.obs_names = adata.obs_names
    ddata.var_names = out_genes

    # create OntoVAE matched adata and register ontology information

    ndata = ad.concat([adata, ddata], join="outer", axis=1)
    ndata = ndata[:,ndata.var_names.sort_values()]
 
    ndata.obs = adata.obs
    ndata.obsm = adata.obsm
    ndata.uns['_ontovae'] = {}
    ndata.uns['_ontovae']['thresholds'] = (top_thresh, bottom_thresh)
    ndata.uns['_ontovae']['annot'] = ontobj.extract_annot(top_thresh=top_thresh, bottom_thresh=bottom_thresh)
    ndata.uns['_ontovae']['genes'] = ontobj.extract_genes(top_thresh=top_thresh, bottom_thresh=bottom_thresh)
    ndata.uns['_ontovae']['masks'] = ontobj.extract_masks(top_thresh=top_thresh, bottom_thresh=bottom_thresh)

    if batch_key is not None:
        ndata.obs['_ontovae_batch'] = pd.factorize(ndata.obs.loc[:,batch_key])[0]
    else:
        ndata.obs['_ontovae_batch'] = 0
    
    if labels_key is not None:
        ndata.obs['_ontovae_labels'] = pd.factorize(ndata.obs.loc[:,labels_key])[0]
    else:
        ndata.obs['_ontovae_labels'] = 0
    
    if categorical_covariate_keys is not None:
        ndata.obs['_ontovae_categorical_covs'] = ndata.obs.loc[:,categorical_covariate_keys].apply(lambda x: pd.factorize(x)[0])      

    #if class_key is not None:
    #    ndata.obs['_ontovae_class'] = pd.factorize(ndata.obs.loc[:,class_key])[0]
    
    if cobra_keys is not None:
        cov_dict = {}
        comb_cov_dict = {} 
        cov_type = {}
        mappings = []
        i = 0
        for k in cobra_keys:
            cov_type[k] = 'distinct'
            classes = list(ndata.obs.loc[:,k].unique())
            if np.any(['+' in s for s in classes]):
                if not control_group:
                    raise ValueError('Please provide the control group!')
                ctrl_group = control_group[i]
                cov_type[k] = 'combinatorial'
                classes = [s.split('+') for s in classes]
                classes = list(set([j for i in classes for j in i]))
                classes.insert(0, classes.pop(classes.index(ctrl_group)))
                i = i+1
            else:
                classes = ['<DUM>'] + classes
            mapping = dict(zip(classes, np.arange(len(classes))))
            mapping = {key: int(value) for key, value in mapping.items()}
            if cov_type[k] == 'combinatorial':
                combos = ndata.obs.loc[:,k].unique()
                cov_dict[k] = {}
                cov_dict[k]['embedding'] = mapping
                cov_dict[k]['classifier'] = dict(zip(combos, range(len(combos))))
                cov_dict[k]['mapping'] = {}
                for cs in combos:
                     cov_dict[k]['mapping'][cov_dict[k]['classifier'][cs]] = [cov_dict[k]['embedding'][c] for c in cs.split('+')]
                comb_values = [len(v) for v in cov_dict[k]['mapping'].values()]
                max_comb = np.max(comb_values)
                to_pad = max_comb - comb_values
                new_values = [list(cov_dict[k]['mapping'].values())[i] + [0] * to_pad[i] for i in range(len(list(cov_dict[k]['mapping'].values())))]
                cov_dict[k]['mapping'] = dict(zip(list(cov_dict[k]['mapping'].keys()), new_values))
                mappings.append(ndata.obs.loc[:,k].map(cov_dict[k]['classifier']))
            else:
                cov_dict[k] = mapping
                mappings.append(ndata.obs.loc[:,k].map(cov_dict[k]))
        ndata.uns['cov_dict'] = cov_dict
        ndata.uns['cov_type'] = cov_type
        ndata.obsm['_cobra_categorical_covs'] = pd.concat(mappings, axis=1)

    return ndata

def setup_anndata_vanillavae(adata: AnnData,
                batch_key: Optional[str] = None,
                labels_key: Optional[str] = None,
                categorical_covariate_keys: Optional[list[str]] = None,
                cobra_keys: Optional[list[str]] = None, 
                layer: Optional[str] = None):
    
    """
    Sets up anndata for the Vanilla VAE or adversarial VAE.

    Parameters
    ----------
        adata
            Scanpy single-cell AnnData object
        batch_key
            Observation to be used as batch
        labels_key
            Observation containing the labels
        categorical_covariate_keys
            Observations to use as covariate keys
        cobra_keys
            Observations to use for disentanglement of latent space (only for advVAE)
        layer
            layer of AnnData containing the data

    Returns
    -------
        ndata
            updated object if copy is True
    """

    if adata.is_view:
        raise ValueError(
            "Current adata object is a View. Please run `adata = adata.copy()`"
        )
    
    if layer is not None:
         adata.X = adata.layers[layer].copy()
        
    if len(list(adata.layers.keys())) > 0:
        for k in list(adata.layers.keys()):
            del adata.layers[k]

    if batch_key is not None:
        adata.obs['_ontovae_batch'] = pd.factorize(adata.obs.loc[:,batch_key])[0]
    else:
        adata.obs['_ontovae_batch'] = 0
    
    if labels_key is not None:
        adata.obs['_ontovae_labels'] = pd.factorize(adata.obs.loc[:,labels_key])[0]
    else:
        adata.obs['_ontovae_labels'] = 0

    if categorical_covariate_keys is not None:
        adata.obs['_ontovae_categorical_covs'] = adata.obs.loc[:,categorical_covariate_keys].apply(lambda x: pd.factorize(x)[0])      
    
    if cobra_keys is not None:
         adata.obsm['_cobra_categorical_covs'] = adata.obs.loc[:,cobra_keys].apply(lambda x: pd.factorize(x)[0])
    
    return adata



def split_adata(adata: AnnData, train_size: float = 0.9, seed: int = 42):
    """
    Returns train_adata and val/test_adata
    """
    indices = np.random.RandomState(seed=seed).permutation(adata.shape[0])
    X_train_ind = indices[:round(len(indices)*train_size)]
    X_val_ind = indices[round(len(indices)*train_size):]
    train_adata = adata[X_train_ind,:].copy()
    val_adata = adata[X_val_ind,:].copy() 
    return train_adata, val_adata




"""Additional helper functions"""

def data_path():
    """
    Function to access internal package data
    """
    path = pkg_resources.resource_filename(__name__, 'data/')
    return path



class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches
    

    
"""Plotting"""

def plot_scatter(adata: AnnData, color_by: list, act, term1: str, term2: str):
        """ 
        Creates a scatterplot of two pathway activities.

        Parameters
        ----------
        adata
            AnnData object that was processed with setup_anndata_ontovae
        color_by
            list of coavariates by which to color the samples (has to be present in adata)
        act
            numpy array containing pathway activities
        term1
            ontology term on x-axis
        term2
            ontology term on y-axis
        """

        if '_ontovae' not in adata.uns.keys():
            raise ValueError('Please run sconto_vae.module.utils.setup_anndata first.')
        
        for c in color_by:
            if not c in adata.obs:
                raise ValueError('Please set color_by to a covariate present in adata.obs.')

        # extract ontology annot and get term indices
        onto_annot = adata.uns['_ontovae']['annot']
        onto_annot.index = onto_annot.index.astype(int)
        ind1 = onto_annot[onto_annot.Name == term1].index.to_numpy()[0]
        ind2 = onto_annot[onto_annot.Name == term2].index.to_numpy()[0]

        fig, ax = plt.subplots(1,len(color_by), figsize=(len(color_by)*10,10))

        # make scatterplot
        for c in range(len(color_by)):

            # create color dict
            covar_categs = adata.obs[color_by[c]].unique().tolist()
            palette = sns.color_palette(cc.glasbey, n_colors=len(covar_categs))
            color_dict = dict(zip(covar_categs, palette))

            # make scatter plot
            sns.scatterplot(x=act[:,ind1],
                            y=act[:,ind2], 
                            hue=adata.obs[color_by[c]],
                            palette=color_dict,
                            legend='full',
                            s=15,
                            rasterized=True,
                            ax=ax.flatten()[c])
            ax.flatten()[c].set_xlabel(term1)
            ax.flatten()[c].set_ylabel(term2)

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        return fig, ax


"""Classification based on pathway activities"""

def calculate_auc(adata: AnnData, X: np.array, y: np.array, n_splits: int=10):
    """
    Performs sample classification on pathway activities using k-fold cross validation 
    and outputs the ontology annotation with an additional column 'auc' 
    that gives the median classification AUC for each term using a Naive Bayes Classifier

    Parameters
    ----------
    adata
        anndata that was used to calculate the pathway activities
    X
        2D numpy array of pathway activities
    y
        1D numpy array of binary class labels
    n_splits
        how many splits to use for cross-validation
    """

    def cross_val_auc(X, y, n_splits=n_splits):
        # initialize aucs list
        aucs = []

        # cross-val
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

        # iterate over folds
        for train_ind, test_ind in skf.split(X,y,y):
            X_train, X_test, y_train, y_test = X[train_ind], X[test_ind], y[train_ind], y[test_ind]

            # train Naive Bayes
            gnb = GaussianNB()
            gnb.fit(X_train, y_train)

            # make predictions
            y_pred = gnb.predict(X_test)

            # calculate AUC
            fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
            auc = metrics.auc(fpr, tpr)

            # append to list
            aucs.append(auc)

        return np.nanmedian(np.array(auc))
    
    auc = [cross_val_auc(X[:,i].reshape(-1,1), y) for i in range(X.shape[1])]
    annot = adata.uns['_ontovae']['annot'].copy()
    annot['auc'] = auc
    return annot



"""Differential testing between two groups"""

def unpaired_wilcox_test(adata: AnnData, group1, group2):
    """
    Performs unpaired Wilcoxon test between two groups for all pathways.

    Parameters
    ----------
    adata
            AnnData object that was processed with setup_anndata_ontovae
    control
            numpy 2D array of pathway activities of group1
    perturbed
            numpy 2D array of pathway activities of group2
    """

    if '_ontovae' not in adata.uns.keys():
            raise ValueError('Please run sconto_vae.module.utils.setup_anndata first.')
        
    onto_annot = adata.uns['_ontovae']['annot']

    wilcox = [stats.ranksums(group1[:,i], group2[:,i]) for i in range(onto_annot.shape[0])]
    # stat > 0 : higher in group1
    stat = np.array([i[0] for i in wilcox])
    pvals = np.array([i[1] for i in wilcox])
    qvals = fdrcorrection(np.array(pvals))

    res = pd.DataFrame({'stat': stat,
                        'pval': pvals,
                        'qval': qvals[1]})
    res = pd.concat((onto_annot, res))
    
    res = res.sort_values('pval').reset_index(drop=True)
    return(res)


"""Differential testing for pathway activities after perturbation"""

def  wilcox_test(adata: AnnData, control, perturbed, direction='up', option='terms'):
        """ 
        Performs paired Wilcoxon test between activities and perturbed activities.

        Parameters
        ----------
        adata
            AnnData object that was processed with setup_anndata_ontovae
        control
            numpy 2D array of pathway activities 
        perturbed
            numpy 2D array of perturbed pathway activities
        direction
            up: higher in perturbed
            down: lower in perturbed
        option
            'terms' or 'genes'
        """

        if '_ontovae' not in adata.uns.keys():
            raise ValueError('Please run sconto_vae.module.utils.setup_anndata first.')
        
        onto_annot = adata.uns['_ontovae']['annot']
        onto_genes = adata.uns['_ontovae']['genes']

        # perform paired wilcoxon test over all terms
        alternative = 'greater' if direction == 'up' else 'less'
        wilcox = [stats.wilcoxon(perturbed[:,i], control[:,i], zero_method='zsplit', alternative=alternative) for i in range(control.shape[1])]
        stat = np.array([i[0] for i in wilcox])
        pvals = np.array([i[1] for i in wilcox])
        qvals = fdrcorrection(np.array(pvals))

        if option == 'terms':
            res = pd.DataFrame({'stat': stat,
                                'pval' : pvals,
                                'qval': qvals[1]})
            res = pd.concat((onto_annot.reset_index(drop=True), res), axis=1)
        
        else:
            res = pd.DataFrame({'gene': onto_genes,
                                'stat': stat,
                                'pval' : pvals,
                                'qval': qvals[1]})

        res = res.sort_values('pval').reset_index(drop=True)
        return(res)


# class for Early Stopping
# adapted from: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch

class EarlyStopper:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# adapted from torch.optim.swa_utils

@torch.no_grad()
def update_bn(loader, model):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.

    Example:
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None

    for input in loader:
        if isinstance(input, (list, tuple)):
            data = torch.tensor(input[0].todense(), dtype=torch.float32).to(model.module.device)
            cat_list = torch.split(input[1].T.to(model.module.device), 1)
            cov_list = torch.split(input[2].T.to(model.module.device), 1)
            model.module.forward(data, cat_list, cov_list, mixup_lambda=1)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)