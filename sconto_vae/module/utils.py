import numpy as np
import pandas as pd
import torch
import pkg_resources
from scipy.sparse import csr_matrix
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, CategoricalJointObsField, LayerField
import anndata as ad
from anndata import AnnData
from .ontobj import Ontobj
from typing import Optional, Union
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc


"""AnnData handling"""

def setup_anndata_ontovae(adata: AnnData,
                  ontobj: Ontobj,
                top_thresh: int=1000,
                bottom_thresh: int=30,
                batch_key: Optional[str] = None,
                labels_key: Optional[str] = None,
                categorical_covariate_keys: Optional[list[str]] = None,
                layer: Optional[str] = None,
                is_count_data: bool = False):
    
    """
    Matches the dataset to the ontology and creates OntoVAE fields.
    Also creates scvi fields for batch and covariates.

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
            Observation to use as covariate keys
        layer
            layer of AnnData containing the data
        is_count_data
            if data is raw counts or log-normalized

    Returns
    -------
        ndata
            updated object if copy is True
    """

    if adata.is_view:
        raise ValueError(
            "Current adata object is a View. Please run `adata = adata.copy()`"
        )

    if not str(top_thresh) + '_' + str(bottom_thresh) in ontobj.genes.keys():
            raise ValueError('Available trimming thresholds are: ' + ', '.join(list(ontobj.genes.keys())))
    
    genes = ontobj.extract_genes(top_thresh=top_thresh, bottom_thresh=bottom_thresh)
    adata = adata[:,adata.var_names.isin(genes)]

    # create dummy adata for features that were not present in adata
    out_genes = [g for g in genes if g not in adata.var_names.tolist()]
    counts = csr_matrix(np.zeros((adata.shape[0], len(out_genes)), dtype=np.float32))
    ddata = ad.AnnData(counts)
    ddata.obs_names = adata.obs_names
    ddata.var_names = out_genes

    # create OntoVAE matched adata and register ontology information
    ndata = ad.concat([adata, ddata], join="outer", axis=1)
    ndata = ndata[:,ndata.var_names.sort_values()].copy()
    ndata.obs = adata.obs
    ndata.obsm = adata.obsm
    ndata.uns['_ontovae'] = {}
    ndata.uns['_ontovae']['thresholds'] = (top_thresh, bottom_thresh)
    ndata.uns['_ontovae']['annot'] = ontobj.extract_annot(top_thresh=top_thresh, bottom_thresh=bottom_thresh)
    ndata.uns['_ontovae']['genes'] = ontobj.extract_genes(top_thresh=top_thresh, bottom_thresh=bottom_thresh)
    ndata.uns['_ontovae']['masks'] = ontobj.extract_masks(top_thresh=top_thresh, bottom_thresh=bottom_thresh)

    # register SCVI fields
    ndata = ndata.copy()
    anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=is_count_data),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            )
        ]
    adata_manager = AnnDataManager(
        fields=anndata_fields
    )
    adata_manager.register_fields(ndata)
    
    return ndata



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

def paired_wilcox_test(adata: AnnData, control, perturbed, direction='up', option='terms'):
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
        top_thresh
            top threshold for trimming
        bottom_thresh
            bottom_threshold for trimming
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
            res = pd.concat((onto_annot, res))
        
        else:
            res = pd.DataFrame({'gene': onto_genes,
                                'stat': stat,
                                'pval' : pvals,
                                'qval': qvals[1]})

        res = res.sort_values('pval').reset_index(drop=True)
        return(res)