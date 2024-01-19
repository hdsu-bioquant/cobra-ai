
import os
import sys
sys.path.append('workspace/sconto-vae')


from sconto_vae.module.ontobj import *
import scanpy as sc
import scvi
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import anndata as ad

from sconto_vae.module.ontobj import *
from sconto_vae.module.utils import *
from sconto_vae.model.sconto_vae import *

# Creation of Ontology object

def create_ontobj(path, file_ontology, file_gene_term, output, top_thres=1000, bottom_thresh=30):
    """Create an Ontobj
    Input: 
    1. path: path to the folder that stores *.obo file
    2. file_ontology: file name of the ontology file. (.obo)
    3. file_gene_term: file name of the gene-term mapping (.txt)
    4. output: Output file name: full path ending with .ontobj
    5. top_thres: Top thresold for the ontology. (numerial: default: 1000)
    6. bottom_thresh: Bottom thresold for the ontology. (numerial: default: 30)"""    
    

    # initialize the Ontobj
    ontobj = Ontobj(description='GO_ontobj')
    # initialize the ontology
    ontobj.initialize_dag(obo= path + file_ontology,
                    gene_annot= path + file_gene_term)


    # trim the ontology
    ontobj.trim_dag(top_thresh=top_thres, 
                bottom_thresh=bottom_thresh)
    # create masks for decoder initialization
    ontobj.create_masks(top_thresh=top_thres,
                    bottom_thresh=bottom_thresh)
    # compute Wang Semantic similarities (used for app)
    ontobj.compute_wsem_sim(obo= path + file_ontology,
                        top_thresh=top_thres,
                        bottom_thresh=bottom_thresh)
    # save ontobj
    ontobj.save(output)




# initialize Ontobj and load existing object
def load_ontobj(file):
    """Load the ontoobj
    
    Input: full path for the .ontobj file
    Output: an loaded ontobj"""
    ontobj = Ontobj()
    ontobj.load(file)
    return ontobj


def preprocess(file_sc):
    """Preprocess the single cell object
    Input: 
    
    file_sc: single cell rna seq object (full path), ending in h5ad.
    
    Output: a preprocessed adata
    Notice: For the next step, consider to split the adata after this preprocessing. 
    Use sconto_vae.module.utils.split_adata"""
    adata = sc.read_h5ad(file_sc)
    adata.raw = adata # Fix the Seurat-preprocessed data to raw data

    # Change the obs name for the compatiblity (replace "." with "_")
    adata.obs["orig_ident"] = adata.obs['orig.ident']
    adata.obs["cell_type"] = adata.obs['predicted.id']
    adata.obs["pct_counts_mt"] = adata.obs['percent.mt']

    sc.pp.regress_out(adata, ['nCount_RNA', 'pct_counts_mt']) # Regress out the RNA count and the percertage of mt genes

    return adata


def integration(adata, ontobj):
    """
    Integrate adata and ontobj
    
    Input: 
    1. adata
    2. Ontobj
    
    Output:
    adata"""
    adata.varm = ""
    adata = setup_anndata_ontovae(adata, ontobj)
    return adata
    
def trainer(adata, path, learning_rate=1e-4, kl_coefficient=1e-4, batch_size=128, epochs=200):
    """
    Input: 
    1. adata for training
    2. path: The output folder
    3. learning_rate: larning rate: 
    4. kl_coefficient
    5. batch_size: Batch size
    
    Output: Void. A trained scontovae model in the output folder"""
    
    os.makedirs(path, exist_ok=True)
    model = scOntoVAE(adata)
    model.train_model(path,   
                     lr=learning_rate,                                 
                     kl_coeff=kl_coefficient,                           
                     batch_size=batch_size,                          
                     epochs=epochs)      


def loader(adata, path):
    """
    Load the model
    Input: 
    
    1. adata: validation anndata
    2. path: model directory
    
    Output:
    model"""

    # load the best model
    model = scOntoVAE.load(adata=adata,
                        modelpath=path)
    return model


def get_umap_embedding(adata, model):
    """Get the latent space of adata from the scOntoVAE, 
    and get its UMAP embedding
    
    Input:
    1. adata: AnnData
    2. model: scOntoVAE model
    
    Output:
    1. UMAP embedding (a numpy array)
    2. UMAP embedding (a pandas dataframe). *Easier for making plots*"""
    arr = model.to_latent(adata)

    import umap
    import pandas as pd
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(arr)
    
    df_adata_compressed_2D = pd.DataFrame(embedding)
    df_adata_compressed_2D.columns = ["UMAP_1", "UMAP_2"]
    df_adata_compressed_2D["cell_type"] = adata.obs.cell_type.values
    df_adata_compressed_2D["modality"] = adata.obs.orig_ident.values
    
    return embedding, df_adata_compressed_2D



def get_pw_act(model, adata, file):
    """Get the pathway activities
    Input:
    1. model: the one from model_loader
    2. adata: the one used in model_loader
    3. file: give the file name (full path) to save the file
    
    Output:
    The pathway activity pd dataframe"""
  
    # compute pathway activities
    act = model.get_pathway_activities()
    df_act = pd.DataFrame(act)
    df_act.index = adata.obs_names
    df_act.columns = list(model.adata.uns['_ontovae']['annot']['Name'])
    df_act.to_csv(file)
    return df_act

