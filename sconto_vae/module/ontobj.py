import contextlib
import io
import json
import pandas as pd
import numpy as np
import itertools
from goatools.base import get_godag
from goatools.semsim.termwise.wang import SsWang
from sconto_vae.module.ontobj_utils import *


class Ontobj():
    """
    This class functions as a container for a preprocessed ontology.
    An instance of the class Ontobj is needed by the OntoVAE class to train OntoVAE models.

    Predefined slots:
    annot_base: contains annotation files for ontology with the following columns
            'ID': The ID of the DAG term
            'Name': The name 
            'depth': the depth (longest distance to a root node)
            'children': number of children of the term
            'parents': number of parents of the term
            'descendants': number of descendant terms
            'desc_genes': number of genes annotated to term and all its descendants
            'genes': number of genes directly annotated to term
    genes_base: contains genes that can be mapped to ontology in alphabetical order
    graph_base: a dictionary with ontology relationships (children -> parents)
    annot, genes and graph can contain different trimmed versions
    desc_genes: a dictionary with all descendant genes (terms -> descendant genes)
    sem_sim: semantic similarities for all genes of one of the elements in the genes slot

    Parameters
    ----------
    description
        to identify the object, used ontology or gene identifiers can be specified here, for example 'GO' or 'HPO' or 'GO_BP'
    """

    __slots__=(
        'description', 
        'identifiers', 
        'annot_base', 
        'genes_base', 
        'graph_base', 
        'annot', 
        'genes', 
        'graph', 
        'desc_genes', 
        'masks', 
        'sem_sim'
        )

    def __init__(self, description: str=None):
        super(Ontobj, self).__init__()

        self.description = description
        self.identifiers = None
        self.annot_base = None
        self.genes_base = None
        self.graph_base = None
        self.annot = {}
        self.genes = {}
        self.graph = {}
        self.desc_genes = {}
        self.masks = {}
        self.sem_sim = {}

    def _dag_annot(self, dag: dict, gene_annot: pd.DataFrame, filter_id: str=None):

        """
        Creates annotation dataframe from imported obo file.
        
        Parameters
        ----------
        dag
            a dag parsed from an obo file
        gene_annot
            pandas dataframe containing gene -> term annotation
        filter_id
            to pass if ids should be filtered, e.g.
            filter_id = 'biological_process'

        Returns
        -------
        A pandas dataframe with ontology annotation, sorted by depth and ID.
        Each row is one term, the columns are:
            'ID': The ID of the ontology term
            'Name': The name of the ontology term
            'depth': the depth (longest distance to a root node)
            'children': number of children of the term
            'parents': number of parents of the term
        """
  
        # parse obo file and create list of term ids
        term_ids = list(set([vars(dag[term_id])['id'] for term_id in list(dag.keys())]))

        # if filter_id was specified, filter the ontology terms
        if filter_id is not None:
            term_idx = [vars(dag[term_id])['namespace'] == filter_id for term_id in term_ids]
            valid_ids = [N for i,N in enumerate(term_ids) if term_idx[i] == True]
            term_ids = valid_ids
            gene_annot = gene_annot[gene_annot.ID.isin(term_ids)]
        
        # extract information for annot file
        terms = [vars(dag[term_id])['name'] for term_id in term_ids]
        depths = [vars(dag[term_id])['depth'] for term_id in term_ids]
        num_children = [len(vars(dag[term_id])['children']) for term_id in term_ids]
        num_parents = [len(vars(dag[term_id])['parents']) for term_id in term_ids]

        # create annotation pandas dataframe
        annot = pd.DataFrame({'ID': term_ids,
                        'Name': terms,
                        'depth': depths,
                        'children': num_children,
                        'parents': num_parents})
        annot = annot.sort_values(['depth', 'ID'])
        return annot


    def initialize_dag(self, obo: str = None, gene_annot: str = None, filter_id: str = None):

        """
        Dag is initialized from obo file and annotation.
        The slots annot_base, genes_base, and graph_base are filled.

        Parameters
        -------------
        obo
            Path to the obo file if ontology is to be build

        gene_annot
            gene_annot
            Path two a tab-separated 2-column text file containing gene to geneset annotations
            Gene1   Geneset1
            Gene1   Geneset2
            ...

        filter_id
            to pass if ids should be filtered, e.g.
            filter_id = 'biological_process'
        """

        if gene_annot is None:
            raise ValueError("Please provide a file containing gene to geneset mappings.")
        
        gene_annot = pd.read_csv(gene_annot, sep="\t", header=None)
        gene_annot.columns = ['Gene', 'ID']
        
        if obo is not None:

            # load obo file and gene -> term mapping file
            dag = get_godag(obo, optional_attrs={'relationship'}, prt=None)
            self.identifiers = 'Ensembl' if 'ENS' in gene_annot.iloc[0,0] else 'HGNC'

            # create initial annot file
            if filter_id is not None:
                annot = self._dag_annot(dag, gene_annot, filter_id=filter_id)
            else:
                annot = self._dag_annot(dag, gene_annot)

            # convert gene annot file to dictionary
            gene_term_dict = {a: b["ID"].tolist() for a,b in gene_annot.groupby("Gene")}

            # convert the dag to a dictionary
            term_term_dict = {term_id: [x for x in vars(dag[term_id])['_parents'] if x in annot.ID.tolist()] for term_id in annot[annot.depth > 0].ID.tolist()}

            # reverse the DAG to be able to count descendants and descendant genes
            gene_dict_rev = reverse_graph(gene_term_dict)
            term_dict_rev = reverse_graph(term_term_dict)

            # count descendants and descendant genes and add to annot
            num_desc = []
            num_genes = []

            for term in annot.ID.tolist():
                desc = get_descendants(term_dict_rev, term)
                num_desc.append(len(set(desc)) - 1)
                genes = get_descendant_genes(gene_dict_rev, desc)
                num_genes.append(len(set(genes)))
            
            annot['descendants'] = num_desc
            annot['desc_genes'] = num_genes

            # remove terms that don't have any descendant genes
            annot_updated = annot[annot.desc_genes > 0]
            annot_updated = annot_updated.sort_values(['depth', 'ID']).reset_index(drop=True)

            # update the dag dict using only the good IDs
            term_dict = {term_id: [x for x in vars(dag[term_id])['_parents'] if x in annot_updated.ID.tolist()] for term_id in annot_updated[annot_updated.depth > 0].ID.tolist()}
            term_dict.update(gene_term_dict)

            # update the annotation file

            # number of annotated genes
            term_size = gene_annot['ID'].value_counts().reset_index()
            term_size.columns = ['ID', 'genes']
            annot_updated = pd.merge(annot_updated, term_size, how='left', on='ID')
            annot_updated['genes'] = annot_updated['genes'].fillna(0)

            # recalculate number of children
            all_parents = list(term_dict.values())
            all_parents = [item for sublist in all_parents for item in sublist]
            refined_children = [all_parents.count(pid) - annot_updated[annot_updated.ID == pid].genes.values[0] for pid in annot_updated.ID.tolist()]
            annot_updated['children'] = refined_children

            # recalculate number of descendants
            term_dict = {term_id: [x for x in vars(dag[term_id])['_parents'] if x in annot_updated.ID.tolist()] for term_id in annot_updated[annot_updated.depth > 0].ID.tolist()}
            term_dict_rev = reverse_graph(term_dict)
            num_desc = []
            for term in annot_updated.ID.tolist():
                desc = get_descendants(term_dict_rev, term)
                num_desc.append(len(set(desc)) - 1)
            annot_updated['descendants'] = num_desc 
            term_dict.update(gene_term_dict)

        else:
            annot_updated = pd.DataFrame(np.sort(gene_annot.ID.unique()))
            annot_updated.columns = ["ID"]
            annot_updated["Name"] = annot_updated["ID"]
            annot_updated['depth'] = 0
            annot_updated['children'] = 0
            annot_updated['parents'] = 0
            annot_updated['descendants'] = 0
            annot_updated['desc_genes'] = [len(gene_annot[gene_annot.ID == tf]) for tf in annot_updated.ID.tolist()]
            annot_updated['genes'] = annot_updated['desc_genes']

            term_dict = {g: gene_annot[gene_annot.Gene == g].ID.tolist() for g in gene_annot.Gene.unique()}

        # fill the basic slots
        self.annot_base = annot_updated
        self.genes_base = sorted(list(set(gene_annot.Gene.tolist())))
        self.graph_base = term_dict
        

    def trim_dag(self, top_thresh: int=1000, bottom_thresh: int=30):

        """
        DAG is trimmed based on user-defined thresholds.
        Trimmed version is saved in the graph, annot and genes slots.

        Parameters
        ----------
        top_thresh
            top threshold for trimming: terms with > desc_genes will be pruned
        bottom_thresh
            bottom_threshold for trimming: terms with < desc_genes will be pruned and
            their genes will be transferred to their parents
        """

        # check if base versions of files exits
        if self.graph_base is None:
            raise ValueError('Initial graph has not been created, initialize_dag function needs to be run first!')
        else:
            graph_base = self.graph_base.copy()

        if self.annot_base is None:
            raise ValueError('Initial annotation has not been created, initialize_dag function needs to be run first!')
        else:
            annot_base = self.annot_base.copy()

        if len(annot_base.depth.unique()) > 1:


            # get terms for trimming
            top_terms = annot_base[annot_base.desc_genes > top_thresh].ID.tolist()
            bottom_terms = annot_base[annot_base.desc_genes < bottom_thresh].ID.tolist()[::-1]

            # trim the DAG
            with contextlib.redirect_stdout(io.StringIO()):
                term_dict_ttrim = trim_DAG_top(graph_base, annot_base.ID.tolist(), top_terms)
            with contextlib.redirect_stdout(io.StringIO()):
                term_dict_trim = trim_DAG_bottom(term_dict_ttrim, annot_base.ID.tolist(), bottom_terms)

            ### ANNOTATION FILE UPDATE ###

            # adjust the annotation file
            new_annot = annot_base[annot_base.ID.isin(top_terms + bottom_terms) == False].reset_index(drop=True)

            # split the DAG
            term_trim = {key: term_dict_trim[key] for key in list(term_dict_trim.keys()) if key in new_annot.ID.tolist()}
            gene_trim = {key: term_dict_trim[key] for key in list(term_dict_trim.keys()) if key not in new_annot.ID.tolist()}  

            # reverse the separate DAGs
            term_trim_rev = reverse_graph(term_trim)
            gene_trim_rev = reverse_graph(gene_trim)

            # calculate new children, parent and gene numbers
            new_children = [len(term_trim_rev[term]) if term in list(term_trim_rev.keys()) else 0 for term in new_annot.ID.tolist()]
            new_parents = [len(term_trim[term]) if term in list(term_trim.keys()) else 0 for term in new_annot.ID.tolist()]
            new_genes = [len(gene_trim_rev[term]) if term in list(gene_trim_rev.keys()) else 0 for term in new_annot.ID.tolist()]

            # calculate new descendants and descendant genes
            num_desc = []
            num_genes = []

            desc_genes = {}

            for term in new_annot.ID.tolist():
                desc = get_descendants(term_trim_rev, term)
                num_desc.append(len(set(desc)) - 1)
                genes = set(get_descendant_genes(gene_trim_rev, desc))
                desc_genes[term] = list(genes)
                num_genes.append(len(genes))
            
            # update the annot file
            new_annot['children'] = new_children
            new_annot['parents'] = new_parents
            new_annot['genes'] = new_genes
            new_annot['descendants'] = num_desc
            new_annot['desc_genes'] = num_genes

            # set the depth of all terms with 0 parents to 0
            new_annot.loc[new_annot.parents == 0, 'depth'] = 0

            # adjust depth of other terms
            min_depth = np.min(new_annot['depth'][new_annot['depth'] != 0])

            def adjust_depth(row):
                if row['depth'] > 0:
                    return row['depth'] - (min_depth - 1)
                else:
                    return 0
            
            new_annot['depth'] = new_annot.apply(lambda row: adjust_depth(row), axis=1)
            new_annot = new_annot.sort_values(['depth', 'ID']).reset_index(drop=True)

        else:
            new_annot = annot_base[(annot_base.genes >= bottom_thresh) & (annot_base.genes <= top_thresh)]
            graph_rev = reverse_graph(graph_base)
            graph_rev_trim = {key: graph_rev[key] for key in list(graph_rev.keys()) if key in new_annot.ID.tolist()}
            term_dict_trim = reverse_graph(graph_rev_trim)
            gene_trim = term_dict_trim 
            desc_genes = graph_rev_trim

        # save trimming results in respective slots
        self.annot[str(top_thresh) + '_' + str(bottom_thresh)] = new_annot
        self.graph[str(top_thresh) + '_' + str(bottom_thresh)] = term_dict_trim
        self.genes[str(top_thresh) + '_' + str(bottom_thresh)] = sorted(list(gene_trim.keys()))
        self.desc_genes[str(top_thresh) + '_' + str(bottom_thresh)] = desc_genes



    def create_masks(self, top_thresh: int=None, bottom_thresh: int=None):

        """
        Creation of masks to initialize the wiring in the latent space and decoder of OntoVAE.
            
        Parameters
        ----------
        top_thresh
            top threshold for trimming
        bottom_thresh
            bottom_threshold for trimming
        """

        # check if trim_dag function was run
        if not self.annot:
              raise ValueError('The trim_dag() function needs to be run first!')

        # define top and borrom thresh
        if top_thresh is not None and bottom_thresh is not None:
            if not str(top_thresh) + '_' + str(bottom_thresh) in self.annot.keys():
                raise ValueError('Available trimming thresholds are: ' + ', '.join(list(annot.genes.keys())))
        else:
            top_thresh = list(self.annot.keys())[0].split('_')[0]
            bottom_thresh = list(self.annot.keys())[0].split('_')[1]

        annot = self.annot[str(top_thresh) + '_' + str(bottom_thresh)].copy()
        onto_dict = self.graph[str(top_thresh) + '_' + str(bottom_thresh)].copy()
        genes = self.genes[str(top_thresh) + '_' + str(bottom_thresh)].copy()

        # check if mask slot for thresholds exists
        if str(top_thresh) + '_' + str(bottom_thresh) not in self.masks.keys():
            self.masks[str(top_thresh) + '_' + str(bottom_thresh)] = {}

        # get all possible depth combos
        depth = annot.loc[:,['ID', 'depth']]
        gene_depth = pd.DataFrame({'ID': genes, 'depth': np.max(depth.depth)+1})
        depth = pd.concat([depth.reset_index(drop=True), gene_depth], axis=0)
        depth_combos = list(itertools.combinations(list(set(depth['depth'])), 2))

        # create binary matrix for all possible depth combos
        bin_mat_list = [create_binary_matrix(depth, onto_dict, p[1], p[0]) for p in depth_combos]
        masks = self._decoder_masks(depth, bin_mat_list)
        self.masks[str(top_thresh) + '_' + str(bottom_thresh)] = masks


    def _decoder_masks(self, depth, bin_mat_list):
        """
        Helper function to create binary masks for decoder
        """
        levels = ['Level' + str(d) for d in list(set(depth['depth'].tolist()))]
        mask_cols = [list(levels)[0:i+1][::-1] for i in range(len(levels)-1)]
        mask_rows = levels[1:]

        idx = [[mat.columns.name in mask_cols[i] and mat.index.name == mask_rows[i] for mat in bin_mat_list] for i in range(len(mask_rows))]
        masks = [np.array(pd.concat([N for i,N in enumerate(bin_mat_list) if j[i] == True][::-1], axis=1)) for j in idx]
        return masks


    def compute_wsem_sim(self, obo: str, top_thresh: int=None, bottom_thresh: int=None):

        """
        Wang semantic similarities between a list of ontology terms are computed.
        
        Parameters
        ----------
        obo
            Path to the obo file
        top_thresh
            top threshold for trimming
        bottom_thresh
            bottom_threshold for trimming
        """

        # check if trim_dag function was run
        if not self.annot:
              raise ValueError('The trim_dag() function needs to be run first!')

        # define top and borrom thresh
        if top_thresh is not None and bottom_thresh is not None:
            if not str(top_thresh) + '_' + str(bottom_thresh) in self.annot.keys():
                raise ValueError('Available trimming thresholds are: ' + ', '.join(list(self.annot.genes.keys())))
        else:
            top_thresh = list(self.annot.keys())[0].split('_')[0]
            bottom_thresh = list(self.annot.keys())[0].split('_')[1]

        annot = self.annot[str(top_thresh) + '_' + str(bottom_thresh)].copy()

        dag = get_godag(obo, optional_attrs={'relationship'}, prt=None)
        ids = annot['ID'].tolist()
        wang = SsWang(ids, dag)
        wsem_sim = [[wang.get_sim(id1, id2) for id2 in ids] for id1 in ids]
        wsem_sim = np.array(wsem_sim)
        self.sem_sim[str(top_thresh) + '_' + str(bottom_thresh)] = wsem_sim

    
    def extract_annot(self, top_thresh: int=None, bottom_thresh: int=None):
        """
        Helper function to extract table from annot slot.

        Parameters
        ----------
        top_thresh
            top threshold for trimming
        bottom_thresh
            bottom_threshold for trimming
        """

        # check if trim_dag function was run
        if not self.annot:
              raise ValueError('The trim_dag() function needs to be run first!')

        # define top and borrom thresh
        if top_thresh is not None and bottom_thresh is not None:
            if not str(top_thresh) + '_' + str(bottom_thresh) in self.annot.keys():
                raise ValueError('Available trimming thresholds are: ' + ', '.join(list(self.annot.genes.keys())))
        else:
            top_thresh = list(self.annot.keys())[0].split('_')[0]
            bottom_thresh = list(self.annot.keys())[0].split('_')[1]

        return self.annot[str(top_thresh) + '_' + str(bottom_thresh)].copy()

    def extract_genes(self, top_thresh: int=None, bottom_thresh: int=None):
        """
        Helper function to extract list from genes slot.

        Parameters
        ----------
        top_thresh
            top threshold for trimming
        bottom_thresh
            bottom_threshold for trimming
        """
        # check if trim_dag function was run
        if not self.annot:
              raise ValueError('The trim_dag() function needs to be run first!')

        # define top and borrom thresh
        if top_thresh is not None and bottom_thresh is not None:
            if not str(top_thresh) + '_' + str(bottom_thresh) in self.annot.keys():
                raise ValueError('Available trimming thresholds are: ' + ', '.join(list(self.annot.genes.keys())))
        else:
            top_thresh = list(self.annot.keys())[0].split('_')[0]
            bottom_thresh = list(self.annot.keys())[0].split('_')[1]

        return self.genes[str(top_thresh) + '_' + str(bottom_thresh)].copy()

    def extract_masks(self, top_thresh: int=None, bottom_thresh: int=None):
        """
        Helper function to extract masks from masks slot.

        Parameters
        ----------
        top_thresh
            top threshold for trimming
        bottom_thresh
            bottom_threshold for trimming
        """
        # check if trim_dag function was run
        if not self.annot:
              raise ValueError('The trim_dag() function needs to be run first!')

        # define top and borrom thresh
        if top_thresh is not None and bottom_thresh is not None:
            if not str(top_thresh) + '_' + str(bottom_thresh) in self.annot.keys():
                raise ValueError('Available trimming thresholds are: ' + ', '.join(list(self.annot.genes.keys())))
        else:
            top_thresh = list(self.annot.keys())[0].split('_')[0]
            bottom_thresh = list(self.annot.keys())[0].split('_')[1]

        return self.masks[str(top_thresh) + '_' + str(bottom_thresh)].copy()
    
    def remove_link(self, term: str, gene: str, top_thresh: int=None, bottom_thresh: int=None):
        """
        Modifies the masks slot by removing the link between a gene and a term.

        Parameters
        ----------
        term
            id of the term
        gene
            the gene
        top_thresh
            top threshold for trimming
        bottom_thresh
            bottom_threshold for trimming
        """

        # check if trim_dag function was run
        if not self.annot:
              raise ValueError('The trim_dag() function needs to be run first!')

        # define top and borrom thresh
        if top_thresh is not None and bottom_thresh is not None:
            if not str(top_thresh) + '_' + str(bottom_thresh) in self.annot.keys():
                raise ValueError('Available trimming thresholds are: ' + ', '.join(list(self.annot.genes.keys())))
        else:
            top_thresh = list(self.annot.keys())[0].split('_')[0]
            bottom_thresh = list(self.annot.keys())[0].split('_')[1]
            
        onto_annot = self.extract_annot(top_thresh=top_thresh,
                                        bottom_thresh=bottom_thresh)
        genes = self.extract_genes(top_thresh=top_thresh,
                                        bottom_thresh=bottom_thresh)

        # retrieve indices to remove link
        # for the term, we need to work around, as terms in masks are sorted reversed (Depth 15 -> Depth 14 -> Depth 13 ...)
        term_depth = onto_annot[onto_annot.ID == term].depth.to_numpy()[0]
        depth_counts = onto_annot.depth.value_counts().sort_index(ascending=False)
        start_point = depth_counts[depth_counts.index > term_depth].sum()
        annot_sub = onto_annot[onto_annot.depth == term_depth]
        term_idx = annot_sub[annot_sub.ID == term].index.to_numpy()
        gene_idx = genes.index(gene)

        self.masks[str(top_thresh) + '_' + str(bottom_thresh)][-1][gene_idx, start_point + term_idx] = 0


    def save(self, path: str):
        """
        Function to save the ontobj as a dict
        """
        ontobj = {}
        ontobj['description'] = self.description
        ontobj['identifiers'] = self.identifiers 
        ontobj['annot_base'] = self.annot_base.to_dict()
        ontobj['genes_base'] = self.genes_base
        ontobj['graph_base'] = self.graph_base 
        ontobj['annot'] = {key: value.to_dict() for key, value in self.annot.items()}
        ontobj['genes'] = self.genes
        ontobj['graph'] = self.graph
        ontobj['desc_genes'] = self.desc_genes
        ontobj['masks'] = {key: [m.tolist() for m in value] for key, value in self.masks.items()}
        ontobj['sem_sim'] = {key: value.tolist() for key, value in self.sem_sim.items()}

        with open(path, 'w') as fp:
            json.dump(ontobj, fp)
    
    def load(self, path: str):
        """
        Helper function to load existing ontobj
        """
        with open(path, 'r') as fp:
            ontobj = json.load(fp)
        
        self.description = ontobj['description']
        self.identifiers = ontobj['identifiers']
        self.annot_base = pd.DataFrame(ontobj['annot_base'])
        self.genes_base = ontobj['genes_base']
        self.graph_base = ontobj['graph_base']
        self.annot = {key: pd.DataFrame(value).reset_index(drop=True) for key, value in ontobj['annot'].items()}
        self.genes = ontobj['genes']
        self.graph = ontobj['graph']
        self.desc_genes = ontobj['desc_genes']
        self.masks = {key: [np.array(m) for m in value] for key, value in ontobj['masks'].items()}
        self.sem_sim = {key: np.array(value) for key, value in ontobj['sem_sim'].items()}

        del ontobj