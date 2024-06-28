import pandas as pd
import itertools
import copy


"""Graph traversal functions"""

def reverse_graph(graph: dict) -> dict:
    """
    Takes a graph represented as dict and returns a dict with reversed mappings.
    """
    reverse = {}
    for v in graph:
        for e in graph[v]:
            if e not in reverse:
                reverse[e] = []
            reverse[e].append(v)
    return reverse


def get_descendants(dag: dict, term: str) -> list:
    """
    Iterates over a directed acyclic graph (DAG) and returns all descendant terms of a given node.

    Parameters
    ----------
    dag: the DAG in dict format
    term: the node to be queried

    Output
    ------
    descendants: a list of all descendant nodes
    """

    descendants = []
    queue = []

    descendants.append(term)
    queue.append(term)

    while len(queue) > 0:
        node = queue.pop(0)
        if node in list(dag.keys()):
            children = dag[node]
            descendants.extend(children)
            queue.extend(children)
        else:
            pass

    return descendants


def get_descendant_genes(dag: dict, descendants: list) -> list:
    """
    Given a DAG, returns a list of genes belonging to a list of terms.
    """
    desc_dict = {key: dag[key] for key in list(dag.keys()) if key in descendants}
    genes = list(desc_dict.values())
    genes = [item for sublist in genes for item in sublist]
    return genes


def find_all_paths(graph: dict, start: str, end: str, path=[]) -> list:
    """
    recursive path-finding function (https://www.python-kurs.eu/hanser-blog/examples/graph2.py)

    Parameters
    ----------
    graph: a graph in dict form
    start: start point
    end: end point

    Output
    ------
    paths: a list of all possible paths between the two nodes
    """
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph:
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            new_paths = find_all_paths(graph, node, end, path)
            for p in new_paths: 
                paths.append(p)
    return paths


"""Graph trimming functions"""

def _trim_term_bottom(term: str, term_term_dict: dict, term_dict_rev: dict, gene_dict_rev: dict):
    """
    Function that prunes a term from the bottom of a DAG. 
    The term is being removed from its parents, at the same time, annotated genes are transferred to the parents.

    Parameters
    ----------
    term: the term to be trimmed off
    term_term_dict: mapping children -> parents excluding the genes
    term_dict_rev: parents(terms) -> children(terms)
    gene_dict_rev: parents(terms) -> children(genes)

    Output
    ------
    This function is changing the term_dict_rev and the gene_dict_rev variables
    """

    # check if term has parents (depth0 won't have)
    if term in list(term_term_dict.keys()):
        parents = copy.deepcopy(term_term_dict[term])
    else:
        parents = []

    # iterate over parents and remove the term from their children
    # also add the genes of the term to the genes of its parents
    if len(parents) > 0:
        for p in parents:
            term_dict_rev[p].remove(term)
            if p not in list(gene_dict_rev.keys()):
                gene_dict_rev[p] = []
            gene_dict_rev[p].extend(gene_dict_rev[term])
            gene_dict_rev[p] = list(set(gene_dict_rev[p])) # remove eventual duplicates

    # remove the term -> genes and term -> term entries from the dicts
    del gene_dict_rev[term]
    if term in list(term_dict_rev.keys()):
        del term_dict_rev[term]



def trim_DAG_bottom(dag: dict, all_terms, trim_terms) -> dict:
    """
    Takes a DAG and a list of terms which are pruned from the bottom of the DAG, and returns the trimmed DAG.

    Parameters
    ----------
    dag: the DAG to be trimmed
    all_terms: all ontology terms 
    trim_terms: ontology terms that need to be trimmed off

    Output
    ------
    term_dict: the trimmed DAG
    """

    # separate dict for terms only
    term_term_dict = {key: dag[key] for key in list(dag.keys()) if key in all_terms}

    # separate dict for genes only
    term_gene_dict = {key: dag[key] for key in list(dag.keys()) if key not in all_terms}   
    
    # reverse the separate dicts
    term_dict_rev = reverse_graph(term_term_dict)
    gene_dict_rev = reverse_graph(term_gene_dict)

    # run the trim_term function over all terms to update the dicts
    for t in trim_terms:
        print(t)
        _trim_term_bottom(t, term_term_dict, term_dict_rev, gene_dict_rev)

    # reverse back the dicts and combine
    term_dict = reverse_graph(term_dict_rev)
    gene_dict = reverse_graph(gene_dict_rev)
    term_dict.update(gene_dict)

    return term_dict


def _trim_term_top(term: str, term_dict_rev: dict, gene_dict_rev: dict):
    """
    Function that prunes a term from the top of a DAG.

    Input
    -----
    term: the term to be trimmed off
    term_dict_rev: parents(terms) -> children(terms)
    gene_dict_rev: parents(terms) -> children(genes)

    Output
    ------
    This function is changing the term_dict_rev and the gene_dict_rev variables

    """

    if term in list(term_dict_rev.keys()):
        del term_dict_rev[term]
    if term in list(gene_dict_rev.keys()):
        del gene_dict_rev[term]


def trim_DAG_top(dag: dict, all_terms: list, trim_terms: list) -> dict:
    """
    Takes a DAG and a list of terms which are pruned from the top of the DAG, and returns the trimmed DAG.

    Parameters
    ----------
    dag: the DAG to be trimmed
    all_terms: all ontology terms 
    trim_terms: ontology terms that need to be trimmed off

    Output
    ------
    term_dict: the trimmed DAG
    """

    # separate dict for ontology terms only
    term_term_dict = {key: dag[key] for key in list(dag.keys()) if key in all_terms}

    # separate dict for genes only
    term_gene_dict = {key: dag[key] for key in list(dag.keys()) if key not in all_terms}   
    
    # reverse the separate dicts
    term_dict_rev = reverse_graph(term_term_dict)
    gene_dict_rev = reverse_graph(term_gene_dict)

    # run the trim_term function over all terms to update the dicts
    for t in trim_terms:
        print(t)
        _trim_term_top(t, term_dict_rev, gene_dict_rev)

    # reverse back the dicts and combine
    term_dict = reverse_graph(term_dict_rev)
    gene_dict = reverse_graph(gene_dict_rev)
    term_dict.update(gene_dict)

    return term_dict


"""Mask creation"""

def create_binary_matrix(depth, dag: dict, childnum: int, parentnum: int):
    """
    Function that creates a binary matrix between terms of two different layers, indicating if there is a connection or not.

    Parameters
    ----------
    depth: a DF of all ontology terms with two columns ('ID', 'Depth')
    dag: the used DAG
    childnum: depth level to use for incoming connections
    parentnum: depth level to use for outgoing connections

    Output
    ------
    df: pandas DataFrame with connections between parents and children
    """
    children = depth.loc[depth['depth'] == childnum, 'ID'].tolist()
    parents = depth.loc[depth['depth'] == parentnum, 'ID'].tolist()

    df = pd.DataFrame(list(itertools.product(children, parents)), columns=['Level' + str(childnum), 'Level' + str(parentnum)])

    interact = [1 if y in dag[x] else 0 for x, y in zip(df['Level' + str(childnum)], df['Level' + str(parentnum)])]
    df['interact'] = interact

    df = df.pivot(index='Level' + str(childnum), columns='Level' + str(parentnum), values='interact')

    return(df)