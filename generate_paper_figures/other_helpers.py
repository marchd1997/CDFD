import numpy as np
import networkx as nx
import numpy as np
import os


from scipy.sparse import lil_array, csr_array, csc_array, diags_array, coo_array, eye
from scipy.sparse.linalg import spsolve, inv
from scipy.sparse.csgraph import connected_components


from random import sample
from collections import Counter



# Moving to parent directory to import from CDFD and other_helpers
current_dir = os.getcwd()
os.chdir("..")
from CDFD import _group_index_labels 
# Moving back to original directory
os.chdir(current_dir)

def trophic_coherence(G):
    """Gets network coherence of a graph.   
    
    Parameters
    ----------
    G : nx.DiGraph (or Graph)
        Weighted  digraph
    
    Returns
    -------
     coherence : float
        Trophi coherence of G
    """
    coherence = 1-trophic_incoherence(G)
    return coherence

def trophic_incoherence(G):
    """Gets network incoherence of a graph.   
    
    Parameters
    ----------
    G : nx.DiGraph (or Graph)
        Weighted  digraph
    
    Returns
    -------
     incoherence : float
        Trophi incoherence of G
    """
    h = trophic_levels(G)
    W = coo_array(nx.adjacency_matrix(G)) # advantage of handeling this with matrix is node ordering same as in h
    
    incoherence = 0
    for i, j, weight in zip(W.row, W.col, W.data):
        incoherence += weight * (h[j]-h[i]-1)**2  
    total_weight = np.sum(W.data)
    incoherence = incoherence/total_weight
    
    return incoherence 
    
def trophic_levels(G): 
    """Gets trophic levels of G.   
    
    Parameters
    ----------
    G : nx.DiGraph (or Graph)
        Weighted  digraph
    
    Returns
    -------
     incoherence : np.array (float)
        Trophic levels of G
    """
    W = nx.adjacency_matrix(G)
    out_strength = W.sum(axis=1)
    in_strength = W.sum(axis=0).T
    
    # Get linear system
    net_strength = in_strength - out_strength
    L = diags_array( in_strength + out_strength )  - W - W.transpose()
    
    n_components, labels = connected_components(csgraph=L, directed=True)  # connection='strong'
    components_idx = _group_index_labels(labels)
    
    # Modify some rows so solution is unique (adds up to 0 in each component). 
    # Note that in each component any row is linear comb of others
    L_mod = lil_array(L)
    net_strength_mod = net_strength
    for idx in components_idx:
        L_mod.rows[idx[0]] = list(idx)
        L_mod.data[idx[0]] = len(idx) * [1]
        net_strength_mod[idx[0]] = 0
    L_mod =  csr_array(L_mod)
    
    # compute heights solving linear system
    h = spsolve(L_mod, net_strength_mod)
    return h

def in_cycle_ratio(G):
    '''Weight proportion that edges contained in any cycle represent'''
    W = nx.adjacency_matrix(G)
    n_components, labels = connected_components(csgraph=W, directed=True, connection='strong')  # connection='strong'
    components_idx = _group_index_labels(labels)
    weight_in_cycles = 0
    for idx in components_idx:
        weight_in_cycles += np.sum((W[idx, :][:, idx]).data)
    weight_in_cycles_ratio = weight_in_cycles / np.sum(W.data)
    return weight_in_cycles_ratio

def finn_cycling_index(G, tol_balanced=1e-8):
    """
    Compute Finn’s Cycling Index (FCI) for a large sparse network.

    Parameters
    ----------
    G : nx.DiGraph (or Graph)
        Weighted  digraph

    tol_balanced : float
        Relative tolarence to accept that network is balanced.

    Returns
    -------
    float
        System‑level FCI in [0,1].
    """
    # ensure CSR format and float 
    w = nx.adjacency_matrix(G)
    w = csc_array(w, dtype = 'float')

    # Find weakly connected compoenents 
    n_components, labels = connected_components(csgraph=w, directed=True, connection='weak')  
    components_idx = _group_index_labels(labels)
    
    # find unnormalized fci by adding scc
    fci = 0
    for idx in components_idx:  
        # Get the strongly connected component
        scc = w[idx, :][:, idx]
        fci += _fci_connected(scc)

    # Normalize fcc
    node_imbalances = w.sum(axis=0) - w.sum(axis=1)
    m = np.maximum(-node_imbalances, 0) # assume input(m) and output minimal so all nodes in w balanced. 
    x = w.sum(axis=0) + m 
    return float(fci / x.sum())

def _fci_connected(w, tol_balanced=1e-8):
    """
    Compute unnormalized Finn’s Cycling Index (FCI) for a connected sparse network.

    Parameters
    ----------
    w : array or scipy.sparse (n×n)
        Flows matrix, w[i,j] = flow from i → j.
    tol_balanced : float
        Relative tolarence to accept that network is balanced.

    Returns
    -------
    float
        System‑level unnormalized FCI.
    """
    # ensure CSR format and float 
    w = csc_array(w, dtype = 'float')
    n = w.shape[0]

    # Compute m and x
    node_imbalances = w.sum(axis=0) - w.sum(axis=1)
    m = np.maximum(-node_imbalances, 0) # assume input(m) and output minimal so all nodes in w balanced. 
    x = w.sum(axis=0) + m 

    # If balanced up to relative tolerance, fci is 1 so unnormalized is x.sum()
    max_weight = w.data.max() if w.data.size > 0 else 0
    if node_imbalances.max() <= tol_balanced * max_weight: 
        return float(x.sum())

    # Compute A
    normalizing_factors = np.divide(1., x, out=np.zeros_like(x), where = x!=0)
    normalizing_matrix = diags_array(normalizing_factors)
    A = w @ normalizing_matrix

    # Compute fci at node level
    L = inv(eye(n, format="csc") - A)
    L_diag = L.diagonal()
    fci_node =  np.divide(1., L_diag, out=np.ones_like(L_diag), where = L_diag!=0)
    fci_node = np.ones_like(fci_node) - fci_node
    
    return float((fci_node * x).sum() )

def uniform_multigraph (n,m):
    '''Generates ER directed weighted random graph with n nodes and m edges (self loops not allowed as in ER).'''
    edges = [tuple(sample(range(n), 2)) for _ in range(m)] # maybe sort and count some how to get weights directly
    weighted_edges = Counter(edges).items()
    
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for (i, j), weight in weighted_edges:
        G.add_edge(i, j, weight=weight)
    
    return G