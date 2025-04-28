import numpy as np
import networkx as nx
import numpy as np
from scipy.sparse import lil_array, csr_array, diags_array, coo_array, identity, csr_matrix, eye
from scipy.sparse.linalg import spsolve, lsqr,cg,inv
from scipy.sparse.csgraph import connected_components
from CDFD import _group_index_labels 
from CDFD import get_circularity, is_decomposition, CDFD

from random import sample
from collections import Counter


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
    
    return incoherence 

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

def uniform_multigraph (n,m):
    '''Generates ER directed weighted random graph with n nodes and m edges (self loops not allowed as in ER).'''
    edges = [tuple(sample(range(n), 2)) for _ in range(m)] # maybe sort and count some how to get weights directly
    weighted_edges = Counter(edges).items()
    
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for (i, j), weight in weighted_edges:
        G.add_edge(i, j, weight=weight)
    
    return G

def finn_cycling_index(w, eps=0, solver="cg", tol=1e-8, maxiter=None):
    """
    Compute Finn’s Cycling Index (FCI) for a large sparse network.

    Parameters
    ----------
    w : array or scipy.sparse (n×n)
        Flows matrix, w[i,j] = flow from i → j.
    eps : float
        Tiny diagonal regularizer for I–A.
    solver : {"lsqr","cg","spsolve"}
        Method to solve (I–A+eps·I)x = e_i.
    tol : float
        Convergence tolerance for iterative solvers.
    maxiter : int or None
        Max iterations (for CG).

    Returns
    -------
    float
        System‑level FCI in [0,1].
    """
    # USE ARRAY INSTEAD OF MATRIX
    # ensure CSR format and float 
    w = csr_array(w)
    w.data = w.data.astype(float)
    # Add some little perturbation to avoid percfect singular
    w.data += np.random.normal(loc=0.0, scale=1e-8, size=w.data.shape)
    n = w.shape[0]

    # Assume input/output in economy is so that all nodes in w are balanced
    # As work in columns only need to worry about input
    # CHECK CAN INCREASe in and output (compansating) and results dont changes PROBABLY FALSE. 
    m = w.sum(axis=0) - w.sum(axis=1) # net inflow (in - out)
    m = np.maximum(-m, 0) # Only need add inflow when there is deficit (oposite sign is deal with outflow) 
    m = np.ravel(m) # ensure format of m (CHECK IF NEEDED)

    x = np.ravel(w.sum(axis=0)) + m 


    # MUTIPLY BY DIAGONAL
    # column‑normalize to form A (including incoming flow)
    for j in range(n):
        if x[j] != 0:
            s, e = w.indptr[j], w.indptr[j+1]
            w.data[s:e] /= x[j] 
    A = w.tocsr()
    L = inv(eye(n, format="csr") - A)
    fci_node = np.zeros(n)
    for i in range(n):
        Lii = L[i,i]
        if Lii > eps: # (1-eps)^-1 = epsilon (to first order)
            fci_node[i] = 1 - 1 / Lii
  
    sum_x = x.sum()
    if sum_x == 0:
        return 0.0
    
    return float((fci_node * x).sum() / sum_x)