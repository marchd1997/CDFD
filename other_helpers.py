import numpy as np
import networkx as nx
import numpy as np
from scipy.sparse import lil_array, csr_array, diags_array, coo_array, identity, csr_matrix, eye
from scipy.sparse.linalg import spsolve, lsqr,cg
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




def finn_cycling_index_sparse(T, x, eps=1e-8, solver="lsqr", tol=1e-8, maxiter=None):
    """
    Compute Finn’s Cycling Index (FCI) for a large sparse network.

    Parameters
    ----------
    T : array or scipy.sparse (n×n)
        Flows matrix, T[i,j] = flow from i → j.
    x : array_like (n,)
        Total output of each node (column sums of T).
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
    # ensure CSR format and 1‑d x
    T = csr_matrix(T)
    x = np.ravel(x).astype(float)
    n = T.shape[0]

    # column‑normalize to form A
    Tc = T.tocsc(copy=True)
    Tc.data = Tc.data.astype(float)
    for j in range(n):
        if x[j] != 0:
            s, e = Tc.indptr[j], Tc.indptr[j+1]
            Tc.data[s:e] /= x[j]
    A = Tc.tocsr()

    # build regularized Leontief matrix L = I – A + eps·I
    L = eye(n, format="csr") - A
    L = L.tolil()
    L.setdiag(L.diagonal() + eps)
    L = L.tocsr()

    # compute node FCI
    fci_node = np.zeros(n)
    valid   = np.zeros(n, bool)
    for i in range(n):
        b = np.zeros(n); b[i] = 1.0

        if solver == "lsqr":
            sol = lsqr(L, b, atol=tol, btol=tol)[0]
            Lii = sol[i] if np.isfinite(sol[i]) else 0.0

        elif solver == "cg":
            sol, info = cg(L, b, tol=tol, maxiter=maxiter)
            Lii = sol[i] if info == 0 and np.isfinite(sol[i]) else 0.0

        else:  # direct
            try:
                sol = spsolve(L, b)
                Lii = sol[i]
            except:
                Lii = 0.0

        if Lii > eps:
            f = 1 - 1 / Lii
            fci_node[i] = np.clip(f, 0, 1)
            valid[i]   = True

    # aggregate valid nodes
    if not valid.any():
        return 0.0
    w = x[valid]
    return float((fci_node[valid] * w).sum() / w.sum())

