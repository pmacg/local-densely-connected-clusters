"""
File containing methods for help with processing hermitian matrices.
"""
import scipy as sp
import scipy.sparse


def load_hermitian_adjacency(filename):
    """
    Given a file containing an edgelist of a directed graph, construct the hermitian adjacency matrix.

    Parameters
    ----------
    filename - the edgelist file of the graph

    Returns
    -------
    A scipy sparse complex matrix.
    """
    # Load all of the edges into memory
    all_edges = []
    n = 0
    with open(filename, 'r') as f_in:
        for line in f_in.readlines():
            new_edge = tuple([int(x) for x in line.strip().split('\t')])
            all_edges.append(new_edge)
            if max(new_edge) >= n:
                n = 1 + max(new_edge)

    # Create the adjacency matrix...
    adjacency_matrix = sp.sparse.lil_matrix((n, n), dtype=complex)
    for u, v in all_edges:
        adjacency_matrix[u, v] = 1j
        adjacency_matrix[v, u] = -1j

    # ...and return it
    return adjacency_matrix.tocsr()


