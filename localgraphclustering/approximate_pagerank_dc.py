"""
This module provides a method for computing the approximate pagerank on the double cover of a graph locally.
"""
import warnings
from .cpp import *


def approximate_pagerank_dc(G,
                            ref_nodes,
                            iterations: int = 1000000,
                            alpha: float = 0.15,
                            epsilon: float = 1.0e-6,
                            normalize: bool = True):
    """
    Computes PageRank vector on the double cover locally.
    --------------------------------

    Uses the double-cover push operation.

    Parameters
    ----------

    G: GraphLocal
        The graph n which to operate

    ref_nodes: Sequence[int]
        A sequence of reference nodes, i.e., nodes of interest around which
        we are looking for a target cluster.

    Parameters (optional)
    ---------------------

    alpha: float
        Default == 0.15
        Teleportation parameter of the personalized PageRank linear system.
        The smaller the more global the personalized PageRank vector is.

    epsilon: float
        Default == 1.0e-6
        Approximation error when computing the personalised pagerank.

    iterations: int
        Default = 1000000
        Maximum number of iterations of ACL algorithm.

    normalize: bool
        Default = True
        Normalize the output to be directly input into sweepcut routines.

    Returns
    -------

    A tuple containing
     - An np.ndarray with the node ids of the returned cluster.
     - An np.ndarray with the values of the pagerank vector on the returned nodes.
    """
    # Initialise variables
    x_ids, values = None, None

    if G.weighted:
        warnings.warn("The weights of the graph will be discarded.")

    n = G.adjacency_matrix.shape[0]
    (x_ids_1, x_ids_2, values_1, values_2) = dcpagerank_cpp(n, G.ai, G.aj, alpha, epsilon, ref_nodes,
                                                                    iterations, simplify=False)

    if normalize:
        # we can't use degrees because it could be weighted
        for i in range(len(x_ids_1)):
            values_1[i] /= (G.ai[x_ids_1[i] + 1] - G.ai[x_ids_1[i]])
        for i in range(len(x_ids_2)):
            values_2[i] /= (G.ai[x_ids_2[i] + 1] - G.ai[x_ids_2[i]])

    return x_ids_1, x_ids_2, values_1, values_2
