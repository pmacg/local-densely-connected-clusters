from typing import *
import numpy as np
from .cpp import *
from .GraphLocal import GraphLocal
from .algorithms import sweepcut
import warnings


def sweep_cut(G: GraphLocal,
              p: Union[Sequence[float],Tuple[Sequence[int],Sequence[float]]],
              do_sort: bool = True,
              cpp: bool = True):
    """
    It implements a sweep cut rounding procedure for local graph clustering.

    Parameters
    ----------

    G: GraphLocal

    p: Sequence[float] or Tuple[Sequence[int],Sequence[float]]
        There are three ways to describe the vector used for sweepcut
        The first is just a list of n numbers where n is the nubmer of vertices
        of the graph.

        The second is a pair of vectors that describe a sparse input.

    do_sort: binary
        default = True
        If do_sort is equal to 1 then vector p is sorted in descending order first.
        If do_sort is equal to 0 then vector p is not sorted. In this case,
        only the order of the elements is used. This really only makes
        sense for the sparse input case.

    cpp: bool
        default = True
        Use the faster C++ version or not.

    Returns
    -------

    It returns in a list of length 2 with the following:

    output 0: list
        Stores indices of the best clusters found by the last called rounding procedure.

    output 1: float
        Stores the value of the best conductance found by the last called rounding procedure.
    """

    n = G.adjacency_matrix.shape[0]

    sparsevec = False
    if isinstance(p, tuple):
        if len(p) == 2: # this is a valid sparse input
            nnz_idx = p[0]
            nnz_val = p[1]
            sparsevec = True
        elif len(p) == n:
            pass # this is okay, and will be handled below
        else:
            raise Exception("Unknown input type.")

    if sparsevec == False:
        nnz_idx = np.array(range(0,n), dtype=G.aj.dtype)
        nnz_val = np.array(p, copy=False)
        assert(len(nnz_val) == n)

    nnz_ct = len(nnz_idx)

    if nnz_ct == 0:
        return [[],1]

    if cpp:
        #if fun == None: fun = sweepcut_cpp(G.ai, G.aj, G.lib, 1 - do_sort)
        (length,clus,cond) = sweepcut_cpp(n, G.ai, G.aj, G.adjacency_matrix.data, nnz_idx, nnz_ct, nnz_val, 1 - do_sort)
        return [clus,cond]
    else:
        if sparsevec:
            warnings.warn("Input will be converted to a dense vector, set \"cpp\" to be True for better performance")
        tmp = np.zeros(n)
        tmp[nnz_idx] = nnz_val
        output = sweepcut(tmp,G)
        return [output[0],output[1]]


def sweep_cut_dc(G: GraphLocal, x_ind_1, x_ind_2, values_1, values_2, normalise_by_degree=False):
    """
    Compute the sweep cut on the double cover of a graph. G is the graph on which to operate.
        x_ind_1 and values_1 give the sparse values on the first set of vertices
        x_ind_2 and values_2 give the sparse values on the second set of vertices
    :param G: the GraphLocal object on which we are operating
    :param x_ind_1: the vertex indices for the values on the first set of vertices
    :param x_ind_2: the vertex indices for the values on the second set of vertices
    :param values_1: the values on the first set of vertices
    :param values_2: the values on the second set of vertices
    :param normalise_by_degree: whether to first normalise the values in the given vector by the degrees of the vertices
    :return: tuple containing:
      - list of vertex indices giving the best cluster
      - conductance (in the double cover) of this cluster
    """
    # n is the number of vertices in the original graph (not the double cover)
    n = G.adjacency_matrix.shape[0]

    # Normalise if we need to
    if normalise_by_degree:
        for i, v in enumerate(values_1):
            values_1[i] = v / G.d[i]
        for i, v in enumerate(values_2):
            values_2[i] = v / G.d[i]

    # Construct the vector of indices for the double cover
    x_ind = np.concatenate((x_ind_1, [ind + n for ind in x_ind_2]))

    # Get the length of the input vector
    x_len = len(x_ind)
    if x_len == 0:
        return [], 1

    values = np.concatenate((values_1, values_2))

    (length, cluster, conductance) = sweepcut_cpp(
        2 * n, G.ai_dc, G.aj_dc, G.adjacency_matrix_dc.data, x_ind, x_len, values, 0)
    return cluster, conductance


def sweep_cut_dc_from_signed_vec(G, x_ind, values, normalise_by_degree=False):
    """
    Given a (positive and negative) vector on the vertices in the graph G, run the two-sided sweep set algorithm to
    find an almost-bipartite set.

    :param G: the graph on which to run the algorithm.
    :param x_ind: the vertex indices on which the input vector is defined
    :param values: the input vector values on these indices
    :param normalise_by_degree: whether to normalise the vector by the vertex degrees before running the sweep set
                                algorithm.
    :return: A tuple containing:
      - L - the vertices in the left set
      - R - the vertices in the right set
      - bipart - the bipartiteness of the resulting set.
    """
    # Get the number of vertices in the graph
    n = G.adjacency_matrix.shape[0]

    # We here take advantage of the equivalence between bipartiteness in a graph and the conductance in the double cover
    # of the graph. As such, we can use the sweep set process on the double cover of the graph to get the two-sided
    # sweep cut from the original graph.
    #
    # We start by splitting the eigenvector into the positive and negative parts and assign these to each set of
    # vertices in the double cover.
    x_ind_1 = []
    x_ind_2 = []
    values_1 = []
    values_2 = []
    for i, ind in enumerate(x_ind):
        if values[i] > 0:
            x_ind_1.append(ind)
            values_1.append(values[i])
        elif values[i] < 0:
            x_ind_2.append(ind)
            values_2.append(-values[i])

    # Now, we can run the sweep set procedure on the double cover of the graph.
    sweepset_dc, conductance_dc = sweep_cut_dc(G, x_ind_1, x_ind_2, values_1, values_2,
                                               normalise_by_degree=normalise_by_degree)

    # Split the sweep set into L and R
    L = []
    R = []
    for index in sweepset_dc:
        if index < n:
            L.append(index)
        else:
            R.append(index - n)

    return L, R, conductance_dc

