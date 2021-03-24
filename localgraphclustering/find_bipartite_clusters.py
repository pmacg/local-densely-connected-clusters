"""
This file gives several methods for finding an almost-bipartite set.
Some of the methods are local in the sense that they will find a set close to some seed vertex in a graph.
"""
from .cpp import *
from .sweep_cut import sweep_cut_dc, sweep_cut_dc_from_signed_vec
from .algorithms import eig_nL
import localgraphclustering as lgc
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
from sklearn.cluster import KMeans
import math
import random
import numpy.random as npr
import numpy as np
from matplotlib import pyplot as plt

# Filter warnings for debugging
import warnings
# warnings.filterwarnings('error', category=sp.sparse.SparseEfficiencyWarning)


def lp_almost_bipartite(G, starting_vertex, T=100, xi_0=0.01, debug=False):
    """
    Find an almost-bipartite set close to starting_vertex, using the truncated power method algorithm given by
    Li and Peng.

    :param debug: whether to print debug statements
    :param G: a GraphLocal object on which to perform the algorithm
    :param starting_vertex: the index of the vertex at which to start the algorithm
    :param T: the number of iterations of the power method to perform.
    :param xi_0: the starting threshold for truncating the vectors.
    :return: A tuple containing:
      - L - the vertices in the left set
      - R - the vertices in the right set
      - bipart - the bipartiteness of the resulting set.
    """
    # Construct the pseudo-laplacian of the graph for use in the power method
    n = G.adjacency_matrix.shape[0]
    # D_inv = sp.sparse.spdiags(G.dn.transpose(), 0, n, n)
    # M = sp.sparse.identity(n) - G.adjacency_matrix.dot(D_inv)
    M = G.rw_laplacian

    # Construct the starting vector for the power method
    r = sp.sparse.csc_matrix((n, 1))
    r[starting_vertex] = 1

    best_bipartiteness = 1
    best_L = [starting_vertex]
    best_R = []
    xi_t = xi_0
    for t in range(T):
        # Perform the matrix product
        q_t = M.dot(r).asformat('csc')

        # Truncate the new vector
        indices = []
        data = []
        new_data_length = 0
        for i, v in enumerate(q_t.data):
            if abs(v) > xi_t:
                data.append(v)
                indices.append(q_t.indices[i])
                new_data_length += 1
        indptr = [0, new_data_length]
        r = sp.sparse.csc_matrix((data, indices, indptr), (n, 1))

        if debug:
            print(f"Iteration: {t}")
            print(indices)
            print(data)
            print()
            not_S = list(set(range(n)) - set(indices))
            G.draw_groups([indices, not_S], pos=lgc.get_pos_for_sbm_cycle(int(n / 8), 8))
            plt.show()

        # Run the bipartiteness sweep set
        L, R, bipart = sweep_cut_dc_from_signed_vec(G, indices, data, normalise_by_degree=True)
        if bipart < best_bipartiteness:
            best_bipartiteness = bipart
            best_R = R
            best_L = L

        # Increment the value of the truncation parameter
        xi_t = 2 * xi_t

    return best_L, best_R, best_bipartiteness


def ms_almost_bipartite(G, starting_vertex, alpha=0.1, epsilon=1e-5, max_iterations=1000000):
    """
    Find an almost-bipartite set close to starting_vertex, using the double cover pagerank algorithm given by
    Macgregor and Sun.

    :param G: a GraphLocal object on which to perform the algorithm.
    :param starting_vertex: the vertex id at which to start the algorithm.
    :param alpha: the alpha parameter for the approximate pagerank computation
    :param epsilon: the epsilon parameter for the approximate pagerank computation
    :param max_iterations: the maximum number of iterations for the pagerank computation
    :return: A tuple containing:
      - L - the vertices in the left set
      - R - the vertices in the right set
      - bipart - the bipartiteness of the resulting set.
    """
    # If alpha is equal to 0, then this algorithm is not defined
    if alpha == 0:
        raise AssertionError("Parameter alpha cannot be 0 for double cover pagerank algorithm.")

    # First, compute the approximate pagerank on the double cover of the graph
    # The result is simplified before being returned.
    n = G.adjacency_matrix.shape[0]

    if G.weighted:
        x_ind_1, x_ind_2, values_1, values_2 = dcpagerank_weighted_cpp(n, G.ai, G.aj, G.adjacency_matrix.data,
                                                                       alpha, epsilon, [starting_vertex],
                                                                       max_iterations, xlength=n)
    else:
        x_ind_1, x_ind_2, values_1, values_2 = dcpagerank_cpp(n, G.ai, G.aj, alpha, epsilon, [starting_vertex],
                                                              max_iterations, xlength=n)

    # Perform the sweep set procedure on the pagerank vector on the double cover of the graph.
    sweepset_dc, conductance_dc = sweep_cut_dc(G, x_ind_1, x_ind_2, values_1, values_2, normalise_by_degree=True)

    # Split the sweep set into L and R
    L = []
    R = []
    for index in sweepset_dc:
        if index < n:
            L.append(index)
        else:
            R.append(index - n)

    return L, R, conductance_dc


def ms_evo_cut_directed(G, starting_vertices, target_phi, T=None, debug=False):
    """
    An implementation of the EvoCutDirected algorithm. This implementation is not fully optimised.
    The graph is assumed to be unweighted.

    :param G: the semi-double cover of the directed graph on which to operate
    :param starting_vertices: a list of starting vertices
    :param target_phi: the flow ratio of the target sets
    :param T: Optionally specify the internal parameter to use instead of the one computed from phi
    :return: the returned clusters L and R, as vertex indices on the original graph, along with the cut imbalance and
    flow ratio
    """
    # Compute the value of T to use
    if T is None:
        T = max(2, math.floor(1 / (100 * (target_phi ** (2/3)))))

    if debug:
        print(f"T: {T}")

    # Get the adjacency matrix of the graph
    A = G.adjacency_matrix

    # Compute the evolving set process for T steps.
    # S will be the current evolving set.
    # X will be the position of the random walk particle
    S = set(starting_vertices)
    probabilities = [G.d[v] / G.volume(list(S)) for v in S]
    X = npr.choice(list(S), p=probabilities)

    # Define the probability function from a vertex to a set
    def p(vert, evolv_set):
        """Get the probability of moving from vert to a vertex inside evolv_set"""
        d_vert = G.d[vert]
        w_vert_S = 0
        for v in G.neighbors(vert):
            if v in evolv_set:
                w_vert_S += A[vert, v]
        if vert in evolv_set:
            return 0.5 + (0.5 * w_vert_S / d_vert)
        else:
            return 0.5 * w_vert_S / d_vert

    for t in range(T):
        if debug:
            print(f"Time {t}; S = {S}")
        # Choose the next location for the random walk particle
        # This is GenerateSample step 1(a)
        X_neighbours = G.neighbors(X)
        X_degree = G.d[X]
        probabilities = [A[X, v] / X_degree for v in X_neighbours]
        X = npr.choice(X_neighbours, p=probabilities)
        if debug:
            print(f"X_(t+1) = {X}")

        # Choose the value of Z
        # This is GenerateSample step 1(b)
        Z = random.uniform(0, p(X, S))
        if debug:
            print(f"Z = {Z}")

        # Update the evolving set S
        # this is GenerateSample step 1(c)
        S_new = set()
        checked = set()  # keep track of which vertices have been checked already
        for v in S:
            if debug:
                print(f"Examining vertex {v}")
            # Check whether v is still inside S
            checked.add(v)
            if p(v, S) >= Z:
                if debug:
                    print(f"Adding {v} to S.")
                S_new.add(v)

            # Check each neighbour of S
            # Note that this is not efficient
            for u in G.neighbors(v):
                if u not in checked:
                    checked.add(u)
                    if p(u, S) >= Z:
                        if debug:
                            print(f"Adding neighbour {u} to S.")
                        S_new.add(u)
        S = S_new

    # Return the left and right sets
    n = int(G.adjacency_matrix.shape[0] / 2)
    if debug:
        print(f"n = {n}")
    L = []
    R = []
    L_other = []
    R_other = []
    for v in S:
        if debug:
            print(f"Processing {v}")
        if v < n:
            if (v + n) not in S:
                L.append(v)
                L_other.append(v + n)
        else:
            if (v - n) not in S:
                R.append(v)
                R_other.append(v - n)

    # If either cluster is empty, return
    if len(L) == 0 or len(R) == 0:
        return L, R_other, 1, 1

    # Compute the cut imbalance
    w_L_R = G.compute_weight(L, R)
    w_R_L = G.compute_weight(R_other, L_other)
    CI = (1/2) * abs((w_L_R - w_R_L)/(w_L_R + w_R_L))

    # Compute the flow ratio
    FR = G.compute_conductance(L + R)
    # w_L_R = G.compute_weight(L, R)
    # vol_out_L = G.volume(L)
    # vol_in_R = G.volume(R)
    # FR = 1 - (2 * w_L_R) / (vol_out_L + vol_in_R)

    return L, R_other, CI, FR


def bipart_cheeger_cut(G):
    """Find the almost-bipartite set given by a sweep-set operation on the top eigenvector of the normalised graph
    laplacian matrix. See [Trevisan 2012] for details.

    :return: A tuple containing:
        - L - the vertices in the left set
        - R - the vertices in the right set
        - bipart - the bipartiteness of the resulting set.
    """
    # Get the number of vertices in the graph
    n = G.adjacency_matrix.shape[0]

    # Find the top eigenvector of the graph laplacian matrix.
    top_eigvec, _ = eig_nL(G, find_top_eigs=True)

    # Perform the sweep cut and return
    return sweep_cut_dc_from_signed_vec(G, range(n), top_eigvec, normalise_by_degree=True)


def clsz_clusters(A, k):
    """
    Given a hermitian adjacency matrix, compute the clusters given by the CLSZ algorithm

    Parameters
    ----------
    A - the hermitian adjacency matrix of the graph
    k - the number of clusters to look for

    Returns
    -------
    A partitioning of the vertices into clusters (as a list of lists).
    """
    eigenvalues, eigenvectors = sp.sparse.linalg.eigsh(A, k=int(2 * math.floor(k / 2)))
    p = eigenvectors @ eigenvectors.transpose()
    input_to_kmeans = np.block([[np.real(p), np.imag(p)]])
    kmeans = KMeans(n_clusters=k).fit(input_to_kmeans)
    return kmeans.labels_
