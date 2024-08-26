"""
An example script showing how to run the code.
"""
import scipy
import stag.graph
import stag.random
import stag.graphio
import stag.cluster
import localgraphclustering as lgc


def simplify(num_g_vertices: int, sparse_vector):
    """
    Given a sparse vector (presumably from an approximate pagerank calculation on the double cover),
    and the number of vertices in the original graph, compute the 'simplified' approximate pagerank vector.
    """
    # Initialise the new sparse vector
    new_vector = scipy.sparse.lil_matrix((2 * num_g_vertices, 1))

    # Iterate through the entries in the matrix
    for i in range(min(num_g_vertices, sparse_vector.shape[0] - num_g_vertices)):
        if sparse_vector[i, 0] > sparse_vector[i + num_g_vertices, 0]:
            new_vector[i, 0] = sparse_vector[i, 0] - sparse_vector[i + num_g_vertices, 0]
        elif sparse_vector[i + num_g_vertices, 0] > sparse_vector[i, 0]:
            new_vector[i + num_g_vertices, 0] = sparse_vector[i + num_g_vertices, 0] - sparse_vector[i, 0]

    return new_vector.tocsc()


def local_bipart_dc(g: stag.graph.Graph, start_vertex: int, alpha: float, eps: float):
    """
    An implementation of the local_bipart_dc algorithm using the STAG library.
    """
    # Now, we construct the double cover graph of g
    adj_mat = g.adjacency().to_scipy()
    identity = scipy.sparse.csc_matrix((g.number_of_vertices(), g.number_of_vertices()))
    double_cover_adj = scipy.sparse.bmat([[identity, adj_mat], [adj_mat, identity]])
    h = stag.graph.Graph(double_cover_adj)

    # Run the approximate pagerank on the double cover graph
    seed_vector = scipy.sparse.lil_matrix((h.number_of_vertices(), 1))
    seed_vector[start_vertex, 0] = 1
    p, r = stag.cluster.approximate_pagerank(h, seed_vector.tocsc(), alpha, eps)

    # Compute the simplified pagerank vector
    p_simplified = simplify(g.number_of_vertices(), p.to_scipy())

    # Compute the sweep set in the double cover
    sweep_set = stag.cluster.sweep_set_conductance(h, p_simplified)
    bipartiteness = stag.cluster.conductance(h, sweep_set)

    # Split the returned vertices into those in the same cluster as the seed, and others.
    this_cluster = [i for i in sweep_set if i < g.number_of_vertices()]
    that_cluster = [i - g.number_of_vertices() for i in sweep_set if i >= g.number_of_vertices()]
    return this_cluster, that_cluster, bipartiteness


def main():
    # Create an SBM graph with n_1 = 1000, p_1 = 0.001, q_1 = 0.002
    G = stag.random.sbm(2000, 2, 0.004, 0.008)

    # Run the STAG implementation of the bipartite clusters algorithm.
    starting_vertex = 1
    L, R, bipartiteness = local_bipart_dc(
        G, starting_vertex, 0.5, 4e-7
    )
    print(f"Cluster One: {sorted(L)}")
    print(f"Cluster Two: {sorted(R)}")
    print(f"Bipartiteness: {bipartiteness:.3f}")

    # Run the original implementation of the bipartite clusters algorithm

    # Load the created graph as a GraphLocal object
    temp_edgelist_filename = "sbm_graph.edgelist"
    stag.graphio.save_edgelist(G, temp_edgelist_filename)
    graph_local = lgc.GraphLocal(filename=temp_edgelist_filename,
                                 separator=' ')

    # Run the local bipartite clusters algorithm.
    L, R, bipartiteness = lgc.find_bipartite_clusters.local_bipartite_dc(
        graph_local, starting_vertex, alpha=0.5, epsilon=4e-7
    )
    print(f"Cluster One: {sorted(L)}")
    print(f"Cluster Two: {sorted(R)}")
    print(f"Bipartiteness: {bipartiteness:.3f}")




if __name__ == '__main__':
    main()
