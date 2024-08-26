"""
An example script showing how to run the code.
"""
import stag.random
import stag.graphio
import localgraphclustering as lgc


def main():
    # Create an SBM graph with n_1 = 1000, p_1 = 0.001, q_1 = 0.002
    G = stag.random.sbm(2000, 2, 0.004, 0.008)
    temp_edgelist_filename = "sbm_graph.edgelist"
    stag.graphio.save_edgelist(G, temp_edgelist_filename)

    # Load the created graph as a GraphLocal object
    graph_local = lgc.GraphLocal(filename=temp_edgelist_filename,
                                 separator=' ')

    # Run the local bipartite clusters algorithm.
    starting_vertex = 1
    L, R, bipartiteness = lgc.find_bipartite_clusters.local_bipartite_dc(
        graph_local, starting_vertex, alpha=0.5, epsilon=4e-7
    )
    print(f"Cluster One: {sorted(L)}")
    print(f"Cluster Two: {sorted(R)}")
    print(f"Bipartiteness: {bipartiteness:.3f}")


if __name__ == '__main__':
    main()
