"""
This file provides methods for constructing graphs from the stochastic block model. The graphs can be saved as edgelist
files for later loading.
"""
import random
import math
import numpy as np


def idx_from_cluster_idx(cluster_idx, idx_in_cluster, cluster_size):
    """
    Helper function for stochastic block models.

    Given a cluster number and the index of a vertex inside the cluster, get the index of this vertex in the entire
    graph.

    Everything is 0-indexed.

    :param cluster_idx: the index of the cluster containing the vertex.
    :param idx_in_cluster: the index of the vertex within the cluster
    :param cluster_size: the size of each cluster
    :return: the index of the vertex in the entire graph
    """
    return (cluster_size * cluster_idx) + idx_in_cluster


def create_block_path_graph(filename, n, k, p, q):
    """
    Generates a block path graph from the stochastic block model and saves it as a tab-seperated edgelist to the
    specified file.

    :param filename: The edgelist filename to save the graph to.
    :param n: The number of vertices in each cluster in the graph.
    :param k: The number of clusters
    :param p: The probability of each edge inside a cluster
    :param q: The probability of each edge between neighbouring clusters.
    :return: Nothing
    """
    with open(filename, 'w') as output_f:
        for cluster_idx in range(k):
            # For each pair of vertices in this cluster, add an edge between them if appropriate.
            for idx_1 in range(n):
                for idx_2 in range(idx_1 + 1, n):
                    if random.random() < p:
                        output_f.write(f"{idx_from_cluster_idx(cluster_idx, idx_1, n)}\t"
                                       f"{idx_from_cluster_idx(cluster_idx, idx_2, n)}\n")

            # If this is not the last cluster in the path, add edges to the next cluster
            if cluster_idx < k - 1:
                for idx_1 in range(n):
                    for idx_2 in range(n):
                        if random.random() < q:
                            output_f.write(f"{idx_from_cluster_idx(cluster_idx, idx_1, n)}\t"
                                           f"{idx_from_cluster_idx(cluster_idx + 1, idx_2, n)}\n")


def get_pos_for_sbm_cycle(n, k):
    """Generate positions for a sbm cycle graph"""
    cluster_centers_x = []
    cluster_centers_y = []
    for cluster_idx in range(k):
        a = (cluster_idx / k) * 2 * math.pi
        cluster_centers_x.append(math.cos(a))
        cluster_centers_y.append(math.sin(a))

    cluster_radius = 0.75 * 0.5 * math.sqrt(abs(cluster_centers_x[0] - cluster_centers_x[1]) ** 2 +
                                            abs(cluster_centers_y[0] - cluster_centers_y[1]) ** 2)

    pos = np.zeros((n * k, 2))

    overall_idx = 0
    for cluster_idx in range(k):
        for v_idx in range(n):
            cluster_angle = random.random() * 2 * math.pi
            this_r = cluster_radius * math.sqrt(random.random())
            pos[overall_idx, 0] = cluster_centers_x[cluster_idx] + (this_r * math.cos(cluster_angle))
            pos[overall_idx, 1] = cluster_centers_y[cluster_idx] + (this_r * math.sin(cluster_angle))
            overall_idx += 1

    return pos


def create_block_cycle_graph(filename, n, k, p, q):
    """
    Generates a block cycle graph from the stochastic block model and saves it as a tab-seperated edgelist to the
    specified file.

    :param filename: The edgelist filename to save the graph to.
    :param n: The number of vertices in each cluster in the graph.
    :param k: The number of clusters
    :param p: The probability of each edge inside a cluster
    :param q: The probability of each edge between neighbouring clusters.
    :return: Nothing
    """
    # This doesn't make sense if k is less than 3
    if k < 3:
        raise AssertionError("Cannot create cycle graph with k < 3")

    # First, create a path graph
    create_block_path_graph(filename, n, k, p, q)

    # Then, add the edges between the first and last clusters
    with open(filename, 'a') as output_f:
        # If this is not the last cluster in the path, add edges to the next cluster
        for idx_1 in range(n):
            for idx_2 in range(n):
                if random.random() < q:
                    output_f.write(f"{idx_from_cluster_idx(0, idx_1, n)}\t"
                                   f"{idx_from_cluster_idx(k - 1, idx_2, n)}\n")


def create_locally_bipartite_graph(filename, n1, n2, degree, prop_bipart_crossing, prop_other_crossing):
    """
    Create a graph with a locally almost-bipartite component.

    :param filename: the filename to save the graph to
    :param n1: the number of vertices in each half of the almost-bipartite component
    :param n2: the number of vertices in the rest of the graph
    :param p1: the probability of an edge inside one half of the bipartite component
    :param p1: the probability of an edge between the bipartite component and the rest of the graph
    :param q: the probability of an edge between the halves of the bipartite component.
    :return: Nothing
    """
    # If degree is large enough, then 'cheat' slightly when generating the graph
    if (1 - prop_bipart_crossing) * degree > 2:
        # This procedure may create duplicate edges. These will be ignored when read in as a GraphLocal object.
        with open(filename, 'w') as output_f:
            # For a given vertex inside the bipartite component, compute its edges
            for cluster_idx in [0, 1]:
                for idx_1 in range(n1):
                    bipartite_crossing_edges = [idx_from_cluster_idx(1 - cluster_idx, idx, n1) for idx in
                                                random.sample(range(n1), round(0.5 * prop_bipart_crossing * degree))]
                    other_crossing_edges = [idx + (2 * n1) for idx in
                                            random.sample(range(n2), round(prop_other_crossing * degree))]
                    own_cluster_edges = [idx_from_cluster_idx(cluster_idx, idx, n1) for idx in
                                         random.sample(range(n1),
                                                       round(0.5 * (1 - prop_bipart_crossing - prop_other_crossing) * degree))]
                    own_idx = idx_from_cluster_idx(cluster_idx, idx_1, n1)
                    for other_vertex in bipartite_crossing_edges + other_crossing_edges + own_cluster_edges:
                        if other_vertex != own_idx:
                            output_f.write(f"{own_idx}\t{other_vertex}\n")

            # For a given vertex inside the giant component, compute its edges
            for idx_1 in range(n2):
                own_cluster_edges = [(2 * n1) + idx for idx in
                                     random.sample(range(n2), round(0.5 * (1 - (2 * n1 * prop_other_crossing) / n2) * degree))]
                own_idx = (2 * n1) + idx_1
                for other_vertex in own_cluster_edges:
                    if other_vertex != own_idx:
                        output_f.write(f"{own_idx}\t{other_vertex}\n")

    else:
        # Generate with the 'true' SBM process
        prob_inside_bipart_cluster = (1 - prop_bipart_crossing - prop_other_crossing) * (degree / n1)
        prob_between_bipart_cluster = prop_bipart_crossing * (degree / n1)
        prob_between_other_cluster = prop_other_crossing * (degree / n2)
        prob_inside_other_cluster = (degree - prob_between_other_cluster * n1) / n2
        total_necessary_steps = (n1 * n1) + (n1 * n1) + (2 * n1 * n2) + (0.5 * n2 * n2)
        with open(filename, 'w') as output_f:
            # Create the edges inside each half of the bipartite component
            for cluster_idx in [0, 1]:
                for idx_1 in range(n1):
                    for idx_2 in range(idx_1 + 1, n1):
                        # steps_taken += 1
                        # if steps_taken % gaps == 0:
                        #     print(f"Steps taken: {int(steps_taken/gaps)}/{int(total_necessary_steps/gaps)}")
                        if random.random() < prob_inside_bipart_cluster:
                            output_f.write(f"{idx_from_cluster_idx(cluster_idx, idx_1, n1)}\t"
                                           f"{idx_from_cluster_idx(cluster_idx, idx_2, n1)}\n")

            # Create edges between the two halves of the bipartite component.
            for idx_1 in range(n1):
                for idx_2 in range(n1):
                    # steps_taken += 1
                    # if steps_taken % 100000 == 0:
                    #     print(f"Steps taken: {steps_taken}/{total_necessary_steps}")
                    if random.random() < prob_between_bipart_cluster:
                        output_f.write(f"{idx_from_cluster_idx(0, idx_1, n1)}\t"
                                       f"{idx_from_cluster_idx(1, idx_2, n1)}\n")

            # Create edges from the bipartite component to the rest of the graph
            for idx_1 in range(2 * n1):
                for idx_2 in range(2 * n1, (2 * n1) + n2):
                    # steps_taken += 1
                    # if steps_taken % 100000 == 0:
                    #     print(f"Steps taken: {steps_taken}/{total_necessary_steps}")
                    if random.random() < prob_between_other_cluster:
                        output_f.write(f"{idx_1}\t{idx_2}\n")

            # Create edges inside the rest of the graph
            for idx_1 in range(2 * n1, (2 * n1) + n2):
                for idx_2 in range(idx_1, (2 * n1) + n2):
                    # steps_taken += 1
                    # if steps_taken % 100000 == 0:
                    #     print(f"Steps taken: {steps_taken}/{total_necessary_steps}")
                    if random.random() < prob_inside_other_cluster:
                        output_f.write(f"{idx_1}\t{idx_2}\n")


def pure_locally_bipartite_graph(filename, n1, n2, p1, q1, p2, q2):
    """
    Create a locally bipartite graph from the stochastic block model using the SBM parameters directly.
    :param filename: the edgelist file to save the graph to
    :param n1: the number of vertices in the two smaller clusters
    :param n2: the number of vertices in the large cluster
    :param p1: the probability of an edge inside the smaller clusters
    :param q1: the probability of an edge between the smaller clusters
    :param p2: the probability of an edge inside the larger cluster
    :param q2: the probability of an edge between the smaller clusters and the larger cluster
    :return: nothing
    """
    # Generate with the 'true' SBM process
    with open(filename, 'w') as output_f:
        # Create the edges inside each half of the bipartite component
        for cluster_idx in [0, 1]:
            for idx_1 in range(n1):
                for idx_2 in range(idx_1 + 1, n1):
                    # steps_taken += 1
                    # if steps_taken % gaps == 0:
                    #     print(f"Steps taken: {int(steps_taken/gaps)}/{int(total_necessary_steps/gaps)}")
                    if random.random() < p1:
                        output_f.write(f"{idx_from_cluster_idx(cluster_idx, idx_1, n1)}\t"
                                       f"{idx_from_cluster_idx(cluster_idx, idx_2, n1)}\n")

        # Create edges between the two halves of the bipartite component.
        for idx_1 in range(n1):
            for idx_2 in range(n1):
                # steps_taken += 1
                # if steps_taken % 100000 == 0:
                #     print(f"Steps taken: {steps_taken}/{total_necessary_steps}")
                if random.random() < q1:
                    output_f.write(f"{idx_from_cluster_idx(0, idx_1, n1)}\t"
                                   f"{idx_from_cluster_idx(1, idx_2, n1)}\n")

        # Create edges from the bipartite component to the rest of the graph
        for idx_1 in range(2 * n1):
            for idx_2 in range(2 * n1, (2 * n1) + n2):
                # steps_taken += 1
                # if steps_taken % 100000 == 0:
                #     print(f"Steps taken: {steps_taken}/{total_necessary_steps}")
                if random.random() < q2:
                    output_f.write(f"{idx_1}\t{idx_2}\n")

        # Create edges inside the rest of the graph
        for idx_1 in range(2 * n1, (2 * n1) + n2):
            for idx_2 in range(idx_1, (2 * n1) + n2):
                # steps_taken += 1
                # if steps_taken % 100000 == 0:
                #     print(f"Steps taken: {steps_taken}/{total_necessary_steps}")
                if random.random() < p2:
                    output_f.write(f"{idx_1}\t{idx_2}\n")


def get_pos_for_locally_bipartite_graph(n1, n2):
    """
    Generate positions for drawing the locally bipartite graph.
    :param n1: the number of vertices in each half of the almost-bipartite component
    :param n2: the number of vertices in the rest of the graph
    :return:
    """
    cluster_centers_x = [-0.75, 0.75, 0]
    cluster_centers_y = [1, 1, -0.5]
    cluster_radii = [0.5, 0.5, 0.8]

    pos = np.zeros((2 * n1 + n2, 2))

    overall_idx = 0
    for cluster_idx in range(2):
        for v_idx in range(n1):
            cluster_angle = random.random() * 2 * math.pi
            this_r = cluster_radii[cluster_idx] * math.sqrt(random.random())
            pos[overall_idx, 0] = cluster_centers_x[cluster_idx] + (this_r * math.cos(cluster_angle))
            pos[overall_idx, 1] = cluster_centers_y[cluster_idx] + (this_r * math.sin(cluster_angle))
            overall_idx += 1

    for v_idx in range(n2):
        cluster_angle = random.random() * 2 * math.pi
        this_r = cluster_radii[2] * math.sqrt(random.random())
        pos[overall_idx, 0] = cluster_centers_x[2] + (this_r * math.cos(cluster_angle))
        pos[overall_idx, 1] = cluster_centers_y[2] + (this_r * math.sin(cluster_angle))
        overall_idx += 1

    return pos


def create_local_flow_graph(filename, n1, n2, p1, q1, p2, q2, f):
    """
    Create a graph with a local flow structure from the directed stochastic block model.
    :param filename: the edgelist file to save the graph to
    :param n1: the number of vertices in the two smaller clusters
    :param n2: the number of vertices in the large cluster
    :param p1: the probability of an edge inside the smaller clusters
    :param q1: the probability of an edge between the smaller clusters
    :param p2: the probability of an edge inside the larger cluster
    :param q2: the probability of an edge between the smaller clusters and the larger cluster
    :param f: the flow probability matrix as in Cucuringu et al. The clusters are ordered small, small, big.
    :return: nothing
    """
    random.seed()
    with open(filename, 'w') as output_f:
        # Create the edges inside each half of the bipartite component
        for cluster_idx in [0, 1]:
            for idx_1 in range(n1):
                for idx_2 in range(idx_1 + 1, n1):
                    # steps_taken += 1
                    # if steps_taken % gaps == 0:
                    #     print(f"Steps taken: {int(steps_taken/gaps)}/{int(total_necessary_steps/gaps)}")
                    if random.random() < p1:
                        if random.random() < f[cluster_idx][cluster_idx]:
                            output_f.write(f"{idx_from_cluster_idx(cluster_idx, idx_1, n1)}\t"
                                           f"{idx_from_cluster_idx(cluster_idx, idx_2, n1)}\n")
                        else:
                            output_f.write(f"{idx_from_cluster_idx(cluster_idx, idx_2, n1)}\t"
                                           f"{idx_from_cluster_idx(cluster_idx, idx_1, n1)}\n")

        # Create edges between the two halves of the bipartite component.
        for idx_1 in range(n1):
            for idx_2 in range(n1):
                # steps_taken += 1
                # if steps_taken % 100000 == 0:
                #     print(f"Steps taken: {steps_taken}/{total_necessary_steps}")
                if random.random() < q1:
                    if random.random() < f[0][1]:
                        output_f.write(f"{idx_from_cluster_idx(0, idx_1, n1)}\t"
                                       f"{idx_from_cluster_idx(1, idx_2, n1)}\n")
                    else:
                        output_f.write(f"{idx_from_cluster_idx(1, idx_2, n1)}\t"
                                       f"{idx_from_cluster_idx(0, idx_1, n1)}\n")

        # Create edges from the bipartite component to the rest of the graph
        for cluster_idx in [0, 1]:
            for idx_1 in range(n1):
                for idx_2 in range(2 * n1, (2 * n1) + n2):
                    # steps_taken += 1
                    # if steps_taken % 100000 == 0:
                    #     print(f"Steps taken: {steps_taken}/{total_necessary_steps}")
                    if random.random() < q2:
                        if random.random() < f[cluster_idx][2]:
                            output_f.write(f"{idx_from_cluster_idx(cluster_idx, idx_1, n1)}\t{idx_2}\n")
                        else:
                            output_f.write(f"{idx_2}\t{idx_from_cluster_idx(cluster_idx, idx_1, n1)}\n")

        # Create edges inside the rest of the graph
        for idx_1 in range(2 * n1, (2 * n1) + n2):
            for idx_2 in range(idx_1, (2 * n1) + n2):
                # steps_taken += 1
                # if steps_taken % 100000 == 0:
                #     print(f"Steps taken: {steps_taken}/{total_necessary_steps}")
                if random.random() < p2:
                    if random.random() < f[2][2]:
                        output_f.write(f"{idx_1}\t{idx_2}\n")
                    else:
                        output_f.write(f"{idx_2}\t{idx_1}\n")


def create_triangle_flow_graph(filename, n, p, q, eta):
    """
    Create a directed graph with a net flow around a triangle of clusters. Terminology is from the 'cyclic block model'
    of CLSZ.

    Parameters
    ----------
    filename - the name of the file to save the graph to
    n - the number of vertices in each cluster
    p - the probability of an edge inside a cluster
    q - the probability of an edge between clusters
    eta  - the probability of an edge between clusters matching the clockwise 'flow'

    Returns
    -------
    Nothing, saves the graph as an edgelist to the filename provided.
    """
    create_local_flow_graph(filename, n, n, p, q, p, q, [[0.5, eta, 1-eta], [1-eta, 0.5, eta], [eta, 1-eta, 0.5]])
