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


def create_locally_bipartite_graph(filename, n1, n2, p1, q1, p2, q2):
    """
    Create a locally bipartite graph from the stochastic block model.
    :param filename: the edgelist file to save the graph to
    :param n1: the number of vertices in the two smaller clusters
    :param n2: the number of vertices in the large cluster
    :param p1: the probability of an edge inside the smaller clusters
    :param q1: the probability of an edge between the smaller clusters
    :param p2: the probability of an edge inside the larger cluster
    :param q2: the probability of an edge between the smaller clusters and the larger cluster
    :return: nothing
    """
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
