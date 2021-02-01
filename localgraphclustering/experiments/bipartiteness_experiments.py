"""
Provides methods for running experiments for finding sets with low bipartiteness.
These experiments will compare the following three algorithms:
  - double cover pagerank
  - Li & Peng bipartitness algorithm
  - Andersen densest subset algorithm
"""
import numpy as np
from matplotlib import pyplot as plt
import time
import localgraphclustering as lgc
import multiprocessing
import sklearn as skl
from . import process_results, migrationdata, migration_visualisation
import math
import gc
import random

import warnings
warnings.filterwarnings("ignore")


def load_slashdot_graph(negative=False, semi_double_cover=False, largest_component=True):
    """Returns a GraphLocal object representing the unsigned slashdot graph used in our experiments.
    We always extract the largest connected component of the graph."""
    if negative:
        G = lgc.GraphLocal(filename="datasets/slashdot/negative-unsigned-slashdot.edgelist",
                           semi_double_cover=semi_double_cover)
    else:
        G = lgc.GraphLocal(filename="datasets/slashdot/unsigned-slashdot.edgelist",
                           semi_double_cover=semi_double_cover)
    if largest_component:
        return G.largest_component()
    else:
        return G


def load_reddit_graph():
    """Load the graph given by the negative edges in the reddit dataset.
    Returns the GraphLocal version of the graph, and a dictionary giving the mapping from vertex ids to
    """
    G = lgc.GraphLocal(filename=f"datasets/reddit/reddit-negative.edgelist")
    vertex_subreddit_dict = {}
    subreddit_vertex_dict = {}
    with open("datasets/reddit/reddit-subreddit-vertex.csv", 'r') as f_in:
        for line in f_in.readlines():
            clean_line = line.strip().split('\t')
            vertex_subreddit_dict[int(clean_line[0])] = clean_line[1]
            subreddit_vertex_dict[clean_line[1]] = int(clean_line[0])
    return G, vertex_subreddit_dict, subreddit_vertex_dict


def load_mid_graph(start_year, end_year):
    """Load the graph given by the military conflict dataset.
    Returns the GraphLocal version of the graph, and a dictionary giving the mapping from vertex ids to countries.
    """
    G = lgc.GraphLocal(filename=f"datasets/mid/dyadic_mid_{start_year}_{end_year}.edgelist")
    vertex_country_dict = {}
    country_vertex_dict = {}
    with open("datasets/mid/COW country codes.csv", 'r') as f_in:
        for i, line in enumerate(f_in.readlines()):
            if i > 0:
                clean_line = line.strip().split(',')
                vertex_country_dict[int(clean_line[1])] = clean_line[2]
                country_vertex_dict[clean_line[2]] = int(clean_line[1])
    return G, vertex_country_dict, country_vertex_dict


def load_sbm_graph(size=1, d=20, prop=None):
    """Returns a GraphLocal object representing the sbm-generated path graph.

    :param size: If loading the cycle graph, optionally specify the cluster size
    :param d: The expected degree of the graph
    :param prop: If specified, gives the graph generated with the specified proportion of edges crossing the cut
    """
    if prop is None:
        return lgc.GraphLocal(filename=f"datasets/sbm/local_cluster_size_{size}00_{size}000_degree_{d}.edgelist")
    else:
        return lgc.GraphLocal(
            filename=f"datasets/sbm/local_cluster_size_{size}00_{size}000_degree_{d}_prop_{prop}.edgelist")


def load_pure_sbm_graph(n1, n2, p1, q1, p2, q2):
    """Load an SBM graph with the given parameters."""
    return lgc.GraphLocal(filename=f"datasets/sbm/local_cluster_size_{n1}_{n2}_{p1}_{q1}_{p2}_{q2}.edgelist")


def report_ms_performance(G, s, alpha, epsilon, target_L, target_R, show_output=True):
    """Print the performance of the MS algorithm for the given inputs."""
    start_time = time.clock()
    L, R, bipart = lgc.find_bipartite_clusters.ms_almost_bipartite(G, s, alpha=alpha, epsilon=epsilon)
    total_time_s = time.clock() - start_time
    ari = compute_ari(L, R, target_L, target_R, G.adjacency_matrix.shape[0])
    symdiff = compute_symmetric_difference(L, R, target_L, target_R)
    volume = G.volume(L + R)
    if show_output:
        print(f"\ttime:\t\t\t{total_time_s:.3f}\n\tbipartiteness:\t{bipart:.3f}\n\tvolume:\t\t\t{int(G.volume(L + R))}"
              f"\n\tari:\t\t\t{ari:.3f}\n\tsymdiff:\t\t{symdiff:.3f}")
        print()
    return total_time_s, bipart, volume, ari, symdiff


def report_lp_performance(G, s, T, xi_0, target_L, target_R, plot=False, show_output=True):
    """Print the performance of the LP algorithm for the given inputs."""
    start_time = time.clock()
    L, R, bipart = lgc.find_bipartite_clusters.lp_almost_bipartite(G, s, T=T, xi_0=xi_0, debug=False)
    total_time_s = time.clock() - start_time
    ari = compute_ari(L, R, target_L, target_R, G.adjacency_matrix.shape[0])
    symdiff = compute_symmetric_difference(L, R, target_L, target_R)
    volume = G.volume(L + R)
    if show_output:
        print(f"\ttime:\t\t\t{total_time_s:.3f}\n\tbipartiteness:\t{bipart:.3f}\n\tvolume:\t\t\t{int(G.volume(L + R))}"
              f"\n\tari:\t\t\t{ari:.3f}\n\tsymdiff:\t\t{symdiff:.3f}")
        print()

    if plot:
        size = int(G.adjacency_matrix.shape[0] / 800)
        not_S = list(set(range(size * 100 * 8)) - set(L) - set(R))
        G.draw_groups([L, R, not_S], pos=lgc.get_pos_for_sbm_cycle(size * 100, 8))
        plt.show()

    return total_time_s, bipart, volume, ari, symdiff


def compare_bipartieness_algs(G, s, show_clusters=False, skip_cheeger=False):
    """
    Given a graph and a starting vertex, run several bipartiteness algorithms and compare their performance.

    :param G: the graph on which to operate
    :param s: the index of the starting vertex
    :param show_clusters: whether to print out the clusters generated by each method
    :param skip_cheeger: whether to compare with the cheeger cut algorithm
    :return:
    """
    # Bipartiteness Cheeger cut algorithm
    if not skip_cheeger:
        print("Bipartite Cheeger Cut")
        start_time = time.clock()
        L, R, bipart = lgc.find_bipartite_clusters.bipart_cheeger_cut(G)
        total_time_s = time.clock() - start_time
        print(f"\ttime:\t\t\t{total_time_s:.3f}\n\tbipartiteness:\t{bipart:.3f}\n\tvolume:\t\t\t{int(G.volume(L + R))}")
        if show_clusters:
            print(f"\tL:\t\t\t{sorted(L)}\n\tR:\t\t\t{sorted(R)}")
        print()

    # Double cover pagerank algorithm
    print("Double Cover Pagerank")
    start_time = time.clock()
    # L, R, bipart = lgc.find_bipartite_clusters.ms_almost_bipartite(G, s, alpha=0.002, epsilon=1e-6)
    L, R, bipart = lgc.find_bipartite_clusters.ms_almost_bipartite(G, s, alpha=0.5, epsilon=1e-5)
    total_time_s = time.clock() - start_time
    print(f"\ttime:\t\t\t{total_time_s:.3f}\n\tbipartiteness:\t{bipart:.3f}\n\tvolume:\t\t\t{int(G.volume(L + R))}")
    if show_clusters:
        print(f"\tL:\t\t\t{sorted(L)}\n\tR:\t\t\t{sorted(R)}")
    print()

    # Li-Peng truncated power method algorithm
    print("Truncated Power Method")
    start_time = time.clock()
    # L, R, bipart = lgc.find_bipartite_clusters.lp_almost_bipartite(G, s, T=4, eps=0.00001)
    L, R, bipart = lgc.find_bipartite_clusters.lp_almost_bipartite(G, s, T=4, xi_0=1e-5)
    total_time_s = time.clock() - start_time
    print(f"\ttime:\t\t\t{total_time_s:.3f}\n\tbipartiteness:\t{bipart:.3f}\n\tvolume:\t\t\t{int(G.volume(L + R))}")
    if show_clusters:
        print(f"\tL:\t\t\t{sorted(L)}\n\tR:\t\t\t{sorted(R)}")
    print()


def test_ms_parameters(G, s, min_eps_pow, max_eps_pow, min_alph_pow, max_alph_pow, num_eps, num_alph, target_L,
                       target_R, max_time=86400, filename=None, update_results=False, linear=False):
    """
    Given a graph and starting vertex, test the effect of the parameters on the performance of the double cover
    pagerank clustering algorithm. Show some pretty plots.

    :param G: the graph on which to operate.
    :param s: the starting vertex for the algorithm.
    :param max_time: this is the maximum time allowed for each run of the algorithm.
    :param filename: If filename is given, results are saved to file, and graphs are not shown
    :param update_results: If true, then add to the existing experimental results in filename
    :param linear: Whether to use linear scales when searching the parameter space.
    :return: Nothing.
    """
    def algorithm_worker(graph, starting_v, a, e, return_dict):
        """We will run the algorithm in a seperate process to allow us to handle the timeout. The return values
        from the algorithm need to be accessible outside of the child process. This worker method wraps the algorithm
        call and allows the output to be shared between processes."""
        start_time = time.clock()
        left, right, bipartiteness = lgc.find_bipartite_clusters.ms_almost_bipartite(
            graph, starting_v, alpha=a, epsilon=e)
        total_time_s = time.clock() - start_time
        return_dict["L"] = left
        return_dict["R"] = right
        return_dict["bipart"] = bipartiteness
        return_dict["time"] = total_time_s

    fout = None
    if filename is not None and not update_results:
        fout = open(filename, 'w')
        fout.write(f"alpha, epsilon, bipartiteness, ari, symdiff, volume, time\n")
    if filename is not None and update_results:
        fout = open(filename, 'a')

    # Get the size of the graph
    n = G.adjacency_matrix.shape[0]

    # The manager and return_dictionary fill be used for each function call
    process_manager = multiprocessing.Manager()
    output_dict = process_manager.dict()

    # Which values of epsilon to try
    if not linear:
        epsilons = np.logspace(min_eps_pow, max_eps_pow, num=num_eps)
    else:
        epsilons = np.linspace(10 ** min_eps_pow, 10 ** max_eps_pow, num=num_eps)

    # Vary the value of alpha and record the runtime, bipartiteness and volume for the algorithm.
    if not linear:
        alphas = np.logspace(min_alph_pow, max_alph_pow, num=num_alph)
    else:
        alphas = np.linspace(10 ** min_alph_pow, 10 ** max_alph_pow, num=num_alph)

    # Create the grid of plots
    plt_fig, plt_axes = plt.subplots(3, len(epsilons), constrained_layout=True)

    # Collect the data for a scatter plot of all times and bipartitenesses
    all_times = []
    all_biparts = []

    # Keep track of the best sub-0.4 bipartiteness run
    best_time = max_time
    best_bipart = 1
    best_volume = 0
    best_alpha = 0
    best_epsilon = 0

    i = 0
    for plot_row, eps in enumerate(epsilons):
        # Reset the data for this value of epsilon
        # We will store the data only for runs of the algorithm which do not time out
        times = []
        biparts = []
        volumes = []
        good_alphas = []

        for alpha in alphas:
            i += 1
            print(f"Execution {i}/{len(epsilons) * len(alphas)} with alpha = {alpha:.2e} and epsilon = {eps:.2e}")

            # Spawn the child process
            timed_out = False
            child_process = multiprocessing.Process(target=algorithm_worker, args=(G, s, alpha, eps, output_dict))
            child_process.start()

            # Give it the allocated maximum time to return
            child_process.join(max_time)
            if child_process.is_alive():
                # The process did not terminate in time
                child_process.terminate()
                timed_out = True
                print("\t !! TIMED OUT")

            if not timed_out:
                # If the process terminated, get the output values and add them to our plots
                L = output_dict["L"]
                R = output_dict["R"]
                volume = int(G.volume(L + R))
                bipart = output_dict["bipart"]
                time_s = output_dict["time"]

                # Get the scores for the clustering
                ari = compute_ari(L, R, target_L, target_R, n)
                symdiff = compute_symmetric_difference(L, R, target_L, target_R)

                good_alphas.append(alpha)
                times.append(time_s)
                biparts.append(bipart)
                volumes.append(volume)

                if filename is not None:
                    str_to_print = f"{alpha}, {eps}, {bipart}, {ari}, {symdiff}, {volume}, {time_s}\n"
                    print(str_to_print[:-1])
                    fout.write(str_to_print)

                all_times.append(time_s)
                all_biparts.append(bipart)

                # Check whether this is sub 0.4 bipartiteness and remember if so
                if bipart < 0.4 and filename is None:
                    print(f"\tbipartiteness:\t{bipart:.2f}\n\tvolume:\t\t\t{volume}\n\ttime:\t\t\t{time_s:.2f}")
                    if time_s < best_time:
                        best_alpha = alpha
                        best_epsilon = eps
                        best_bipart = bipart
                        best_volume = volume
                        best_time = time_s
            elif filename is not None:
                fout.write(f"{alpha}, {eps}, 1, 0, 1, 0, TIMED_OUT\n")

        # Add the results to the appropriate plot
        plt_axes[0, plot_row].set_title(f"Time (eps: {eps:.0e})")
        plt_axes[0, plot_row].plot(good_alphas, times)
        plt_axes[0, plot_row].set_xscale('log')

        plt_axes[1, plot_row].set_title(f"Bipartiteness (eps: {eps:.0e})")
        plt_axes[1, plot_row].plot(good_alphas, biparts)
        plt_axes[1, plot_row].set_xscale('log')

        plt_axes[2, plot_row].set_title(f"Volume (eps: {eps:.0e})")
        plt_axes[2, plot_row].plot(good_alphas, volumes)
        plt_axes[2, plot_row].set_xscale('log')

    # Display the 'best' parameters
    if filename is None:
        print()
        print("Best Parameters")
        print(f"\talpha:\t\t\t{best_alpha:.2e}\n\tepsilon:\t\t\t{best_epsilon:.2e}\n\tbipartiteness:\t{best_bipart:.2f}"
              f"\n\tvolume:\t\t\t{best_volume}\n\ttime:\t\t\t{best_time:.2f}")

    if filename is None:
        # Show the results
        plt.show()

        # Now, show a scatter plot of all times and bipartitenesses
        plt.scatter(all_times, all_biparts)
        plt.show()

    if filename is not None:
        fout.close()


def test_lp_parameters(G, s, min_T, max_T, min_xi_pow, max_xi_pow, num_xi, target_L, target_R, max_time=86400,
                       filename=None, update_results=False, linear=False):
    """
    Given a graph and starting vertex, test the effect of the parameters on the performance of the double cover
    pagerank clustering algorithm. Show some pretty plots.

    :param G: the graph on which to operate.
    :param s: the starting vertex for the algorithm.
    :param max_time: this is the maximum time allowed for each run of the algorithm.
    :param filename: If filename is given, results are saved to file, and graphs are not shown
    :return: Nothing.
    """
    def algorithm_worker(graph, starting_v, T, xi_0, return_dict):
        """We will run the algorithm in a seperate process to allow us to handle the timeout. The return values
        from the algorithm need to be accessible outside of the child process. This worker method wraps the algorithm
        call and allows the output to be shared between processes."""
        start_time = time.clock()
        left, right, bipartiteness = lgc.find_bipartite_clusters.lp_almost_bipartite(graph, starting_v, T=T, xi_0=xi_0)
        total_time_s = time.clock() - start_time
        return_dict["L"] = left
        return_dict["R"] = right
        return_dict["bipart"] = bipartiteness
        return_dict["time"] = total_time_s

    fout = None
    if filename is not None and not update_results:
        fout = open(filename, 'w')
        fout.write(f"xi_0, T, bipartiteness, ari, symdiff, volume, time\n")
    if filename is not None and update_results:
        fout = open(filename, 'a')

    # The manager and return_dictionary fill be used for each function call
    process_manager = multiprocessing.Manager()
    output_dict = process_manager.dict()

    # Size of the graph
    n = G.adjacency_matrix.shape[0]

    # Which values of T to try
    ts = list(range(min_T, max_T + 1))

    # Vary the value of xi_0 and record the runtime, bipartiteness and volume for the algorithm.
    if linear:
        xis = np.linspace(10 ** min_xi_pow, 10 ** max_xi_pow, num=num_xi)
    else:
        xis = np.logspace(min_xi_pow, max_xi_pow, num=num_xi)

    # Create the grid of plots
    plt_fig, plt_axes = plt.subplots(3, len(ts), constrained_layout=True)

    # Collect the data for a scatter plot of all times and bipartitenesses
    all_times = []
    all_biparts = []

    # Keep track of the best sub-0.4 bipartiteness run
    best_time = max_time
    best_bipart = 1
    best_volume = 0
    best_xi = 0
    best_T = 0

    i = 0
    for plot_row, this_T in enumerate(ts):
        # Reset the data for this value of epsilon
        # We will store the data only for runs of the algorithm which do not time out
        times = []
        biparts = []
        volumes = []
        good_xis = []

        for this_xi in xis:
            i += 1
            print(f"Execution {i}/{len(ts) * len(xis)} with xi_0 = {this_xi:.2e} and T = {this_T}")

            # Spawn the child process
            timed_out = False
            child_process = multiprocessing.Process(target=algorithm_worker, args=(G, s, this_T, this_xi, output_dict))
            child_process.start()

            # Give it the allocated maximum time to return
            child_process.join(max_time)
            if child_process.is_alive():
                # The process did not terminate in time
                child_process.terminate()
                timed_out = True
                print("\t !! TIMED OUT")

            if not timed_out:
                # If the process terminated, get the output values and add them to our plots
                L = output_dict["L"]
                R = output_dict["R"]
                volume = int(G.volume(L + R))
                bipart = output_dict["bipart"]
                time_s = output_dict["time"]

                # Get the scores for the clustering
                ari = compute_ari(L, R, target_L, target_R, n)
                symdiff = compute_symmetric_difference(L, R, target_L, target_R)

                good_xis.append(this_xi)
                times.append(time_s)
                biparts.append(bipart)
                volumes.append(volume)

                if filename is not None:
                    str_to_print = f"{this_xi}, {this_T}, {bipart}, {ari}, {symdiff}, {volume}, {time_s}\n"
                    print(str_to_print[:-1])
                    fout.write(str_to_print)

                all_times.append(time_s)
                all_biparts.append(bipart)

                # Check whether this is sub 0.4 bipartiteness and remember if so
                if bipart < 0.4 and filename is None:
                    print(f"\tbipartiteness:\t{bipart:.2f}\n\tvolume:\t\t\t{volume}\n\ttime:\t\t\t{time_s:.2f}")
                    if time_s < best_time:
                        best_xi = this_xi
                        best_T = this_T
                        best_bipart = bipart
                        best_volume = volume
                        best_time = time_s
            elif filename is not None:
                fout.write(f"{this_xi}, {this_T}, 1, 0, 1, 0, TIMED_OUT\n")

        # Add the results to the appropriate plot
        plt_axes[0, plot_row].set_title(f"Time (eps: {this_T:.0e})")
        plt_axes[0, plot_row].plot(good_xis, times)
        plt_axes[0, plot_row].set_xscale('log')

        plt_axes[1, plot_row].set_title(f"Bipartiteness (eps: {this_T:.0e})")
        plt_axes[1, plot_row].plot(good_xis, biparts)
        plt_axes[1, plot_row].set_xscale('log')

        plt_axes[2, plot_row].set_title(f"Volume (eps: {this_T:.0e})")
        plt_axes[2, plot_row].plot(good_xis, volumes)
        plt_axes[2, plot_row].set_xscale('log')

    # Display the 'best' parameters
    if filename is None:
        print()
        print("Best Parameters")
        print(f"\txi_0:\t\t\t{best_xi:.2e}\n\tT:\t\t\t{best_T:.2e}\n\tbipartiteness:\t{best_bipart:.2f}"
              f"\n\tvolume:\t\t\t{best_volume}\n\ttime:\t\t\t{best_time:.2f}")

    if filename is None:
        # Show the results
        plt.show()

        # Now, show a scatter plot of all times and bipartitenesses
        plt.scatter(all_times, all_biparts)
        plt.show()

    if filename is not None:
        fout.close()


def test_lp_parameters_old(G, s, max_time=86400):
    """
    Given a graph and starting vertex, test the effect of the parameters on the performance of the truncated power
    method clustering algorithm. Show some pretty plots.

    :param G: the graph on which to operate.
    :param s: the starting vertex for the algorithm.
    :param max_time: this is the maximum time allowed for each run of the algorithm.
    :return: Nothing.
    """
    def algorithm_worker(graph, starting_v, steps, xi_start, return_dict):
        """We will run the algorithm in a seperate process to allow us to handle the timeout. The return values
        from the algorithm need to be accessible outside of the child process. This worker method wraps the algorithm
        call and allows the output to be shared between processes."""
        start_time = time.clock()
        left, right, bipartiteness = lgc.find_bipartite_clusters.lp_almost_bipartite(graph, starting_v, T=steps,
                                                                                     xi_0=xi_start)
        total_time_s = time.clock() - start_time
        return_dict["L"] = left
        return_dict["R"] = right
        return_dict["bipart"] = bipartiteness
        return_dict["time"] = total_time_s

    # The manager and return_dictionary fill be used for each function call
    process_manager = multiprocessing.Manager()
    output_dict = process_manager.dict()

    # Which values of T to try
    ts = list(range(2, 6))

    # Values of xi_0 to try
    xis = np.logspace(-6, -3, num=50)

    # Create the grid of plots
    plt_fig, plt_axes = plt.subplots(3, len(ts), constrained_layout=True)

    # keep track of all times and bipartitenesses
    all_times = []
    all_biparts = []

    i = 0
    for plot_row, T in enumerate(ts):
        # Reset the data for this value of epsilon
        # We will store the data only for runs of the algorithm which do not time out
        times = []
        biparts = []
        volumes = []
        good_xis = []

        for xi_0 in xis:
            i += 1
            print(f"Execution {i}/{len(ts) * len(xis)} with xi_0 = {xi_0:.2f} and T = {T}")

            # Spawn the child process
            timed_out = False
            child_process = multiprocessing.Process(target=algorithm_worker, args=(G, s, T, xi_0, output_dict))
            child_process.start()

            # Give it the allocated maximum time to return
            child_process.join(max_time)
            if child_process.is_alive():
                # The process did not terminate in time
                child_process.terminate()
                timed_out = True
                print("\t !! TIMED OUT")

            if not timed_out:
                # If the process terminated, get the output values and add them to our plots
                L = output_dict["L"]
                R = output_dict["R"]

                good_xis.append(xi_0)
                times.append(output_dict["time"])
                biparts.append(output_dict["bipart"])
                volumes.append(int(G.volume(L + R)))

                all_times.append(output_dict["time"])
                all_biparts.append(output_dict["bipart"])

        # Add the results to the appropriate plot
        plt_axes[0, plot_row].set_title(f"Time (T: {T})")
        plt_axes[0, plot_row].plot(good_xis, times)
        plt_axes[0, plot_row].set_xscale('log')

        plt_axes[1, plot_row].set_title(f"Bipartiteness (T: {T})")
        plt_axes[1, plot_row].plot(good_xis, biparts)
        plt_axes[1, plot_row].set_xscale('log')

        plt_axes[2, plot_row].set_title(f"Volume (T: {T})")
        plt_axes[2, plot_row].plot(good_xis, volumes)
        plt_axes[2, plot_row].set_xscale('log')

    # Show the results
    plt.show()

    # Show the time/bipart scatter
    plt.scatter(all_times, all_biparts)
    plt.show()


def compute_ari(L, R, target_L, target_R, n):
    """Compute the adjusted rand score for a bipartiteness clustering."""
    true_labels = np.zeros((n, ))
    true_labels[target_L] = 1
    true_labels[target_R] = 2

    predicted_labels = np.zeros((n, ))
    predicted_labels[L] = 1
    predicted_labels[R] = 2

    return skl.metrics.adjusted_rand_score(true_labels, predicted_labels)


def compute_symmetric_difference(L, R, target_L, target_R):
    """Compute the symmetric difference for the clustering."""
    L = set(L)
    R = set(R)
    target_L = set(target_L)
    target_R = set(target_R)

    # Try both 'ways round' when computing the symmetric difference
    sym_diff_1 = (len(target_L.symmetric_difference(L)) + len(target_R.symmetric_difference(R))) / \
                 (len(target_L.union(L)) + len(target_R.union(R)))
    sym_diff_2 = (len(target_L.symmetric_difference(R)) + len(target_R.symmetric_difference(L))) / \
                 (len(target_L.union(R)) + len(target_R.union(L)))

    return min(sym_diff_1, sym_diff_2)


def get_parameter_lists(size, degree):
    """Get the key parameters from the old runs with the given size and degree.
    Returns lists of pairs of parameters for each algorithm."""
    alphas1, epsilons1, Ts1, xis1, _, _, _, _ = process_results.get_optimal_parameters(
        size=size, degree=degree, objective='ari', minimise=False, plot=False)
    alphas2, epsilons2, Ts2, xis2, _, _, _, _ = process_results.get_optimal_parameters(
        size=size, degree=degree, objective='bipartiteness', minimise=True, plot=False)
    alphas3, epsilons3, Ts3, xis3, _, _, _, _ = process_results.get_optimal_parameters(
        size=size, degree=degree, objective='symdiff', minimise=True, plot=False)
    ms_params = []
    lp_params = []
    for alph_list, eps_list in [(alphas1, epsilons1), (alphas2, epsilons2), (alphas3, epsilons3)]:
        for i, possible_alpha in enumerate(alph_list):
            possible_eps = eps_list[i]
            if (possible_alpha, possible_eps) not in ms_params:
                ms_params.append((possible_alpha, possible_eps))
    for t_list, xi_list in [(Ts1, xis1), (Ts2, xis2), (Ts3, xis3)]:
        for i, possible_t in enumerate(t_list):
            possible_xi = xi_list[i]
            if (possible_t, possible_xi) not in lp_params:
                lp_params.append((possible_t, possible_xi))
    return ms_params, lp_params


def run_ratio_experiment():
    """Run the ratio p1/q1 experiment for the paper."""
    size = 10
    n1 = size * 100
    n2 = size * 1000
    max_time = 3
    p1 = 0.001
    p2 = 0.002
    q2 = 0.0001
    # multiples = [1, 2, 4, 8, 16, 32]
    multiples = [7, 9, 10]

    # Get the lists of parameters that we will use for the experiment
    ms_params, lp_params = get_parameter_lists(10, 20)
    num_param_options = len(lp_params) + len(ms_params)

    num_starts = 10
    for mult_idx, mult in enumerate(multiples):
        # Load the graph
        q1 = mult * p1
        G = load_pure_sbm_graph(n1, n2, p1, q1, p2, q2)
        target_L = list(range(size * 100))
        target_R = list(range(size * 100, size * 200))

        # Choosing the first 5 vertices in the left and right target clusters is equivalent to choosing random vertices
        # since the graph is randomly generated.
        starting_vertices = [1, 2, 3, 4, 5] + \
                            [(size * 100) + 1, (size * 100) + 2, (size * 100) + 3, (size * 100) + 4, (size * 100) + 5]

        for start_num, v_start in enumerate(starting_vertices):
            # Create the output files for the results.
            with open(f"results/ms_{size}00_vertex_{v_start}_mult_{mult}.csv", 'w') as f_out:
                f_out.write(f"alpha, epsilon, bipartiteness, ari, symdiff, volume, time\n")
            with open(f"results/lp_{size}00_vertex_{v_start}_mult_{mult}.csv", 'w') as f_out:
                f_out.write(f"xi_0, T, bipartiteness, ari, symdiff, volume, time\n")

            # For each pair of key parameters, run the algorithm for 9 parameters close around the key one.
            param_option = 0
            for T, xi in lp_params:
                # Free memory
                gc.collect()

                param_option += 1
                print()
                print(f"Mult {mult} ({mult_idx + 1}/{len(multiples)}). Start {start_num + 1}/{num_starts}. Alg LP run "
                      f"{param_option}/{len(lp_params)}. Parameters option {param_option}/{num_param_options}")
                test_lp_parameters(G,                                 # Graph
                                   v_start,                           # starting vertex
                                   max(T - 1, 1),                     # min_T
                                   T + 1,                             # max_T
                                   math.log10(0.4 * xi),              # min xi_0 power
                                   min(math.log10(2 * xi), 1),        # max xi_0 power
                                   3,                                 # num xi_0
                                   target_L, target_R,
                                   update_results=True,
                                   linear=True,
                                   max_time=max_time, filename=f"results/lp_{size}00_vertex_{v_start}_mult_{mult}.csv")
            for alpha, epsilon in ms_params:
                param_option += 1
                print()
                print(f"Mult {mult} ({mult_idx + 1}/{len(multiples)}). Start {start_num + 1}/{num_starts}. Alg MS run "
                      f"{param_option - len(lp_params)}/{len(ms_params)}. "
                      f"Parameters option {param_option}/{num_param_options}")
                test_ms_parameters(G,                                     # Graph
                                   v_start,                               # Starting vertex
                                   math.log10(0.4 * epsilon),             # min_eps_pow
                                   min(math.log10(2 * epsilon), 1),       # max_eps_pow
                                   math.log10(0.4 * alpha),               # min_alph_pow
                                   min(math.log10(2 * alpha), 1),         # max_alph_pow
                                   3,                                     # num_eps
                                   3,                                     # num_alpha
                                   target_L, target_R,
                                   update_results=True,
                                   linear=True,
                                   max_time=max_time, filename=f"results/ms_{size}00_vertex_{v_start}_mult_{mult}.csv")


def fixed_param_experiment(size, d, prop, starting_vertices, lp_eps, show_output=True):
    """Use the parameters given in the formal algorithm descriptions."""
    G = load_sbm_graph(size=size, d=d, prop=prop)
    target_L = list(range(size * 100))
    target_R = list(range(size * 100, size * 200))

    target_volume = d * size * 200
    target_bipartiteness = 1 - prop
    if show_output:
        print(f"Target volume: {target_volume}\nTarget bipartiteness: {target_bipartiteness:.4f}\n")

    # Compute the parameters for the MS algorithm
    beta_hat = math.sqrt(7560 * target_bipartiteness)
    alpha = 20 * target_bipartiteness
    epsilon = 1 / (20 * target_volume)
    if show_output:
        print(f"beta_hat = {beta_hat:.3f}, alpha = {alpha:.3e}, epsilon = {epsilon:.3e}")

    # Compute the parameters for the LP algorithm
    eps = lp_eps
    T = (eps * math.log(1600 * target_volume)) / (6 * target_bipartiteness)
    xi = 1 / ((target_volume ** (1 + eps)) * 800 * T)
    T = round(T)
    if show_output:
        print(f"T = {T}, xi_0 = {xi:.3e}")

    ms_times = []
    ms_biparts = []
    ms_vols = []
    ms_aris = []
    ms_symdiffs = []
    lp_times = []
    lp_biparts = []
    lp_vols = []
    lp_aris = []
    lp_symdiffs = []
    for v_start in starting_vertices:
        print(f"Starting vertex: {v_start}")
        ms_time, ms_bipart, ms_vol, ms_ari, ms_symdiff = report_ms_performance(
            G, v_start, alpha, epsilon, target_L, target_R, show_output=show_output)
        lp_time, lp_bipart, lp_vol, lp_ari, lp_symdiff = report_lp_performance(
            G, v_start, T, xi, target_L, target_R, show_output=show_output)
        ms_times.append(ms_time)
        ms_biparts.append(ms_bipart)
        ms_vols.append(ms_vol)
        ms_aris.append(ms_ari)
        ms_symdiffs.append(ms_symdiff)
        lp_times.append(lp_time)
        lp_biparts.append(lp_bipart)
        lp_vols.append(lp_vol)
        lp_aris.append(lp_ari)
        lp_symdiffs.append(lp_symdiff)

    return [np.mean(x) for x in [
        ms_times, ms_biparts, ms_vols, ms_aris, ms_symdiffs, lp_times, lp_biparts, lp_vols, lp_aris, lp_symdiffs]]


def bipartiteness_experiment():
    """Choose some vertices at random from the slashdot graph and find an almost-bipartite graph around it."""
    # G = load_sbm_graph(cycle=True, size=1)
    # report_LP_performance(G, 1, 3, 0.0002067, plot=True)

    # print("Loading graph...")
    # G = load_slashdot_graph(negative=True)
    # test_lp_parameters(G, 1, max_time=5)

    size = 1000
    d = 30
    max_time = 100
    G = load_sbm_graph(size=size, d=d)
    target_L = list(range(size * 100))
    target_R = list(range(size * 100, size * 200))

    test_lp_parameters(G,    # Graph
                       1,    # starting vertex
                       10,    # min_T
                       12,    # max_T
                       -7,   # min xi_0 power
                       -6,   # max xi_0 power
                       10,  # num xi_0
                       target_L, target_R,
                       update_results=True,
                       linear=True,
                       max_time=max_time, filename=f"results/lp_local_{size}00_degree_{d}.csv")

    # test_ms_parameters(G,    # Graph
    #                    1,    # Starting vertex
    #                    -6,   # min_eps_pow
    #                    -5,   # max_eps_pow
    #                    -3,   # min_alph_pow
    #                    -2,   # max_alph_pow
    #                    20,   # num_eps
    #                    20,  # num_alpha
    #                    target_L, target_R,
    #                    update_results=True,
    #                    linear=True,
    #                    max_time=max_time, filename=f"results/ms_local_{size}00_degree_{d}.csv")

    # test_ms_parameters(G,    # Graph
    #                    1,    # Starting vertex
    #                    -9,   # min_eps_pow
    #                    -5,   # max_eps_pow
    #                    -2,   # min_alph_pow
    #                    -0,   # max_alph_pow
    #                    20,   # num_eps
    #                    10,  # num_alpha
    #                    target_L, target_R,
    #                    update_results=True,
    #                    max_time=max_time, filename=f"results/ms_local_{size}00_degree_{d}.csv")

    # print(f"disconnected? {G.is_disconnected()}")
    # G.draw_groups([list(range(100)), list(range(100, 200)), list(range(200, 1200))],
    #               pos=lgc.get_pos_for_locally_bipartite_graph(100, 1000))
    # plt.show()

    # Get the parameter map for the MS algorithm on all of the test graphs.
    # time_limits = [1]
    # sizes = [10]
    # for i, size in enumerate(sizes):
    #     k = 7
    #     d = 5
    #     G = load_sbm_graph(size=size, k=k, d=d)
    #     n = k * size * 100
    #     target_L = list(range(size * 100))
    #     target_R = list(range(size * 100, 2 * size * 100)) + list(range(n - (size * 100), n))
    #     test_lp_parameters(G, 1, 1, 8, -9, -1, 150, target_L, target_R, max_time=time_limits[i],
    #                        filename=f"results/lp_{k}_cycle_{size}00_degree_{d}.csv")
    #     test_ms_parameters(G, 1, -6, -3, -3, -0, 20, 50, target_L, target_R, max_time=time_limits[i],
    #                        filename=f"results/ms_{k}_cycle_{size}00_degree_{d}.csv")

    # Visualise the graphs
    # size = 2
    # G = load_sbm_graph(small=False, cycle=True, size=size)
    # L, R, bipart = lgc.find_bipartite_clusters.ms_almost_bipartite(G, 1, 0.2, 1e-5)
    # L, R, bipart = lgc.find_bipartite_clusters.lp_almost_bipartite(G, 1, 6, 1e-7)
    # print(sorted(L))
    # print(sorted(R))
    # print(bipart)
    # G.draw(pos=lgc.get_pos_for_sbm(100, 8))
    # not_S = set(range(size * 100 * 8)) - set(L) - set(R)
    # G.draw_groups([L, R, list(not_S)], pos=lgc.get_pos_for_sbm(size * 100, 8))
    # plt.show()

    # Compare MS algorithm with same parameters on graphs of different sizes
    # alpha = 3e-1
    # epsilon = 6e-6
    # G = load_sbm_graph(small=False, cycle=True, size=1)
    # print("Size 100")
    # report_MS_performance(G, 1, alpha, epsilon)

    # alpha = 1e-2
    # epsilon = 6e-6
    # G = load_sbm_graph(small=False, cycle=True, size=32)
    # print("Size 3200")
    # report_MS_performance(G, 1, alpha, epsilon)


def slashdot_experiment():
    """
    Run experiments on the slashdot graph
    :return:
    """
    G = load_slashdot_graph(negative=True)
    compare_bipartieness_algs(G, 24968, show_clusters=True, skip_cheeger=False)


def directed_experiment():
    """
    Run experiments for the directed case
    :return:
    """
    G = load_slashdot_graph(negative=False, semi_double_cover=True, largest_component=False)
    L, R = lgc.find_bipartite_clusters.ms_evo_cut_directed(G, 1, 0.5, T=2, debug=False)
    print(L)
    print(R)


def reddit_experiment():
    """
    Run experiments on the reddit graph
    :return:
    """
    G, vertex_subreddit_dict, subreddit_vertex_dict = load_reddit_graph()

    #########################################
    # Run the algorithm on every subreddit
    #########################################
    bipartitenesses = {}
    volumes = {}
    for v in range(G.adjacency_matrix.shape[0]):
        if v % 100 == 0:
            print(f"Vertex {v}")
        if len(G.neighbors(v)) > 0:
            L, R, bipart = lgc.find_bipartite_clusters.ms_almost_bipartite(G, v, alpha=0.8, epsilon=1e-5)
            bipartitenesses[v] = bipart
            volumes[v] = G.volume(L + R)

    sorted_subreddits = sorted(bipartitenesses, key=bipartitenesses.get)
    for sub in sorted_subreddits:
        if volumes[sub] > 100:
            print(f"{sub:<10}{vertex_subreddit_dict[sub]:<30}{bipartitenesses[sub]}\t{volumes[sub]}")

    ##########################################
    # Run the algorithm on a certain subreddit
    ##########################################
    # compare_bipartieness_algs(G, subreddit_vertex_dict["the_donald"])
    L, R, bipart = lgc.find_bipartite_clusters.ms_almost_bipartite(G, subreddit_vertex_dict["wikileaks"], alpha=0.8, epsilon=1e-5)
    print(f"Bipartiteness: {bipart}")
    for i in range(max(len(L), len(R))):
        left_subreddit = ""
        right_subreddit = ""
        if i < len(L):
            left_subreddit = vertex_subreddit_dict[L[i]]
        if i < len(R):
            right_subreddit = vertex_subreddit_dict[R[i]]
        print(f"{left_subreddit:<30}{right_subreddit}")


def print_country_groups(L, R, vertex_country_dict):
    """Given the L and R sets for the MID dataset, print the countries for Mathematica"""
    left_countries = sorted([vertex_country_dict[x] for x in L])
    right_countries = sorted([vertex_country_dict[x] for x in R])
    print("group1={"
          f""" {", ".join(['"' + x + '"' for x in left_countries])} """
          "};")
    print("group2={"
          f""" {", ".join(['"' + x + '"' for x in right_countries])} """
          "};")


def mid_experiment():
    """Run experiments with the military dataset"""
    # Load the three graphs of interest
    G, vertex_country_dict, country_vertex_dict = load_mid_graph(1950, 1990)
    graphs = [G]
    # for start_year in range(1800, 2000, 200):
    #     end_year = start_year + 100
    #     graph, vertex_country_dict, country_vertex_dict = load_mid_graph(start_year, end_year)
    #     graphs.append(graph)

    starting_country = "United States of America"

    # Compute the parameters for the algorithms. Based on the result of the cheeger cut algorithm
    # MS_parameters = [(0.001, 1e-8), (0.5, 1e-6), (0.5, 1e-6)]
    # LP_parameters = [(10, 1e-7), (4, 1e-5), (4, 1e-5)]
    MS_parameters = []
    LP_parameters = []
    for i, graph in enumerate(graphs):
        L, R, target_bipartiteness = lgc.find_bipartite_clusters.bipart_cheeger_cut(graph)
        target_volume = graph.volume(L + R)
        print(f"Graph {i + 1}, volume = {target_volume}, bipartiteness = {target_bipartiteness}")

        alpha = 20 * target_bipartiteness
        epsilon = 1 / (20 * target_volume)
        MS_parameters.append((min(alpha, 0.5), epsilon))
        print(f"MS Parameters: {min(alpha, 0.5):.3f}, {epsilon:.3e}")

        eps = 0.1
        T = (eps * math.log(1600 * target_volume)) / (6 * target_bipartiteness)
        xi = 1 / ((target_volume ** (1 + eps)) * 800 * T)
        T = int(round(T))
        LP_parameters.append((max(T, 2), xi))
        print(f"LP Parameters: {max(T, 2)}, {xi:.3e}")

    for i, G in enumerate(graphs):
        print(f"Graph {i + 1}")

        # Run the MS algorithm
        start_time = time.clock()
        ms_L, ms_R, ms_bipart = lgc.find_bipartite_clusters.ms_almost_bipartite(
            G, country_vertex_dict[starting_country], alpha=MS_parameters[i][0], epsilon=MS_parameters[i][1]
        )
        ms_time = time.clock() - start_time
        print("MS Algorithm")
        print(f"\ttime:\t\t\t{ms_time:.3f}\n\tbipartiteness:\t{ms_bipart:.3f}\n\tvolume:\t\t\t{int(G.volume(ms_L + ms_R))}")
        print_country_groups(ms_L, ms_R, vertex_country_dict)
        print()

        # Run the LP algorithm
        start_time = time.clock()
        lp_L, lp_R, lp_bipart = lgc.find_bipartite_clusters.lp_almost_bipartite(
            G, country_vertex_dict[starting_country], T=LP_parameters[i][0], xi_0=LP_parameters[i][1]
        )
        lp_time = time.clock() - start_time
        print("LP Algorithm")
        print(f"\ttime:\t\t\t{lp_time:.3f}\n\tbipartiteness:\t{lp_bipart:.3f}\n\tvolume:\t\t\t{int(G.volume(lp_L + lp_R))}")
        print_country_groups(ms_L, ms_R, vertex_country_dict)
        print()


def migration_experiment():
    """
    Experiment on the migration dataset with the directed flow clustering algorithm.
    :return:
    """
    ################################################
    # Run a single trial
    ################################################
    # migration_semi_DC = migrationdata.load_migration_semi_dc()
    # starting_vertices = migrationdata.SAN_FRAN_INDEX_SET[8] + migration_visualisation.MIG_N
    # print(f"Starting vertex: {starting_vertices}")
    # L, R, CI = lgc.find_bipartite_clusters.ms_evo_cut_directed(migration_semi_DC, starting_vertices,
    #                                                            0.1, T=2, debug=False)
    # print(L)
    # print(R)
    # print(CI)
    #
    # migration_visualisation.highlight_two_sets(L, R)
    # plt.show()

    ##########################################
    # Run a bunch of experiments
    #########################################
    migration_semi_DC = migrationdata.load_migration_semi_dc()
    all_indices = migrationdata.NEW_YORK_INDEX_SET + migrationdata.GEORGIA_INDEX_SET + migrationdata.SAN_FRAN_INDEX_SET + migrationdata.OHIO_INDEX_SET + migrationdata.WEST_INDEX_SET + migrationdata.MIDWEST_INDEX_SET + migrationdata.SOUTH_INDEX_SET + migrationdata.FLORIDA_INDEX_SET
    # all_indices = migrationdata.NEW_YORK_INDEX_SET
    starting_vertices_list = [[v + migration_visualisation.MIG_N] for v in all_indices]
    starting_vertices_list = starting_vertices_list + [[v] for v in all_indices]
    print(f"All starting vertices: {starting_vertices_list}")
    for vertex_idx, starting_vertices in enumerate(starting_vertices_list):
        print(f"Starting vertices: {starting_vertices}")
        CIs = []
        volumes = []
        best_CI = 0
        best_L = []
        best_R = []
        num_trials = 20
        for i in range(num_trials):
            print(f"Vertex {vertex_idx + 1}/{len(starting_vertices_list)}; Run {i + 1}/{num_trials}")
            L, R, CI = lgc.find_bipartite_clusters.ms_evo_cut_directed(migration_semi_DC, starting_vertices,
                                                                       0.1, T=2, debug=False)
            volume = migration_semi_DC.volume(L + R)
            print(f"CI: {CI:.4f}")
            print(f"Volume: {volume}")
            print(L)
            print(R)
            CIs.append(CI)
            volumes.append(volume)

            # If the best CI is at least 0.25, then save the image and clusters
            if 0.5 > CI > 0.25:
                print("!!! Saving output")
                plt.clf()
                migration_visualisation.highlight_two_sets(L, R)
                fig = plt.gcf()
                fig.savefig(f"results/migration/normalised/start_{starting_vertices[0]}_ci_{CI:.4f}.png")
                with open(f"results/migration/normalised/start_{starting_vertices[0]}_ci_{CI:.4f}.txt",
                          'w') as fout:
                    fout.write(f"{L}\n{R}\n")
            print()


            if best_CI < CI < 0.5:
                best_CI = CI
                best_L = L
                best_R = R

        print(f"BEST CI: {best_CI}")
        print(f"AVERAGE CI: {np.mean(CIs)}")
        print(f"AVERAGE VOL: {np.mean(volumes)}")

        # If the best CI is at least 0.25, then save the image and clusters
        # if best_CI > 0.25:
        #     print("!!! Saving output")
        #     plt.clf()
        #     migration_visualisation.highlight_two_sets(best_L, best_R)
        #     fig = plt.gcf()
        #     fig.savefig(f"results/migration/start_{starting_vertices[0]}_ci_{best_CI:.4f}_unnormalised.png")
        #     with open(f"results/migration/start_{starting_vertices[0]}_ci_{best_CI:.4f}_unnormalised.txt", 'w') as fout:
        #         fout.write(f"{best_L}\n{best_R}\n")

        # Print a newline before the next iteration
        print()

    #############################
    # Visualise set of vertices
    #############################
    # migration_visualisation.highlight_migration_set(migrationdata.WEST_INDEX_SET)
    # start_vertex = 3466
    # L = [1408, 2, 3, 2443, 1936, 401, 402, 2710, 408, 409, 1954, 291, 421, 296, 428, 49, 309, 442, 454, 456, 328, 464, 2773, 473, 480, 481, 357, 358, 501, 380]
    # R = [512, 391, 472, 474, 115, 1015]
    # migration_visualisation.highlight_two_sets(L, R)
    # plt.show()

    #####################################
    # Find starting indices
    #####################################
    # lat_min = 36
    # lat_max = 39
    # long_min = -130
    # long_max = -121
    # california = migrationdata.find_indices_geo(lat_min, lat_max, long_min, long_max)
    # print(california)
    # migration_visualisation.highlight_migration_set(california)
    # plt.show()

