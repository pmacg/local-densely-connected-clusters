"""
Provides a main method for running code from this package.
You should implement anything you'd like to run as a method in the experiments folder, and call a single function from
this main function.
"""
from stochastic_block_model import create_block_path_graph, create_block_cycle_graph, create_locally_bipartite_graph, pure_locally_bipartite_graph, create_local_flow_graph, create_triangle_flow_graph
import experiments
import os.path
from matplotlib import pyplot as plt


def main():
    """Run a single experiment."""
    # create_block_path_graph("/home/peter/wc/LocalGraphClustering/localgraphclustering/"
    #                         "experiments/datasets/sbm/block_path_graph.edgelist",
    #                         1000, 200, 0.01, 0.09)
    # create_block_path_graph("/home/peter/wc/LocalGraphClustering/localgraphclustering/"
    #                         "experiments/datasets/sbm/small_block_path_graph.edgelist",
    #                         100, 200, 0.1, 0.9)
    # create_block_cycle_graph("/home/peter/wc/LocalGraphClustering/localgraphclustering/"
    #                         "experiments/datasets/sbm/block_cycle_graph.edgelist",
    #                         1000, 50, 0.01, 0.09)
    # create_block_cycle_graph("/home/peter/wc/LocalGraphClustering/localgraphclustering/"
    #                         "experiments/datasets/sbm/block_cycle_graph_test.edgelist",
    #                         50, 5, 0.9, 0.1)
    # for size in [10]:
    #     d = 5
    #     n = size * 100
    #     p = (d / n) * 0.1
    #     q = (d / n) * (0.9 / 2)
    #     k = 7
    #     create_block_cycle_graph(f"/home/peter/wc/LocalGraphClustering/localgraphclustering/"
    #                             f"experiments/datasets/sbm/{k}_cluster_size_{n}_degree_{d}.edgelist",
    #                             n, k, p, q)

    # d = 30
    # prop = 0.99
    # with open("/home/peter/wc/LocalGraphClustering/localgraphclustering/"
    #           "experiments/results/fixed_param_all_eps.csv", 'a') as f_out:
    #     f_out.write("size, d, prop, target_volume, lp_eps, ms_time, ms_bipart, ms_vol, ms_ari, ms_symdiff, lp_time, lp_bipart, lp_vol, lp_ari, lp_symdiff\n")
        # max_time = 60
        # eps_too_long_strike_1 = [False]
        # eps_too_long_strike_2 = [False]
        # for size in list(range(80, 110, 10)):
        #     n1 = size * 100
        #     n2 = size * 1000
        #
            # Create the SBM graph if it does not exist already
            # sbm_filename = f"/home/peter/wc/LocalGraphClustering/localgraphclustering/experiments/datasets/sbm/local_cluster_size_{n1}_{n2}_degree_{d}_prop_{prop}.edgelist"
            # if not os.path.isfile(sbm_filename):
            #     create_locally_bipartite_graph(sbm_filename, n1, n2, d, prop, 0.5 * (1 - prop))
            #
            # starting_vertices = list(range(1, 6)) + [n1 + i for i in range(1, 6)]
            # for eps_idx, lp_eps in enumerate([0.01]):
            #     if not eps_too_long_strike_2[eps_idx]:
            #         results = experiments.fixed_param_experiment(size, d, prop, starting_vertices, lp_eps, show_output=True)
            #
                # Check whether the epsilon curve is too long
                # if results[5] > max_time:
                #     if eps_too_long_strike_1[eps_idx]:
                #         eps_too_long_strike_2[eps_idx] = True
                #     eps_too_long_strike_1[eps_idx] = True

                # f_out.write(f"{size}, {d}, {prop}, {2 * 100 * size * d}, {lp_eps}, " + ", ".join([str(r) for r in results]) + "\n")
                # f_out.flush()


    # experiments.bipartiteness_experiment()
    # experiments.get_optimal_parameters(size=100, degree=20, objective='ari', minimise=False, plot=True, final=True, vertex=5)

    ##############################################
    # Draw averaged curves
    ##############################################
    # size = 100
    # d = 20
    # prop = None
    # max_time = None
    # starting_vertices = list(range(1, 6)) + [(size * 100) + n for n in list(range(1, 6))]
    # starting_vertices = [1, 2, 3]
    # times, ms_obj, lp_obj = experiments.get_avg_curve(size, d, 'ari', False, starting_vertices, prop=prop, max_time=max_time)
    # times, ms_obj, lp_obj = experiments.get_avg_curve(size, d, 'bipartiteness', True, starting_vertices, prop=prop, max_time=max_time)
    # times, ms_obj, lp_obj = experiments.get_avg_curve(size, d, 'symdiff', True, starting_vertices, prop=prop, max_time=max_time)
    # print(list(times))
    # print(ms_obj)
    # print(lp_obj)

    ############################################
    # Analyse ratio experiment
    ############################################
    # size = 10
    # n1 = size * 100
    # n2 = size * 1000
    # p1 = 0.001
    # p2 = 0.002
    # q2 = 0.0001
    # multiples = list(range(1, 11)) #+ [16, 32]
    # starting_vertices = list(range(1, 6)) + [(size * 100) + n for n in list(range(1, 6))]
    # best_ms_objs = []
    # best_lp_objs = []
    # for mult in multiples:
    #     filenames = {}
    #     for v_start in starting_vertices:
    #         filenames[v_start] = (f"results/ms_{size}00_vertex_{v_start}_mult_{mult}.csv", f"results/lp_{size}00_vertex_{v_start}_mult_{mult}.csv")
    #     times, ms_obj, lp_obj = experiments.get_avg_curve(
    #         size, 20, 'symdiff', True, starting_vertices, plot=False, filenames=filenames)
    #
        # Find the best objective in less than some maximum time
        # best_ms_obj = None
        # best_lp_obj = None
        # max_time = 0.25
        # for i, time in enumerate(times):
        #     if time <= max_time:
        #         if best_ms_obj is None or ms_obj[i] < best_ms_obj:
        #             best_ms_obj = ms_obj[i]
        #         if best_lp_obj is None or lp_obj[i] < best_lp_obj:
        #             best_lp_obj = lp_obj[i]
        # print(f"Prop: {mult}\nBest MS obj: {best_ms_obj:.4f}\nBest LP obj: {best_lp_obj}\n")
        # best_lp_objs.append(best_lp_obj)
        # best_ms_objs.append(best_ms_obj)
    # plt.plot(multiples, best_ms_objs)
    # plt.plot(multiples, best_lp_objs)
    # plt.show()

    #########################################################
    # Create graphs for ratio experiment
    #########################################################
    # size = 10
    # n1 = size * 100
    # n2 = size * 1000
    # p1 = 0.001
    # p2 = 0.002
    # q2 = 0.0001
    # multiples = list(range(1, 11))
    # for mult in multiples:
    #     q1 = mult * p1
    #     sbm_filename = f"/home/peter/wc/LocalGraphClustering/localgraphclustering/experiments/datasets/sbm/local_cluster_size_{n1}_{n2}_{p1}_{q1}_{p2}_{q2}.edgelist"
    #     if not os.path.isfile(sbm_filename):
    #         pure_locally_bipartite_graph(sbm_filename, n1, n2, p1, q1, p2, q2)

    ########################################################
    # Create graphs from the directed stochastic block model
    ########################################################
    # ns = [100, 1000]
    # qs = [3, 10]  # to be divided by n
    # ps = [0, 0.5, 1]  # to be multiplied by q
    # etas = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # for n in ns:
    #     for q_unnormalised in qs:
    #         q = q_unnormalised / n
    #         for p_unnormalised in ps:
    #             p = p_unnormalised * q
    #             for eta in etas:
    #                 sbm_filename = f"/home/peter/wc/dcpagerank/localgraphclustering/experiments/datasets/dsbm/csdbm_{n}_{p}_{q}_{eta}.edgelist"
    #                 if not os.path.isfile(sbm_filename):
    #                     create_triangle_flow_graph(sbm_filename, n, p, q, eta)

    ##################################################
    # Test the CLSZ algorithm
    ##################################################
    # filename = "/home/peter/wc/dcpagerank/localgraphclustering/experiments/datasets/dsbm/triangle_flow_100_0.5_0.5_1.edgelist"
    # create_triangle_flow_graph(filename, 100, 0.5, 0.5, 1)

    # experiments.run_full_experiment()
    # experiments.slashdot_experiment()
    # experiments.directed_experiment()
    # experiments.run_ratio_experiment()
    # experiments.reddit_experiment()
    # experiments.mid_experiment()
    # experiments.migration_experiment()
    experiments.run_directed_experiment()


if __name__ == "__main__":
    main()
