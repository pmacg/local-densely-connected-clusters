"""
This file gives several methods for processing the results from the experiments.
"""
import pandas as pd
import sys
import numpy as np
import localgraphclustering as lgc
from matplotlib import pyplot as plt
from . import bipartiteness_experiments


def load_csv_with_timeouts(filename):
    """Load a dataframe from the csv output by the experiments. Fix the missing time data."""
    df = pd.read_csv(filename, skipinitialspace=True)
    df = df.replace(to_replace="TIMED_OUT", value=sys.float_info.max)
    correct_types = {"time": float}
    return df.astype(correct_types)


def get_time_vs_objective(df, objective='ari', minimise=False, lp=False, max_time=None):
    """Given a dataframe create a time vs objective curve, and report the corresponding parameters.

    The LP parameter is used to determine whether we are analysing the MS algorithm or the LP algorithm. If analysing
    the MS algorithm, return the alpha, and epsilon parameters.

    If analysing the LP algorithm, return the T and xi_0 parameters."""
    min_time = df.loc[:, 'time'].min()
    max_time_data = df.query("time <= 31540000").loc[:, 'time'].max()

    if max_time is None:
        max_time = max_time_data
    times = np.linspace(min_time, min(max_time, max_time_data), 100)
    return_times = []
    objectives = []
    param_1 = []
    param_2 = []
    last_objective = None
    for t in times:
        # For each time, find the largest ari found in at most that time
        filtered_df = df.query(f"time <= {t}")

        if minimise:
            best_idx = filtered_df.loc[:, objective].idxmin()
        else:
            best_idx = filtered_df.loc[:, objective].idxmax()

        # Decide whether to report this one
        this_objective = filtered_df.loc[best_idx, objective]
        if last_objective is None or (minimise and this_objective < last_objective) or (not minimise and this_objective > last_objective):
            last_objective = this_objective
            objectives.append(filtered_df.loc[best_idx, objective])
            return_times.append(t)

            if lp:
                param_1.append(filtered_df.loc[best_idx, 'T'])
                param_2.append(filtered_df.loc[best_idx, 'xi_0'])
            else:
                param_1.append(filtered_df.loc[best_idx, 'alpha'])
                param_2.append(filtered_df.loc[best_idx, 'epsilon'])

    return return_times, objectives, param_1, param_2


def generate_picture_sequence(size=1, lp=False, k=8, d=30, objective='ari', minimise=False):
    """Create a sequence of pictures of the clustering from a time/bipartiteness or time/ari curve. Ths parameter lp
    determines whether we are analysing the ms or the lp algorithm."""
    # Load the graph
    G = bipartiteness_experiments.load_sbm_graph(size=size, k=k, d=d)
    positions = lgc.get_pos_for_locally_bipartite_graph(size * 100, size * 1000)

    if lp:
        df = load_csv_with_timeouts(f"results/lp_local_{size}00_degree_{d}.csv")
    else:
        df = load_csv_with_timeouts(f"results/ms_local_{size}00_degree_{d}.csv")

    times, metric, param_1, param_2 = get_time_vs_objective(df, objective=objective, lp=lp)

    # Get the best objective value
    best_objective = min(metric) if minimise else max(metric)

    num_images = len(times)
    for i, time in enumerate(times):
        # For each time, run the algorithm again
        if lp:
            print(f"Iteration {i}/{num_images}, time={time:.3f}, {objective}={metric[i]:.2f}, T={param_1[i]}, xi_0={param_2[i]}")
            L, R, bipart = lgc.find_bipartite_clusters.lp_almost_bipartite(G, 1, T=param_1[i], xi_0=param_2[i])
        else:
            print(f"Iteration {i}/{num_images}, time={time:.3f}, {objective}={metric[i]:.2f}, alpha={param_1[i]}, epsilon={param_2[i]}")
            L, R, bipart = lgc.find_bipartite_clusters.ms_almost_bipartite(G, 1, alpha=param_1[i], epsilon=param_2[i])

        # Get the volume of the result
        volume = G.volume(L + R)

        # Generate the image
        not_S = list(set(range(200 * size + 1000 * size)) - set(L) - set(R))
        G.draw_groups([L, R, not_S], pos=positions)

        # Add various text.
        if lp:
            plt.text(-1.5, 1.8, f"LP Algorithm")
            plt.text(-1.5, 1.65, f"T = {param_1[i]}")
            plt.text(-1.5, 1.5, f"xi_0 = {param_2[i]:.2e}")
        else:
            plt.text(-1.5, 1.8, f"MS Algorithm")
            plt.text(-1.5, 1.65, f"alpha = {param_1[i]:.2e}")
            plt.text(-1.5, 1.5, f"eps = {param_2[i]:.2e}")
        plt.text(-1.5, -1.2, f"time: {time:.3f}")
        plt.text(-1.5, -1.35, f"bipartiteness: {bipart:.3f}")
        plt.text(-1.5, -1.5, f"volume: {volume}")
        if objective == 'ari':
            plt.text(-1.5, -1.65, f"ari: {metric[i]}")
        plt.text(1.1, 1.8, f"size = {size}")
        plt.text(1.1, 1.65, f"d = {d}")

        # print(L)
        # print(R)
        # plt.show()

        fig = plt.gcf()
        if not lp:
            fig.savefig(f"results/pictures/ms_local_{size}_{d}_{i}.png")
        else:
            fig.savefig(f"results/pictures/lp_local_{size}_{d}_{i}.png")

        # Clear the current figure for the next round
        plt.clf()

        # If we've reached the optimal objective, end here
        if metric[i] == best_objective:
            print()
            print(f"Best objective reached at iteration {i}")
            print()
            break


def get_optimal_parameters(size=1, degree=20, objective='ari', minimise=False, plot=True, final=False, vertex=1, prop=None, max_time=None, filenames=None):
    """
    Compare the runtimes of the MS algorithm and the LP algorithm.
    :return:
    """
    if filenames is not None:
        ms_df = load_csv_with_timeouts(filenames[0])
        lp_df = load_csv_with_timeouts(filenames[1])
    else:
        if not final:
            ms_df = load_csv_with_timeouts(f"results/ms_local_{size}00_degree_{degree}.csv")
            lp_df = load_csv_with_timeouts(f"results/lp_local_{size}00_degree_{degree}.csv")
        else:
            if prop is not None:
                ms_df = load_csv_with_timeouts(f"results/ms_{size}00_vertex_{vertex}_prop_{prop}.csv")
                lp_df = load_csv_with_timeouts(f"results/lp_{size}00_vertex_{vertex}_prop_{prop}.csv")
            else:
                ms_df = load_csv_with_timeouts(f"results/ms_{size}00_vertex_{vertex}.csv")
                lp_df = load_csv_with_timeouts(f"results/lp_{size}00_vertex_{vertex}.csv")

    ms_times, ms_obj, alphas, epsilons = get_time_vs_objective(ms_df, objective=objective, minimise=minimise, lp=False, max_time=max_time)
    lp_times, lp_obj, Ts, xi_0s = get_time_vs_objective(lp_df, objective=objective, minimise=minimise, lp=True, max_time=max_time)

    if plot and False:
        print("MS Algorithm")
        for j, t in enumerate(ms_times):
            print(f"time: {t:.4f}, obj: {ms_obj[j]:.4f}, alpha: {alphas[j]:.4e}, eps: {epsilons[j]:.4e}")
        print("\nLP Algorithm")
        for j, t in enumerate(lp_times):
            print(f"time: {t:.4f}, obj: {lp_obj[j]:.4f}, T: {Ts[j]}, xi_0: {xi_0s[j]:.4e}")

    if plot:
        # Add a data point to the plot with the smaller maximum time
        if ms_times[-1] < lp_times[-1]:
            ms_times.append(lp_times[-1])
            ms_obj.append(ms_obj[-1])
        elif lp_times[-1] < ms_times[-1]:
            lp_times.append(ms_times[-1])
            lp_obj.append(lp_obj[-1])

        plt.plot(ms_times, ms_obj, marker='D', mfc='blue')
        plt.plot(lp_times, lp_obj, marker='D', mfc='orange')
        plt.show()

    return alphas, epsilons, Ts, xi_0s, ms_times, ms_obj, lp_times, lp_obj


def get_single_graph_avg(times, objectives, target_time):
    """
    Helper function for get_avg_curve
    """
    # Find the average time, and add to the curve
    prev_time = None
    prev_obj = None
    next_time = None
    next_obj = None
    for i, time in enumerate(times):
        if time < target_time and (prev_time is None or time > prev_time):
            prev_time = time
            prev_obj = objectives[i]
        if time >= target_time and (next_time is None or time < next_time):
            next_time = time
            next_obj = objectives[i]
    if next_time is None:
        return prev_obj
    if prev_time is None:
        return next_obj
    else:
        return prev_obj + ((target_time - prev_time) / (next_time - prev_time)) * (next_obj - prev_obj)


def get_avg_curve(size, degree, objective, minimise, vertices, prop=None, plot=True, max_time=None, filenames=None):
    """Get the average time/objective curves for the given size and degree"""
    ms_curves = []
    lp_curves = []
    graph_times = None
    for vertex in vertices:
        if filenames is not None:
            these_filenames = filenames[vertex]
        else:
            these_filenames = None
        _, _, _, _, ms_times, ms_obj, lp_times, lp_obj = get_optimal_parameters(
            size=size, degree=degree, objective=objective, minimise=minimise, plot=False, final=True, vertex=vertex, prop=prop, max_time=max_time, filenames=these_filenames)
        if graph_times is None:
            graph_times = np.linspace(max(min(ms_times), min(lp_times)) + 0.001, max(max(ms_times), max(lp_times)) - 0.001, 100)

        new_curve = []
        for time in graph_times:
            # Find the average time, and add to the curve
            avg_obj = get_single_graph_avg(ms_times, ms_obj, time)
            new_curve.append(avg_obj)
        ms_curves.append(new_curve)

        new_curve = []
        for time in graph_times:
            # Find the average time, and add to the curve
            avg_obj = get_single_graph_avg(lp_times, lp_obj, time)
            new_curve.append(avg_obj)
        lp_curves.append(new_curve)

    avg_ms_curve = []
    for i in range(len(ms_curves[0])):
        avg_ms_curve.append(np.mean([curve[i] for curve in ms_curves]))
    avg_lp_curve = []
    for i in range(len(lp_curves[0])):
        avg_lp_curve.append(np.mean([curve[i] for curve in lp_curves]))

    if plot:
        last_t = 0
        for i, t in enumerate(graph_times):
            if i != 0:
                print(f"{t:.3f}, {avg_ms_curve[i]:.3f}, {avg_lp_curve[i]:.3f}, {(avg_ms_curve[i] - avg_ms_curve[i - 1]) / (t - last_t):.3f}, {(avg_lp_curve[i] - avg_lp_curve[i - 1]) / (t - last_t):.3f}")
            last_t = t
        plt.plot(graph_times, avg_ms_curve)
        plt.plot(graph_times, avg_lp_curve)
        plt.show()

    return graph_times, avg_ms_curve, avg_lp_curve


def get_key_stats(ms_objectives, ms_times, lp_objectives, lp_times, objective_str, minimise, target_objective):
    """Helper function for full analysis."""
    # For each algorithm, find the time that it takes to reach the target ari value
    ms_time = None
    for i, time in enumerate(ms_times):
        if not minimise and ms_objectives[i] > target_objective and (ms_time is None or time < ms_time):
            ms_time = time
        if minimise and ms_objectives[i] < target_objective and (ms_time is None or time < ms_time):
            ms_time = time
    lp_time = None
    for i, time in enumerate(lp_times):
        if not minimise and lp_objectives[i] > target_objective and (lp_time is None or time < lp_time):
            lp_time = time
        if minimise and lp_objectives[i] < target_objective and (lp_time is None or time < lp_time):
            lp_time = time

    if lp_time is None:
        print(f"Target {objective_str}: {target_objective:.4f}, MS time: {ms_time:.4f}, LP time: {lp_time}")
    else:
        print(f"Target {objective_str}: {target_objective:.4f}, MS time: {ms_time:.4f}, LP time: {lp_time:.4f}")

    return ms_time, lp_time


def get_target_objective(size, v_starts, objective, minimise):
    """Given an objective function name, figure out the target for a particular size of graph"""
    # The target objective value is set at 95% of the average maximum objective from either algorithm
    opt_objectives = []
    for v_start in v_starts:
        _, _, _, _, _, ms_objectives, _, lp_objectives = get_optimal_parameters(size=size,
                                                                                objective=objective,
                                                                                minimise=minimise,
                                                                                plot=False,
                                                                                final=True,
                                                                                vertex=v_start)
        if minimise:
            opt_objectives.append(min(lp_objectives + ms_objectives))
        else:
            opt_objectives.append(max(lp_objectives + ms_objectives))

    # Now, calculate 95% of the average optimal objective
    if minimise:
        return 1 - 0.9 * (1 - np.mean(opt_objectives))
    else:
        return 0.9 * np.mean(opt_objectives)


def run_analysis():
    """
    Analyse the experiments!
    :return:
    """
    sizes = [1]

    for size in sizes:
        v_starts = list(range(1, 6)) + list(range(size * 100 + 1, size * 100 + 6))
        ms_ari_times = []
        lp_ari_times = []
        ms_bipart_times = []
        lp_bipart_times = []
        ms_symdiff_times = []
        lp_symdiff_times = []

        # Determine the target objective values for this size of graph
        target_ari = get_target_objective(size, v_starts, 'ari', False)
        target_bipart = get_target_objective(size, v_starts, 'bipartiteness', True)
        target_symdiff = get_target_objective(size, v_starts, 'symdiff', True)

        for v_start in v_starts:
            print(f"Size: {size}, v_start: {v_start}")
            alphas, epsilons, Ts, xis, ms_times, ms_aris, lp_times, lp_aris = get_optimal_parameters(
                size=size, objective='ari', minimise=False, plot=False, final=True, vertex=v_start)
            ms_time, lp_time = get_key_stats(ms_aris, ms_times, lp_aris, lp_times, 'ari', False, target_ari)
            ms_ari_times.append(ms_time)
            lp_ari_times.append(lp_time)

            alphas, epsilons, Ts, xis, ms_times, ms_biparts, lp_times, lp_biparts = get_optimal_parameters(
                size=size, objective='bipartiteness', minimise=True, plot=False, final=True, vertex=v_start)
            ms_time, lp_time = get_key_stats(ms_biparts, ms_times, lp_biparts, lp_times, 'bipartiteness', True, target_bipart)
            ms_bipart_times.append(ms_time)
            lp_bipart_times.append(lp_time)

            alphas, epsilons, Ts, xis, ms_times, ms_symdiffs, lp_times, lp_symdiffs = get_optimal_parameters(
                size=size, objective='symdiff', minimise=True, plot=False, final=True, vertex=v_start)
            ms_time, lp_time = get_key_stats(ms_symdiffs, ms_times, lp_symdiffs, lp_times, 'symdiff', True, target_symdiff)
            ms_symdiff_times.append(ms_time)
            lp_symdiff_times.append(lp_time)

        print()
        print(f"Mean time to target ARI. MS: {np.mean(ms_ari_times):.4f}, LP: {np.mean(lp_ari_times):.4f} ({np.mean(lp_ari_times)/np.mean(ms_ari_times):.4f} X longer)")
        print(f"Mean time to target bipartiteness. MS: {np.mean(ms_bipart_times):.4f}, LP: {np.mean(lp_bipart_times):.4f} ({np.mean(lp_bipart_times)/np.mean(ms_bipart_times):.4f} X longer)")
        print(f"Mean time to target symdiff. MS: {np.mean(ms_symdiff_times):.4f}, LP: {np.mean(lp_symdiff_times):.4f} ({np.mean(lp_symdiff_times)/np.mean(ms_symdiff_times):.4f} X longer)")
        print()
