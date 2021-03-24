"""
This file provides methods for visualising the result of various clustering techniques, on the migration dataset.
"""
from scipy.io import loadmat
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from . import migrationdata
# from uszipcode import SearchEngine


# The number of vertices in the migration dataset.
MIG_N = 3075

# Some useful colormaps
CM_RAINBOW = plt.get_cmap('rainbow')


def get_migration_zipcodes():
    m = loadmat('datasets/migration/ALL_CENSUS_DATA_FEB_2015.mat')
    # This is the array of latitude and longitude values for each vertex in the graph
    lat_long = migrationdata.transpose_lists(m['A'])
    long = lat_long[0]
    lat = lat_long[1]

    # search = SearchEngine()

    zip_codes = []
    # last_zipcode = 0
    # for v in range(len(long)):
    #     result = search.by_coordinates(lat[v], long[v], returns=1)
    #     if len(result) > 0:
    #         zip = result[0].to_dict()["zipcode"]
    #     else:
    #         zip = last_zipcode
    #     last_zipcode = zip
    #     zip_codes.append(zip)

    print('{"' + '", "'.join(zip_codes) + '"}')



def plot_migration_values(v: List[float], cmap=CM_RAINBOW, norm_min=None, norm_max=None, colors=None, ignore_zeros=False):
    """
    Given a vector of values to apply to each vertex in the migration graph, plot a heatmap visualisation, with each
    vertex in place geographically.
    :param v: the vector of values to apply
    :param colors: The color specifications to use for each point
    :param ignore_zeros: Whether to ignore vertices whose value is 0. Valid only with colors
    """
    m = loadmat('datasets/migration/ALL_CENSUS_DATA_FEB_2015.mat')

    # This is the array of latitude and longitude values for each vertex in the graph
    lat_long = migrationdata.transpose_lists(m['A'])

    if norm_min is not None and norm_max is not None:
        normalize = matplotlib.colors.Normalize(vmin=norm_min, vmax=norm_max)
    else:
        normalize = None

    # Show the USA outline
    usa_image = np.array(Image.open('datasets/migration/usa-map-grey-thin.png'), dtype=np.uint8)
    plt.imshow(usa_image, extent=(-126, -65.8, 23.6, 50.3))
    # usa_image = np.array(Image.open('datasets/migration/usa-map.pdf'), dtype=np.uint8)
    # plt.imshow(usa_image, extent=(-126, -65.8, 23.6, 50.3))

    if colors is not None:
        x_vals = []
        y_vals = []
        real_colors = []
        if ignore_zeros:
            for i, value in enumerate(v):
                if value != 0:
                    # print(f"Vertex: {i}, lat, long: {lat_long[0][i]:.4f}, {lat_long[1][i]:.4f}")
                    x_vals.append(lat_long[0][i])
                    y_vals.append(lat_long[1][i])
                    real_colors.append(colors[i])
        else:
            x_vals = lat_long[0]
            y_vals = lat_long[1]
            real_colors = colors
        plt.scatter(x_vals, y_vals, c=real_colors, marker='.', s=100)
        plt.axis('off')
    else:
        plt.scatter(lat_long[0], lat_long[1], c=v, norm=normalize, cmap=cmap, marker='o', s=20)
        plt.axis('off')


def highlight_migration_set(S: List[int]):
    """
    Given a set of vertices S, plot the map with these vertices highlighted.
    :param S: the set of vertices to highlight.
    """
    v = [0] * MIG_N
    colors = ['lightgrey'] * MIG_N
    for i in S:
        if i > MIG_N:
            j = i - MIG_N
        else:
            j = i
        v[j] = 1
        colors[j] = 'tab:red'

    plot_migration_values(v, colors=colors)


def highlight_two_sets(L, R, show_others=False):
    """
    Given two vertex sets, highlight them in different colors.
    :param L:
    :param R:
    :param show_others: whether to show the graph vertices which are not in the key sets
    """
    v = [0] * MIG_N
    colors = ['lightgrey'] * MIG_N
    for i in L:
        if i > MIG_N:
            j = i - MIG_N
        else:
            j = i
        v[j] = 1
        # colors[j] = 'tab:green'
        colors[j] = '#ffb500'
    for i in R:
        if i > MIG_N:
            j = i - MIG_N
        else:
            j = i
        v[j] = 2
        colors[j] = '#3b5998'

    plot_migration_values(v, colors=colors, ignore_zeros=(not show_others))


def visualize_zip_codes():
    """
    A fun function for visualising how the zip code values change accross the country.
    """
    m = loadmat('data/ALL_CENSUS_DATA_FEB_2015.mat')
    zip_codes = migrationdata.transpose_lists(m['CODES'])
    plot_migration_values(zip_codes[0])


def visualize_indices():
    """
    Visualize how the indices of the data are distributed geographically
    :return:
    """
    plot_migration_values(list(range(MIG_N)))
