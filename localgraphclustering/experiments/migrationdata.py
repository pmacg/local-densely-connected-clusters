"""
Contains helper functions for interfacing with the migration dataset.
"""
import math
from scipy.io import loadmat
import localgraphclustering as lgc
import numpy as np


# Some useful indices in the migration dataset
CONNECTICUT_INDEX = 2280  # north east
WISCONSIN_INDEX = 3000  # north
TENNESSEE_INDEX = 500  # mid west
TEXAS_INDEX = 2500  # south
OHIO_INDEX = 2025 # north

# Some useful starting sets
OHIO_INDEX_SET = [2025, 2035, 2050, 2059, 2074, 2084]
FLORIDA_INDEX_SET = [298, 315, 340, 349]
GEORGIA_INDEX_SET = [365, 366, 391, 393, 396, 433, 442, 493, 498, 512, 515]
MIDWEST_INDEX_SET = [86, 89, 125, 1452, 1551, 1553]
SOUTH_INDEX_SET = [1865, 1898, 1899, 1910, 1912, 1915, 1929, 1942, 1954]
WEST_INDEX_SET = [1683, 1709]
NEW_YORK_INDEX_SET = [278, 280, 1740, 1745, 1747, 1748, 1749, 1750, 1751, 1752, 1754, 1756, 1757, 1758, 1795, 1806, 1816, 1822, 1823, 1828, 1832, 1833, 1835, 1836, 1844, 1845, 1848, 1852]
SAN_FRAN_INDEX_SET = [157, 163, 177, 183, 184, 190, 191, 194, 195, 197, 199, 200, 204, 205, 213]


def transpose_lists(l):                                                         
    """Given a list of lists, return the transpose."""
    return list(map(list, zip(*l)))


def load_migration_adjacency():
    """
    Read in the migration dataset.
    :return: return the migration data as a sparse scipy matrix (asymmetric adjacency matrix)
    """
    m = loadmat('data/ALL_CENSUS_DATA_FEB_2015.mat')
    return m['MIG']


def load_migration_lgc():
    """
    Read in the migration dataset as an lgc LocalGraph object.
    :return: a LocalGraph object from the LocalGraphClustering module.
    """
    return lgc.GraphLocal('data/migration_weighted.edgelist', 'edgelist', separator=' ')


def load_migration_semi_dc():
    """
    Read in the semi-double cover of the migration graph as an undirected LocalGraph object.
    :return: a LocalGraph object.
    """
    return lgc.GraphLocal("datasets/migration/migration_data.edgelist",
                          'edgelist', separator=' ', semi_double_cover=True)


def load_migration_a2p_lgc():
    """
    Read in the graph given by (A^2)+ for the migration dataset. This has been precomputed in the data folder.
    :return: a LocalGraph object from the LocalGraphClustering module.
    """
    return lgc.GraphLocal('data/migration_data_a2p.edgelist', 'edgelist', separator=' ')


def load_migration_a2p_normalized_lgc():
    """
    Read in the graph given by the normalized (A^2)+ for the migration dataset.
    :return: a LocalGraph object from the LocalGraphClustering module.
    """
    return lgc.GraphLocal('data/migration_data_a2p_normalized.edgelist', 'edgelist', separator=' ')


def load_migration_hermitian_adjacency():
    """
    Load and compute the hermitian adjacency matrix for the migration dataset.
    :return: a sparse scipy object with the hermitian adjacency
    """
    A = load_migration_adjacency()
    A_norm = np.divide(A, (A + A.T))
    A_norm[np.isnan(A_norm)] = 0
    A_norm[np.isinf(A_norm)] = 0
    return (A_norm * 1.0j) - (np.transpose(A_norm) * 1.0j)


def load_migration_normalized_hermitian_adjacency():
    """
    Load and compute the normalized hermitian adjacency matrix for the migration dataset.
    (The so-called hermitian random walk adjacency matrix).
    :return:
    """
    A = load_migration_hermitian_adjacency()

    # Compute the diagonal degree matrix
    degrees = np.sum(np.absolute(A), axis=1)
    D = np.diagflat(degrees)

    return np.linalg.inv(D) @ A


def compute_migration_a2():
    """
    Compute the A^2 matrix for the migration dataset.
    :return: A^2
    """
    Ah = load_migration_hermitian_adjacency()
    return Ah ** 2


def compute_migration_normalized_a2():
    """
    Compute the normalized A^2 matrix for the migration dataset.
    :return: (D^-1 A)^2
    """
    Ah = load_migration_normalized_hermitian_adjacency()
    return Ah ** 2


def compute_migration_a2p():
    """
    Compute the positive part of the A^2 matrix for the migration dataset.
    :return: (A^2)+
    """
    A2 = compute_migration_a2()
    A2[A2 < 0] = 0
    return A2


def compute_migration_normalized_a2p():
    """
    Compute the positive part of the normalized A^2 matrix for the migration dataset.
    :return: ((D^-1 A)^2)^+
    """
    A2 = compute_migration_normalized_a2()
    A2[A2 < 0] = 0
    return A2


def get_index_of_zipcode(z):
    """
    Given a zip code, find its index in the migration graph.
    If the zip code is not found, returns the index of the nearest zip code (numerically).

    :param z: the zip code to find, as an integer
    :return: the index of the node in the graph, and the zip code it represents
    """
    m = loadmat('data/ALL_CENSUS_DATA_FEB_2015.mat')
    zip_codes = transpose_lists(m['CODES'])
    print(zip_codes)

    best_distance = 10000
    best_index = -1
    best_code = -1
    for i, code in enumerate(zip_codes[0]):
        if math.fabs(code - z) < best_distance:
            best_index = i
            best_distance = math.fabs(code - z)
            best_code = code

    return best_index, best_code


def find_indices_geo(lat_min, lat_max, long_min, long_max):
    """
    Given latitude and longitude constraints, find every index which satisfies them
    :param lat_min:
    :param lat_max:
    :param long_min:
    :param long_max:
    :return:
    """
    m = loadmat('datasets/migration/ALL_CENSUS_DATA_FEB_2015.mat')
    lat_long = transpose_lists(m['A'])
    good_indices = []

    for i in range(len(lat_long[0])):
        if long_min <= lat_long[0][i] <= long_max and lat_min <= lat_long[1][i] <= lat_max:
            good_indices.append(i)

    return good_indices
