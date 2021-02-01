"""A python wrapper for the weighted dcpagerank cpp function. For reference, the key parameters are:
n - number of vertices
ai,aj,a - graph in CSR
alpha - value of alpha
eps - value of epsilon
seedids - the set of indices for seeds
maxsteps - the max number of steps
xids_1, xids_2, actual_length - the solution vector
xlength - the maximum allowed length of the output vector
values_1, values_2 - a pair of vectors representing the pagerank value vector on the double cover
                     for xids_1 and xids_2
"""
import ctypes
import warnings

import numpy as np
from numpy.ctypeslib import ndpointer

from . import _graphlib
from .utility import determine_types, standard_types


def _setup_dc_pagerank_weighted_args(vtypestr, itypestr, fun):
    """
    Configures the types to be used for the arguments to the c++ function based on the types of the CSR matrix.
    :param vtypestr: the type of the index vector from the CSR matrix
    :param itypestr: the type of the indptr vector from the CSR matrix.
    :param fun: the c++ function we are going to call
    :return: the function fun with the argument types set correctly
    """
    float_type, vtype, itype, ctypes_vtype, ctypes_itype, bool_type = standard_types(
        vtypestr, itypestr)

    fun.restype = ctypes_vtype
    fun.argtypes = [ctypes_vtype,                                       # n - number of vertices
                    ndpointer(ctypes_itype, flags="C_CONTIGUOUS"),      # ai - indptr vector
                    ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),      # aj - index vector
                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),   # a - data vector
                    ctypes_vtype,                                       # offset - 0 or 1
                    ctypes.c_double,                                    # alpha
                    ctypes.c_double,                                    # epsilon
                    ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),      # seedids vector
                    ctypes_vtype,                                       # nseedids
                    ctypes_vtype,                                       # maxsteps
                    ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),      # xids_1 - output vector
                    ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),      # xids_2 - output vector
                    ctypes_vtype,                                       # xlength - length of output vector
                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),   # values_1 - first output value vector
                    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),   # values_2 - second output value vector
                    bool_type]                                          # simplify - whether to simplify

    return fun


# Define the python functions wrapping the c++ dcpagerank functions with different argument types
_graphlib_funs_dcpagerank_weighted64 = _setup_dc_pagerank_weighted_args(
    'int64', 'int64', _graphlib.dcpagerank_weighted64)
_graphlib_funs_dcpagerank_weighted32 = _setup_dc_pagerank_weighted_args(
    'uint32', 'uint32', _graphlib.dcpagerank_weighted32)
_graphlib_funs_dcpagerank_weighted32_64 = _setup_dc_pagerank_weighted_args(
    'uint32', 'int64', _graphlib.dcpagerank_weighted32_64)


def _get_dcpagerank_weighted_cpp_types_fun(ai, aj):
    """
    Given the numpy vectors specifying the sparse adjacency matrix, return a tuple containing:
      - c float type name
      - python aj type name
      - python ai type name
      - c aj type name
      - c ai type name
      - the corresponding python function which wraps this c++ function call
    :param ai: the indptr vector from the CSR
    :param aj: the index vector from the CSR
    :return: tuple of types, and function to use
    """
    float_type, vtype, itype, ctypes_vtype, ctypes_itype = determine_types(ai, aj)
    if (vtype, itype) == (np.int64, np.int64):
        fun = _graphlib_funs_dcpagerank_weighted64
    elif (vtype, itype) == (np.uint32, np.int64):
        fun = _graphlib_funs_dcpagerank_weighted32_64
    else:
        fun = _graphlib_funs_dcpagerank_weighted32
    return float_type, vtype, itype, ctypes_vtype, ctypes_itype, fun


def dcpagerank_weighted_cpp(n, ai, aj, a, alpha, eps, seedids, maxsteps, simplify=True, xlength=10**7):
    """
    Python function wrapping c++ dcpagerank function. Computes the pagerank on the double cover of a graph.
    :param n: the number of vertices in the graph
    :param ai: the indptr vector of the CSR adjacency matrix
    :param aj: the index vector of the CSR adjacency matrix
    :param a: the data vector of the CSR adjacency matrix
    :param alpha: the parameter alpha for computing the pagerank
    :param eps: the parameter epsilon for computing the pagerank
    :param seedids: a list of seed node ids
    :param maxsteps: the maximum number of steps to take in the pagerank calculation
    :param simplify: whether to simplify the pagerank vector before returning it
    :param xlength: the 'guesses' length of the support of the pagerank vector. If actual pagerank vector support is
                    larger than this, then the algorithm will be run twice.
    :return: a tuple:
        actual_xids_1 - the vertex ids of the support of the pagerank vector
        actual_xids_2 - the vertex ids of the support of the pagerank vector
        actual_values_1 - the pagerank values of the vertices in xids_1
        actual_values_2 - the pagerank values of the vertices in xids_2
    """
    # Find the appropriate types and the function to call
    float_type, vtype, itype, ctypes_vtype, ctypes_itype, fun = _get_dcpagerank_weighted_cpp_types_fun(ai, aj)

    # Set up the parameters for the function call, including making sure their types are correct.
    nseedids = len(seedids)
    seedids = np.array(seedids, dtype=vtype)
    xids_1 = np.zeros(xlength, dtype=vtype)
    xids_2 = np.zeros(xlength, dtype=vtype)
    values_1 = np.zeros(xlength, dtype=float_type)
    values_2 = np.zeros(xlength, dtype=float_type)

    # Set the array offset. In python, this is 0.
    offset = 0

    # Call the c++ function to compute the double cover pagerank.
    actual_length = fun(n, ai, aj, a, offset, alpha, eps, seedids, nseedids, maxsteps, xids_1, xids_2, xlength,
                        values_1, values_2, simplify)

    # If the actual output is longer than we expected, we will need to run the algorithm again to ensure we get the
    # correct output.
    if actual_length > xlength:
        warnings.warn("Running pagerank for a second time. The xlength parameter was not long enough.")

        # Re-initialise the output vectors
        xlength = actual_length
        xids_1 = np.zeros(xlength, dtype=vtype)
        xids_2 = np.zeros(xlength, dtype=vtype)
        values_1 = np.zeros(xlength, dtype=float_type)
        values_2 = np.zeros(xlength, dtype=float_type)

        # Call the pagerank method again with more memory allocated.
        actual_length = fun(n, ai, aj, a, offset, alpha, eps,
                            seedids, nseedids, maxsteps, xids_1, xids_2, xlength, values_1, values_2, simplify)

    actual_values_1 = values_1[0:actual_length]
    actual_values_2 = values_2[0:actual_length]
    actual_xids_1 = xids_1[0:actual_length]
    actual_xids_2 = xids_2[0:actual_length]

    # Since actual_length is the length of the longest of xids_1 and xids_2, check whether the actual value is 0, and
    # if so, ignore it.
    num_zeros_1 = 0
    len_1 = len(actual_values_1)
    len_2 = len(actual_values_2)
    while len_1 > 0 and actual_values_1[-1] == 0:
        num_zeros_1 += 1
        actual_values_1 = actual_values_1[:-1]
        len_1 -= 1
    num_zeros_2 = 0
    while len_2 > 0 and actual_values_2[-1] == 0:
        num_zeros_2 += 1
        actual_values_2 = actual_values_2[:-1]
        len_2 -= 1
    if num_zeros_1 > 0:
        actual_xids_1 = actual_xids_1[:-num_zeros_1]
    if num_zeros_2 > 0:
        actual_xids_2 = actual_xids_2[:-num_zeros_2]

    return actual_xids_1, actual_xids_2, actual_values_1, actual_values_2
