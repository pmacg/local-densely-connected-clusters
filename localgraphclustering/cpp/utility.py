import ctypes

import numpy as np


def determine_types(ai, aj):
    """
    Given the numpy vectors given by ai and aj, determine which types to use for the cpp+ functions.
    :param ai: the indptr vector from the CSR adjacency matrix
    :param aj: the index vector from the CSR adjacency matrix
    :return: a tuple of types to use for the c++ function - see standard_types documentation
    """
    float_type = ctypes.c_double
    dt = np.dtype(ai[0])
    (itype, ctypes_itype) = (np.int64, ctypes.c_int64) if dt.name == 'int64' else (
        np.uint32, ctypes.c_uint32)
    dt = np.dtype(aj[0])
    (vtype, ctypes_vtype) = (np.int64, ctypes.c_int64) if dt.name == 'int64' else (
        np.uint32, ctypes.c_uint32)

    return float_type, vtype, itype, ctypes_vtype, ctypes_itype


def standard_types(vtypestr, itypestr):
    """
    Given the types of the data in the CSR matrix, determine which python and c types to use accross the python/c
    interface. Returns a tuple with:
      - c float type
      - python vtype
      - python itype
      - c vtype
      - c itype
      - c bool type
    :param vtypestr: a string describing the type used for the CSR matrix indptr vector
    :param itypestr: a string describing the type used for the CSR matrix index vector
    :return:
    """
    float_type = ctypes.c_double
    (vtype, ctypes_vtype) = (np.int64, ctypes.c_int64) if vtypestr == 'int64' else (
        np.uint32, ctypes.c_uint32)
    (itype, ctypes_itype) = (np.int64, ctypes.c_int64) if itypestr == 'int64' else (
        np.uint32, ctypes.c_uint32)
    bool_type = ctypes.c_bool
    return float_type, vtype, itype, ctypes_vtype, ctypes_itype, bool_type
