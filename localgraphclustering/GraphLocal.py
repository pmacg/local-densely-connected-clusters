"""
Provides the GraphLocal class for representing a graph for use by local algorithms.
"""
import collections as cole
import multiprocessing as mp
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from scipy import sparse as sp
from scipy.sparse import csgraph

from .GraphDrawing import GraphDrawing
from .algorithms import eig_nL
from .cpp import *


def _load_from_shared(sabuf, dtype, shape):
    return np.frombuffer(sabuf, dtype=dtype).reshape(shape)


def _copy_to_shared(a):
    """ Create shared memory that can be passed to a child process,
    wrapped in a numpy array."""
    # determine the numpy type of a.
    dtype = a.dtype
    shape = a.shape
    sabuf = mp.RawArray(ctypes.c_uint8, a.nbytes)
    sa = _load_from_shared(sabuf, dtype, shape)
    np.copyto(sa, a)  # make a copy
    return sa, (sabuf, dtype, shape)


class GraphLocal:
    """
    This class implements graph loading from an edgelist, gml or graphml and provides methods that operate on the graph.

    Attributes
    ----------
    adjacency_matrix : scipy csr matrix

    ai : numpy vector
        CSC format index pointer array, its data type is determined by "itype" during initialization

    aj : numpy vector
        CSC format index array, its data type is determined by "vtype" during initialization

    _num_vertices : int
        Number of vertices

    _num_edges : int
        Number of edges

    weighted : boolean
        Declares if it is a weighted graph or not

    d : float64 numpy vector
        Degrees vector

    dn : float64 numpy vector
        Component-wise reciprocal of degrees vector

    d_sqrt : float64 numpy vector
        Component-wise square root of degrees vector

    dn_sqrt : float64 numpy vector
        Component-wise reciprocal of sqaure root degrees vector

    vol_G : float64 numpy vector
        Volume of graph

    components : list of sets
        Each set contains the indices of a connected component of the graph

    number_of_components : int
        Number of connected components of the graph

    bicomponents : list of sets
        Each set contains the indices of a biconnected component of the graph

    number_of_bicomponents : int
        Number of connected components of the graph

    core_numbers : dictionary
        Core number for each vertex

    Methods
    -------

    read_graph(filename, file_type='edgelist', separator='\t')
        Reads the graph from a file

    compute_statistics()
        Computes statistics for the graph

    connected_components()
        Computes the connected components of the graph

    is_disconnected()
        Checks if graph is connected

    biconnected_components():
        Computes the biconnected components of the graph

    core_number()
        Returns the core number for each vertex

    neighbors(vertex)
        Returns a list with the neighbors of the given vertex

    list_to_gl(source,target)
        Create a GraphLocal object from edge list
    """

    def __init__(self,
                 filename=None,
                 file_type='edgelist',
                 separator='\t',
                 remove_whitespace=False, header=False, headerrow=None,
                 vtype=np.uint32, itype=np.uint32, semi_double_cover=False):
        """
        Initializes the graph from a gml or a edgelist file and initializes the attributes of the class.

        Parameters
        ----------
        See read_graph for a description of the parameters.
        """
        self.rw_laplacian = None

        if filename is not None:
            self.read_graph(filename, file_type=file_type, separator=separator, remove_whitespace=remove_whitespace,
                            header=header, header_row=headerrow, vtype=vtype, itype=itype,
                            semi_double_cover=semi_double_cover)

    def __eq__(self, other):
        if not isinstance(other, GraphLocal):
            return NotImplemented
        return np.array_equal(self.ai, other.ai) and np.array_equal(self.aj, other.aj) and np.array_equal(self.adjacency_matrix.data, other.adjacency_matrix.data)

    def read_graph(self, filename, file_type='edgelist', separator='\t', remove_whitespace=False, header=False,
                   header_row=None, vtype=np.uint32, itype=np.uint32, semi_double_cover=False):
        """
        Reads the graph from an edgelist, gml or graphml file and initializes the class attribute adjacency_matrix.

        Parameters
        ----------
        filename : string
            Name of the file, for example 'JohnsHopkins.edgelist', 'JohnsHopkins.gml', 'JohnsHopkins.graphml'.

        file_type : string
            Type of file. Currently only 'edgelist', 'gml' and 'graphml' are supported.
            Default = 'edgelist'

        separator : string
            used if file_type = 'edgelist'
            Default = '\t'

        remove_whitespace : bool
            set it to be True when there is more than one kinds of separators in the file
            Default = False

        header : bool
            This lets the first line of the file contain a set of header
            information that should be ignore_index
            Default = False

        header_row : int
            Use which row as column names. This argument takes precedence over
            the header=True using headerrow = 0
            Default = None

        vtype
            numpy integer type of CSC format index array
            Default = np.uint32

        itype
            numpy integer type of CSC format index pointer array
            Default = np.uint32

        semi_double_cover
            whether to read a directed graph and construct the semi-double cover. Otherwise, the adjacency matrix will
            just be symettrized in the straightforward way. Works only with edgelist filetypes.
        """
        if semi_double_cover and file_type != 'edgelist':
            raise Exception("Can only read the semi-double cover graph from an edgelist file.")

        if file_type == 'edgelist':
            # The rows of the edgelist file must be one of the following formats, and all rows must have the same format
            #    <int> <int> <float>
            # or
            #    <int> <int>
            # with items separated by the separator given in the parameters.
            # The first two numbers are the vertex ids of the edge. The third number if it exists is the weight of the
            # edge.
            #
            # Lines beginning with '#' are treated as comments.

            # If the header_row is not explicitly specified, use the first row.
            if header and header_row is None:
                header_row = 0

            if remove_whitespace:
                df = pd.read_csv(filename, header=header_row, delim_whitespace=remove_whitespace, comment='#')
            else:
                df = pd.read_csv(
                    filename, sep=separator, header=header_row, delim_whitespace=remove_whitespace, comment='#')
            cols = [0, 1, 2]
            if header is not None:
                cols = list(df.columns)
            source = df[cols[0]].values
            target = df[cols[1]].values
            if df.shape[1] == 2:
                weights = np.ones(source.shape[0])
            elif df.shape[1] == 3:
                weights = df[cols[2]].values
            else:
                raise Exception(
                    'GraphLocal.read_graph: df.shape[1] not in (2, 3)')
            self._num_vertices = max(source.max() + 1, target.max() + 1)

            if semi_double_cover:
                self._num_vertices = 2 * self._num_vertices

            if not semi_double_cover:
                self.adjacency_matrix = sp.csr_matrix((weights.astype(
                    np.float64), (source, target)), shape=(self._num_vertices, self._num_vertices))
            else:
                self.adjacency_matrix = sp.csr_matrix(
                    (weights.astype(np.float64), (source, [t + (self._num_vertices / 2) for t in target])),
                    shape=(self._num_vertices, self._num_vertices))

        elif file_type == 'gml':
            warnings.warn(
                "Loading a gml is not efficient, we suggest using an edgelist format for this API.")
            G = nx.read_gml(filename).to_undirected()
            self.adjacency_matrix = nx.adjacency_matrix(G).astype(np.float64)
            self._num_vertices = nx.number_of_nodes(G)

        elif file_type == 'graphml':
            warnings.warn(
                "Loading a graphml is not efficient, we suggest using an edgelist format for this API.")
            G = nx.read_graphml(filename).to_undirected()
            self.adjacency_matrix = nx.adjacency_matrix(G).astype(np.float64)
            self._num_vertices = nx.number_of_nodes(G)

        else:
            print('This file type is not supported')
            return

        self.weighted = False
        for i in self.adjacency_matrix.data:
            if i != 1:
                self.weighted = True
                break
        is_symmetric = (self.adjacency_matrix !=
                        self.adjacency_matrix.T).sum() == 0
        if not is_symmetric:
            # Symmetrize matrix, choosing larger weight
            sel = self.adjacency_matrix.T > self.adjacency_matrix
            self.adjacency_matrix = self.adjacency_matrix - self.adjacency_matrix.multiply(sel) + \
                self.adjacency_matrix.T.multiply(sel)
            assert (self.adjacency_matrix != self.adjacency_matrix.T).nnz == 0

        self._num_edges = self.adjacency_matrix.nnz
        self.compute_statistics()
        self.ai = itype(self.adjacency_matrix.indptr)
        self.aj = vtype(self.adjacency_matrix.indices)
        self.initialise_dc()
        self.initialise_rw_laplacian()

    def initialise_rw_laplacian(self):
        """
        Return the random walk laplacian matrix, constructing it if it has not been constructed before.
        :return:
        """
        D_inv = sp.spdiags(self.dn.transpose(), 0, self._num_vertices, self._num_vertices)
        self.rw_laplacian = sp.identity(self._num_vertices) - self.adjacency_matrix.dot(D_inv)

    @classmethod
    def from_networkx(cls, G):
        """
        Create a GraphLocal object from a networkx graph.

        Paramters
        ---------
        G
            The networkx graph.
        """
        if G.is_directed() == True:
            raise Exception(
                "from_networkx requires an undirected graph, use G.to_undirected()")
        rval = cls()
        rval.adjacency_matrix = nx.adjacency_matrix(G).astype(np.float64)
        rval._num_vertices = nx.number_of_nodes(G)

        # TODO, use this in the read_graph
        rval.weighted = False
        for i in rval.adjacency_matrix.data:
            if i != 1:
                rval.weighted = True
                break

        # automatically determine sizes
        if G.number_of_nodes() < 4294967295:
            vtype = np.uint32
        else:
            vtype = np.int64
        if 2*G.number_of_edges() < 4294967295:
            itype = np.uint32
        else:
            itype = np.int64

        rval._num_edges = rval.adjacency_matrix.nnz
        rval.compute_statistics()
        rval.ai = itype(rval.adjacency_matrix.indptr)
        rval.aj = vtype(rval.adjacency_matrix.indices)
        rval.initialise_dc()
        rval.initialise_rw_laplacian()
        return rval

    @classmethod
    def from_sparse_adjacency(cls, A):
        """
        Create a GraphLocal object from a sparse adjacency matrix.

        Paramters
        ---------
        A
            Adjacency matrix.
        """
        self = cls()
        self.adjacency_matrix = A.copy()
        self._num_vertices = A.shape[0]
        self._num_edges = A.nnz

        # TODO, use this in the read_graph
        self.weighted = False
        for i in self.adjacency_matrix.data:
            if i != 1:
                self.weighted = True
                break

        # automatically determine sizes
        if self._num_vertices < 4294967295:
            vtype = np.uint32
        else:
            vtype = np.int64
        if 2*self._num_edges < 4294967295:
            itype = np.uint32
        else:
            itype = np.int64

        self.compute_statistics()
        self.ai = itype(self.adjacency_matrix.indptr)
        self.aj = vtype(self.adjacency_matrix.indices)
        self.initialise_dc()
        self.initialise_rw_laplacian()
        return self

    def renew_data(self, A):
        """
        Update data because the adjacency matrix changed

        Paramters
        ---------
        A
            Adjacency matrix.
        """
        self._num_edges = A.nnz

        # TODO, use this in the read_graph
        self.weighted = False
        for i in self.adjacency_matrix.data:
            if i != 1:
                self.weighted = True
                break

        # automatically determine sizes
        if self._num_vertices < 4294967295:
            vtype = np.uint32
        else:
            vtype = np.int64
        if 2*self._num_edges < 4294967295:
            itype = np.uint32
        else:
            itype = np.int64

        self.compute_statistics()
        self.ai = itype(self.adjacency_matrix.indptr)
        self.aj = vtype(self.adjacency_matrix.indices)
        self.initialise_dc()
        self.initialise_rw_laplacian()

    def list_to_gl(self, source, target, weights, vtype=np.uint32, itype=np.uint32):
        """
        Create a GraphLocal object from edge list.

        Parameters
        ----------
        source
            A numpy array of sources for the edges

        target
            A numpy array of targets for the edges

        weights
            A numpy array of weights for the edges

        vtype
            numpy integer type of CSC format index array
            Default = np.uint32

        itype
            numpy integer type of CSC format index pointer array
            Default = np.uint32
        """

        # TODO, fix this up to avoid duplicating code with read...

        source = np.array(source, dtype=vtype)
        target = np.array(target, dtype=vtype)
        weights = np.array(weights, dtype=np.double)

        self._num_edges = len(source)
        self._num_vertices = max(source.max(initial=0) + 1, target.max(initial=0) + 1)
        self.adjacency_matrix = sp.csr_matrix((weights.astype(
            np.float64), (source, target)), shape=(self._num_vertices, self._num_vertices))
        self.weighted = False
        for i in self.adjacency_matrix.data:
            if i != 1:
                self.weighted = True
                break
        is_symmetric = (self.adjacency_matrix !=
                        self.adjacency_matrix.T).sum() == 0
        if not is_symmetric:
            # Symmetrize matrix, choosing larger weight
            sel = self.adjacency_matrix.T > self.adjacency_matrix
            self.adjacency_matrix = self.adjacency_matrix - \
                self.adjacency_matrix.multiply(
                    sel) + self.adjacency_matrix.T.multiply(sel)
            assert (self.adjacency_matrix != self.adjacency_matrix.T).nnz == 0

        self._num_edges = self.adjacency_matrix.nnz
        self.compute_statistics()
        self.ai = itype(self.adjacency_matrix.indptr)
        self.aj = vtype(self.adjacency_matrix.indices)
        self.initialise_dc()
        self.initialise_rw_laplacian()

    def discard_weights(self):
        """ Discard any weights that were loaded from the data file.
        This sets all the weights associated with each edge to 1.0,
        which is our "no weight" case."""
        self.adjacency_matrix.data.fill(1.0)
        self.weighted = False
        self.compute_statistics()

    def compute_statistics(self):
        """
        Computes statistics for the graph. It updates the class attributes.
        The user needs to read the graph first before calling
        this method by calling the read_graph method from this class.
        """
        self.d = np.ravel(self.adjacency_matrix.sum(axis=1))
        self.dn = np.zeros(self._num_vertices)
        self.dn[self.d != 0] = 1.0 / self.d[self.d != 0]
        self.d_sqrt = np.sqrt(self.d)
        self.dn_sqrt = np.sqrt(self.dn)
        self.vol_G = np.sum(self.d)

    def initialise_dc(self):
        """
        Computes the sparse representation of the double cover of the graph.
        """
        # Determine which types to use for the double cover graph
        # Note that 2147483647 = 4294967295 / 2
        #       and 1073741823 = 4294967295 / 4
        if self._num_vertices < 2147483647:
            vtype = np.uint32
        else:
            vtype = np.int64
        if self._num_edges < 1073741823:
            itype = np.uint32
        else:
            itype = np.int64

        # The double cover adjacency matrix is a block matrix from the original adjacency matrix
        # Note that this assumes there are no self loops in the original graph
        self.adjacency_matrix_dc = sp.bmat([[None, self.adjacency_matrix], [self.adjacency_matrix, None]], format='csr')
        self.ai_dc = vtype(self.adjacency_matrix_dc.indptr)
        self.aj_dc = itype(self.adjacency_matrix_dc.indices)

    def to_shared(self):
        """ Re-create the graph data with multiprocessing compatible
        shared-memory arrays that can be passed to child-processes.

        This returns a dictionary that allows the graph to be
        re-created in a child-process from that variable and
        the method "from_shared"

        At this moment, this doesn't send any data from components,
        core_numbers, or biconnected_components
        """
        sgraphvars = {}
        self.ai, sgraphvars["ai"] = _copy_to_shared(self.ai)
        self.aj, sgraphvars["aj"] = _copy_to_shared(self.aj)
        self.d, sgraphvars["d"] = _copy_to_shared(self.d)
        self.dn, sgraphvars["dn"] = _copy_to_shared(self.dn)
        self.d_sqrt, sgraphvars["d_sqrt"] = _copy_to_shared(self.d_sqrt)
        self.dn_sqrt, sgraphvars["dn_sqrt"] = _copy_to_shared(self.dn_sqrt)
        self.adjacency_matrix.data, sgraphvars["a"] = _copy_to_shared(
            self.adjacency_matrix.data)

        # this will rebuild without copying
        # so that copies should all be accessing exactly the same
        # arrays for caching
        self.adjacency_matrix = sp.csr_matrix(
            (self.adjacency_matrix.data, self.aj, self.ai),
            shape=(self._num_vertices, self._num_vertices))

        # scalars
        sgraphvars["n"] = self._num_vertices
        sgraphvars["m"] = self._num_edges
        sgraphvars["vol"] = self.vol_G
        sgraphvars["weighted"] = self.weighted

        return sgraphvars

    @classmethod
    def from_shared(cls, sgraphvars):
        """ Return a graph object from the output of "to_shared". """
        g = cls()
        g._num_vertices = sgraphvars["n"]
        g._num_edges = sgraphvars["m"]
        g.weighted = sgraphvars["weighted"]
        g.vol_G = sgraphvars["vol"]
        g.ai = _load_from_shared(*sgraphvars["ai"])
        g.aj = _load_from_shared(*sgraphvars["aj"])
        g.adjacency_matrix = sp.csr_matrix(
            (_load_from_shared(*sgraphvars["a"]), g.aj, g.ai),
            shape=(g._num_vertices, g._num_vertices))
        g.d = _load_from_shared(*sgraphvars["d"])
        g.dn = _load_from_shared(*sgraphvars["dn"])
        g.d_sqrt = _load_from_shared(*sgraphvars["d_sqrt"])
        g.dn_sqrt = _load_from_shared(*sgraphvars["dn_sqrt"])
        return g

    def connected_components(self):
        """
        Computes the connected components of the graph. It stores the results in class attributes components
        and number_of_components. The user needs to call read the graph
        first before calling this function by calling the read_graph function from this class.
        """

        output = csgraph.connected_components(
            self.adjacency_matrix, directed=False)

        self.components = output[1]
        self.number_of_components = output[0]

        print('There are ', self.number_of_components,
              ' connected components in the graph')

    def is_disconnected(self):
        """
        The output can be accessed from the graph object that calls this function.

        Checks if the graph is a disconnected graph. It prints the result as a comment and
        returns True if the graph is disconnected, or false otherwise. The user needs to
        call read the graph first before calling this function by calling the read_graph function from this class.
        This function calls Networkx.

        Returns
        -------
        True
             If connected

        False
             If disconnected
        """
        if self.d == []:
            print('The graph has to be read first.')
            return

        self.connected_components()

        if self.number_of_components > 1:
            print('The graph is a disconnected graph.')
            return True
        else:
            print('The graph is not a disconnected graph.')
            return False

    def biconnected_components(self):
        """
        Computes the biconnected components of the graph. It stores the results in class attributes bicomponents
        and number_of_bicomponents. The user needs to call read the graph first before calling this
        function by calling the read_graph function from this class. This function calls Networkx.
        """
        warnings.warn(
            "Warning, biconnected_components is not efficiently implemented.")

        g_nx = nx.from_scipy_sparse_matrix(self.adjacency_matrix)

        self.bicomponents = list(nx.biconnected_components(g_nx))

        self.number_of_bicomponents = len(self.bicomponents)

    def core_number(self):
        """
        Returns the core number for each vertex. A k-core is a maximal
        subgraph that contains nodes of degree k or more. The core number of a node
        is the largest value k of a k-core containing that node. The user needs to
        call read the graph first before calling this function by calling the read_graph
        function from this class. The output can be accessed from the graph object that
        calls this function. It stores the results in class attribute core_numbers.
        """
        warnings.warn("Warning, core_number is not efficiently implemented.")

        g_nx = nx.from_scipy_sparse_matrix(self.adjacency_matrix)

        self.core_numbers = nx.core_number(g_nx)

    def neighbors(self, vertex):
        """
        Returns a list with the neighbors of the given vertex.
        """
        # this will be faster since we store the arrays ourselves.
        return self.aj[self.ai[vertex]:self.ai[vertex+1]].tolist()
        # return self.adjacency_matrix[:,vertex].nonzero()[0].tolist()

    def compute_conductance(self, R, cpp=True):
        """
        Return conductance of a set of vertices.
        """

        records = self.set_scores(R, cpp=cpp)

        return records["cond"]

    def compute_bipartiteness(self, L, R, cpp=True):
        """
        Compute the bipartiteness of a pair of vertex sets
        """
        L_records = self.set_scores(L, cpp=cpp)
        R_records = self.set_scores(R, cpp=cpp)
        S_records = self.set_scores(L + R, cpp=cpp)
        total_volume = S_records["voltrue"]
        cut_L_R = int(0.5 * (L_records["cut"] + R_records["cut"] - S_records["cut"]))
        return 1 - (2 * cut_L_R) / total_volume

    def compute_weight(self, L, R):
        """
        Compute the edge weight between L and R.
        :param L: a set of vertices in the graph
        :param R: another set of vertices
        :return: the edge weight
        """
        A = self.adjacency_matrix.toarray()
        return np.sum(A[np.array(L)[:, None], np.array(R)])

    def volume(self, S, cpp=True):
        """
        Compute the volume of a set of vertices.
        :param S: a list of vertices
        :param cpp: whether to use a c++ implementation
        :return: the volume of the set S
        """
        records = self.set_scores(S, cpp=cpp)
        return records["voltrue"]

    def set_scores(self, R, cpp=True):
        """
        Return various metrics of a set of vertices.
        """
        voltrue, cut = 0, 0
        if cpp:
            voltrue, cut = set_scores_cpp(
                self._num_vertices, self.ai, self.aj, self.adjacency_matrix.data, self.d, R, self.weighted)
        else:
            voltrue = sum(self.d[R])
            v_ones_R = np.zeros(self._num_vertices)
            v_ones_R[R] = 1
            cut = voltrue - \
                np.dot(v_ones_R, self.adjacency_matrix.dot(v_ones_R.T))
        voleff = min(voltrue, self.vol_G - voltrue)

        sizetrue = len(R)
        sizeeff = sizetrue
        if voleff < voltrue:
            sizeeff = self._num_vertices - sizetrue

        # remove the stuff we don't want returned...
        del R
        del self
        if not cpp:
            del v_ones_R
        del cpp

        edgestrue = voltrue - cut
        edgeseff = voleff - cut

        cond = cut / voleff if voleff != 0 else 1
        isop = cut / sizeeff if sizeeff != 0 else 1

        # make a dictionary out of local variables
        return locals()

    def largest_component(self):
        """Return the largest component of the graph as a new GraphLocal object."""
        # Compute the connected components of the graph
        self.connected_components()
        if self.number_of_components == 1:
            return self
        else:
            # find nodes of largest component
            maxccnodes = []
            counter = cole.Counter(self.components)
            what_key = counter.most_common(1)[0][0]
            for i in range(self._num_vertices):
                if what_key == self.components[i]:
                    maxccnodes.append(i)

            warnings.warn("The graph has multiple (%i) components, using the largest with %i / %i nodes" % (
                self.number_of_components, len(maxccnodes), self._num_vertices))

            g_copy = GraphLocal()
            g_copy.adjacency_matrix = self.adjacency_matrix[maxccnodes, :].tocsc()[
                :, maxccnodes].tocsr()
            g_copy._num_vertices = len(maxccnodes)  # AHH!
            g_copy.compute_statistics()
            g_copy.weighted = self.weighted
            dt = np.dtype(self.ai[0])
            itype = np.int64 if dt.name == 'int64' else np.uint32
            dt = np.dtype(self.aj[0])
            vtype = np.int64 if dt.name == 'int64' else np.uint32
            g_copy.ai = itype(g_copy.adjacency_matrix.indptr)
            g_copy.aj = vtype(g_copy.adjacency_matrix.indices)
            g_copy._num_edges = g_copy.adjacency_matrix.nnz
            g_copy.initialise_dc()
            g_copy.initialise_rw_laplacian()
            return g_copy

    def local_extrema(self, vals, strict=False, reverse=False):
        """
        Find extrema in a graph based on a set of values.

        Parameters
        ----------

        vals: Sequence[float]
            a feature value per node used to find the ex against each other, i.e. conductance

        strict: bool
            If True, find a set of vertices where vals(i) < vals(j) for all neighbors N(j)
            i.e. local minima in the space of the graph
            If False, find a set of vertices where vals(i) <= vals(j) for all neighbors N(j)
            i.e. local minima in the space of the graph

        reverse: bool
            if True, then find local maxima, if False then find local minima
            (by default, this is false, so we find local minima)

        Returns
        -------

        minverts: Sequence[int]
            the set of vertices

        minvals: Sequence[float]
            the set of min values
        """
        n = self.adjacency_matrix.shape[0]
        minverts = []
        ai = self.ai
        aj = self.aj
        factor = 1.0
        if reverse:
            factor = -1.0
        for i in range(n):
            vali = factor*vals[i]
            lmin = True
            for nzi in range(ai[i], ai[i+1]):
                v = aj[nzi]
                if v == i:
                    continue  # skip self-loops
                if strict:
                    if vali < factor*vals[v]:
                        continue
                    else:
                        lmin = False
                else:
                    if vali <= factor*vals[v]:
                        continue
                    else:
                        lmin = False

                if lmin == False:
                    break  # break out of the loop

            if lmin:
                minverts.append(i)

        minvals = vals[minverts]

        return minverts, minvals

    @staticmethod
    def _plotting(drawing, edgecolor, edgealpha, linewidth, is_3d, **kwargs):
        """
        private function to do the plotting
        "**kwargs" represents all possible optional parameters of "scatter" function
        in matplotlib.pyplot
        """
        drawing.scatter(**kwargs)
        drawing.plot(color=edgecolor, alpha=edgealpha, linewidths=linewidth)
        axs = drawing.ax
        axs.autoscale()
        if is_3d:
            # Set the initial view
            # TODO: test these angles - are they sensible?
            axs.view_init(30, 30)

    def draw(self, pos=None, position_method='spectral', alpha=1.0, nodesize=5, linewidth=1, nodealpha=1.0,
             edgealpha=1.0, edgecolor='k', nodemarker='o', axs=None, fig=None, values=None, cm=None, valuecenter=None,
             angle=30, figsize=None, nodecolor='r'):
        """
        Standard drawing function when having single cluster

        Optional parameters
        ------------------
        pos: a n-by-2 or n-by-3 array with coordinates for each node of the graph.
            If this is not provided, then the spectral embedding is used.

        position_method: either 'spectral', 'spectral_top' or 'random'. ('spectral' by default)
            Will be used to generate the positions of the vertices if pos is not specified.

        alpha: float (1.0 by default)
            the overall alpha scaling of the plot, [0,1]

        nodealpha: float (1.0 by default)
            the overall node alpha scaling of the plot, [0, 1]

        edgealpha: float (1.0 by default)
            the overall edge alpha scaling of the plot, [0, 1]

        nodecolor: string or RGB ('r' by default)

        edgecolor: string or RGB ('k' by default)

        nodemarker: string ('o' by default)

        nodesize: float (5.0 by default)

        linewidth: float (1.0 by default)

        axs,fig: None,None (default)
            by default it will create a new figure, or this will plot in axs if not None.

        values: Sequence[float] (None by default)
            used to determine node colors in a colormap, should have the same length as coords

        valuecenter: often used with values together to determine vmin and vmax of colormap
            offset = max(abs(values-valuecenter))
            vmax = valuecenter + offset
            vmin = valuecenter - offset

        cm: string or colormap object (None by default)

        figsize: tuple (None by default)

        angle: float (30 by default)
            set initial view angle when drawing 3d

        Returns
        -------

        A GraphDrawing object
        """
        # If the coordinates are not provided by the caller, compute the vertex positions according to the specified
        # method.
        if pos is None:
            if position_method == "spectral":
                # Find the first two non-trivial eigenvectors of the graph laplacian.
                pos, _ = eig_nL(self, dim=2)
            if position_method == "spectral_top":
                # Find the top two eigenvectors of the graph laplacian.
                pos, _ = eig_nL(self, dim=2, find_top_eigs=True)
            if position_method == "random":
                # Generate random positions for each vertex.
                pos = np.random.rand(self._num_vertices, 2)

        drawing = GraphDrawing(self, pos, ax=axs, figsize=figsize)
        if values is not None:
            values = np.asarray(values)
            if values.ndim == 2:
                node_color_list = np.reshape(values, len(pos))
            else:
                node_color_list = values
            vmin = min(node_color_list)
            vmax = max(node_color_list)
            if cm is not None:
                cm = plt.get_cmap(cm)
            else:
                if valuecenter is not None:
                    # when both values and valuecenter are provided, use PuOr colormap to determine colors
                    cm = plt.get_cmap("PuOr")
                    offset = max(abs(node_color_list-valuecenter))
                    vmax = valuecenter + offset
                    vmin = valuecenter - offset
                else:
                    cm = plt.get_cmap("magma")
            self._plotting(drawing, edgecolor, edgealpha, linewidth, len(pos[0]) == 3, c=node_color_list,
                           alpha=alpha*nodealpha, edgecolors='none', s=nodesize, marker=nodemarker, zorder=2, cmap=cm,
                           vmin=vmin, vmax=vmax)
        else:
            self._plotting(drawing, edgecolor, edgealpha, linewidth, len(pos[0]) == 3, c=nodecolor,
                           alpha=alpha*nodealpha, edgecolors='none', s=nodesize, marker=nodemarker, zorder=2)

        return drawing

    def draw_subgraph(self, nodes, pos=None, position_method='spectral', include_neighbourhood=False):
        """
        Given a list of nodes in the graph, draw the subgraph induced by those vertices.

        :param nodes:
            a list of vertices from the graph
        :param pos: (optional)
            a N-by-2 or N-by-3 array with coordinates for each node in the given list where N is len(nodes).
        :param position_method: (optional, default spectral)
            Either 'spectral', 'spectral_top' or 'random'. If pos is not specified, determines how the nodes are
            positioned.
        :param include_neighbourhood: (optional, default False)
            Whether to include the neighborhood of the subgraph.
        :return:
            a GraphDrawing object, similar to the draw() method.
        """
        # Get the number of nodes in the subgraph
        n_subgraph = len(nodes)

        # If we are to include the neighborhood of the subgraph, find the full list of vertices to be plotted
        neighbours = []
        all_nodes = nodes.copy()
        if include_neighbourhood:
            for node in nodes:
                # Find the neighbors of this node, and add them to the list if they are not already there.
                for neighbour in self.neighbors(node):
                    if neighbour not in nodes and neighbour not in neighbours:
                        neighbours.append(neighbour)
                        all_nodes = np.append(all_nodes, neighbour)

        # Construct the induced graph from the list of nodes
        induced_graph_adjacency = self.adjacency_matrix[all_nodes][:, all_nodes]
        induced_graph = GraphLocal.from_sparse_adjacency(induced_graph_adjacency)

        if not include_neighbourhood:
            # If we do not need to color the neighbors differently from the target set, simply return the drawing of
            # the induced graph.
            return induced_graph.draw(pos=pos, position_method=position_method, nodesize=25)
        else:
            # Otherwise, find which vertices in the new graph correspond to the original target set and which correspond
            # to the neighbours of the target set.
            target_set_nodes = []
            neighbour_nodes = []
            sorted_original_nodes = sorted(all_nodes)
            for idx, node in enumerate(sorted_original_nodes):
                if node in nodes:
                    target_set_nodes.append(idx)
                elif node in neighbours:
                    neighbour_nodes.append(idx)

            # Now, return the drawing with the target set and neighbours colored as different clusters.
            return induced_graph.draw_groups([target_set_nodes, neighbour_nodes], pos=pos,
                                             position_method=position_method, edgealpha=1)

    def draw_groups(self, groups, pos=None, position_method='spectral', alpha=1.0, nodesize_list=None,
                    linewidth=1, nodealpha=1.0, edgealpha=0.01, edgecolor='k', nodemarker_list=None,
                    node_color_list=None, nodeorder_list=None, axs=None, fig=None, cm=None, angle=30, figsize=None):
        """
        Standard drawing function when having multiple clusters

        Parameters
        ----------
        groups: list[list] or list, for the first case, each sublist represents a cluster
            for the second case, list must have the same length as the number of nodes and
            nodes with the number are in the same cluster

        Optional parameters
        ------------------
        pos: a n-by-2 or n-by-3 array with coordinates for each node of the graph.
            If this is not provided, then the spectral embedding is used.

        position_method: either 'spectral', 'spectral_top', or 'random'. ('spectral' by default)
            Will be used to generate the positions of the vertices if pos is not specified.

        alpha: float (1.0 by default)
            the overall alpha scaling of the plot, [0,1]

        nodealpha: float (1.0 by default)
            the overall node alpha scaling of the plot, [0, 1]

        edgealpha: float (1.0 by default)
            the overall edge alpha scaling of the plot, [0, 1]

        node_color_list: list of string or RGB ('r' by default)

        edgecolor: string or RGB ('k' by default)

        nodemarker_list: list of strings ('o' by default)

        nodesize_list: list of floats (5.0 by default)

        linewidth: float (1.0 by default)

        axs,fig: None,None (default)
            by default it will create a new figure, or this will plot in axs if not None.

        cm: string or colormap object (None by default)

        figsize: tuple (None by default)

        angle: float (30 by default)
            set initial view angle when drawing 3d

        Returns
        -------

        A GraphDrawing object
        """

        # Initialise default variables
        if nodesize_list is None:
            nodesize_list = []
        if nodemarker_list is None:
            nodemarker_list = []
        if node_color_list is None:
            node_color_list = []
        if nodeorder_list is None:
            nodeorder_list = []

        # If the coordinates are not provided by the caller, compute the vertex positions according to the specified
        # method.
        if pos is None:
            if position_method == "spectral":
                # Find the first two non-trivial eigenvectors of the graph laplacian.
                pos, _ = eig_nL(self, dim=2)
            if position_method == "spectral_top":
                # Find the top two eigenvectors of the graph laplacian.
                pos, _ = eig_nL(self, dim=2, find_top_eigs=True)
            if position_method == "random":
                # Generate random positions for each vertex.
                pos = np.random.rand(self._num_vertices, 2)

        # when values are not provided, use tab20 or gist_ncar colormap to determine colors
        number_of_colors = 1
        l_initial_node_color_list = len(node_color_list)
        l_initial_nodesize_list = len(nodesize_list)
        l_initial_nodemarker_list = len(nodemarker_list)
        l_initial_nodeorder_list = len(nodeorder_list)

        if l_initial_node_color_list == 0:
            node_color_list = np.zeros(self._num_vertices)
        if l_initial_nodesize_list == 0:
            nodesize_list = 25*np.ones(self._num_vertices)
        if l_initial_nodemarker_list == 0:
            nodemarker_list = 'o'
        if l_initial_nodeorder_list == 0:
            nodeorder_list = 2

        groups = np.asarray(groups)

        # If the groups were given as a single list, then convert to the list of lists representation.
        if len(groups) == 1:
            grp_dict = defaultdict(list)
            for idx, key in enumerate(groups):
                grp_dict[key].append(idx)
            groups = np.asarray(list(grp_dict.values()))

        number_of_colors = len(groups)

        # separate the color for different groups as far as we can
        if l_initial_node_color_list == 0:
            for i, g in enumerate(groups):
                node_color_list[g] = i
        if number_of_colors <= 10:
            vmax = 10
            cm = plt.get_cmap("tab10")
        elif number_of_colors <= 20:
            vmax = 20
            cm = plt.get_cmap("tab20")
        else:
            vmax = number_of_colors
            cm = plt.get_cmap("gist_ncar")

        vmin = 0.0
        drawing = GraphDrawing(self, pos, ax=axs, figsize=figsize)

        self._plotting(drawing, edgecolor, edgealpha, linewidth, len(pos[0]) == 3, s=nodesize_list,
                       marker=nodemarker_list, zorder=nodeorder_list, cmap=cm, vmin=vmin, vmax=vmax,
                       alpha=alpha*nodealpha, edgecolors='none', c=node_color_list)

        return drawing
