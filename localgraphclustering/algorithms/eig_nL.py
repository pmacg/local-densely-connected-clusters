import numpy as np
import scipy as sp
import scipy.sparse.linalg as splinalg


def eig_nL(g, tol_eigs=1.0e-6, normalize: bool = True, dim: int = 1, find_top_eigs=False):
    """
       DESCRIPTION
       -----------

       Computes eigenvectors of the normalized Laplacian matrix.

       PARAMETERS (mandatory)
       ----------------------

       g:         graph object

       PARAMETERS (optional)
       ---------------------

       dim: positive, int
            default == 1
            The number of eigenvectors or dimensions to compute.

       find_top_eigs: bool
                      default == False
                      Whether to find the top eigenvalues instead of the bottom.

       tol_eigs: positive float, double
                 default == 1.0e-6
                 Tolerance for computation of the eigenvector that corresponds to
                 the second smallest eigenvalue of the normalized Laplacian matrix.

       normalize: bool,
                  default == True
                  True if we should return the eigenvectors of the generalized
                  eigenvalue problem associated with the normalized Laplacian.
                  This should be on unless you know what you are doing.

       RETURNS
       ------

       p:    Eigenvector or Eigenvector matrix that
             corresponds to the second smallest eigenvalue of the
             normalized Laplacian matrix and larger eigenvectors if dim >= 0.
       v:    Vector of eigenvalues corresponding to the eigenvectors in p.
    """
    n = g.adjacency_matrix.shape[0]

    D_sqrt_neg = sp.sparse.spdiags(g.dn_sqrt.transpose(), 0, n, n)

    L = sp.sparse.identity(n) - D_sqrt_neg.dot((g.adjacency_matrix.dot(D_sqrt_neg)))

    if find_top_eigs:
        emb_eig_val, p = splinalg.eigsh(L, which='LM', k=dim, tol=tol_eigs)
        F = np.real(p)
    else:
        emb_eig_val, p = splinalg.eigsh(L, which='SM', k=1+dim, tol=tol_eigs)
        F = np.real(p[:, 1:])

    if normalize:
        F *= g.dn_sqrt[:, np.newaxis]
    return F, emb_eig_val


"""
Random walks and local cuts in graphs, Chung, LAA 2007
We just form the sub-matrix of the Laplacian and use the eigenvector there.
"""
def eig2nL_subgraph(g, ref_nodes, tol_eigs = 1.0e-6, normalize: bool = True):
    A_sub = g.adjacency_matrix.tocsr()[ref_nodes, :].tocsc()[:, ref_nodes]
    nref = len(ref_nodes)
    D_sqrt_neg = sp.sparse.spdiags(g.dn_sqrt[ref_nodes].transpose(), 0, nref, nref)
    L_sub = sp.sparse.identity(nref) - D_sqrt_neg.dot((A_sub.dot(D_sqrt_neg)))
    emb_eig_val, emb_eig = splinalg.eigsh(L_sub, which='SM', k=1, tol=tol_eigs)
    emb_eig *= -1 if max(emb_eig) < 0 else 1
    f = emb_eig[:,0]
    if normalize:
        f *= g.dn_sqrt[ref_nodes]
    return ((ref_nodes,f), emb_eig_val)
