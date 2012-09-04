"""
Non-specific functions for RSAR
"""
# Author: Samuel Vaiter <samuel.vaiter@ceremade.dauphine.fr>

from __future__ import division
import numpy as np
import scipy
from scipy import linalg

#def null(A, eps=1e-15):
#    """
#    Computes the null space of the real matrix A
#    """
#    n, m = scipy.shape(A)
#    if n > m :
#        return scipy.transpose(null(scipy.transpose(A), eps))
#        return null(scipy.transpose(A), eps)
#    u, s, vh = scipy.linalg.svd(A)
#    s=scipy.append(s,scipy.zeros(m))[0:m]
#    null_mask = (s <= eps)
#    null_space = scipy.compress(null_mask, vh, axis=0)
#    return scipy.transpose(null_space)

def null(A, tol=1e-10, row_wise_storage=True):
    """
    Return the null space of a matrix A.
    If row_wise_storage is True, a two-dimensional array where the
    vectors that span the null space are stored as rows, otherwise
    they are stored as columns.

    Code by Bastian Weber based on code by Robert Kern and Ryan Krauss.
    """
    n, m = A.shape
    if n > m :
        return np.transpose(null(np.transpose(A), tol))

    u, s, vh = linalg.svd(A)
    s = np.append(s, np.zeros(m))[0:m]
    null_mask = (s <= tol)
    null_space = np.compress(null_mask, vh, axis=0)
    null_space = np.conjugate(null_space)  # in case of complex values
    if row_wise_storage:
        return null_space
    else:
        return np.transpose(null_space)

# Adrien Gaidon - INRIA - 2011
def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex

    Solves the optimisation problem (using the algorithm from [1]):

        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0

    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project

    s: int, optional, default: 1,
       radius of the simplex

    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex

    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.

    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
        # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w

# Adrien Gaidon - INRIA - 2011
def l1ball_projection(v, s=1):
    """ Compute the Euclidean projection on a L1-ball

    Solves the optimisation problem (using the algorithm from [1]):

        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s

    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project

    s: int, optional, default: 1,
       radius of the L1-ball

    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s

    Notes
    -----
    Solves the problem by a reduction to the positive simplex case

    See also
    --------
    euclidean_proj_simplex
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    u = np.abs(v)
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        return v
        # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = euclidean_proj_simplex(u, s=s)
    # compute the solution to the original problem on v
    w *= np.sign(v)
    return w