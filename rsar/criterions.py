"""
Criterions for analysis regularization: IC, wRC, IC-noker
"""
# Author: Samuel Vaiter <samuel.vaiter@ceremade.dauphine.fr>

from __future__ import division
import numpy as np
import scipy
from scipy.sparse.linalg import cgs, LinearOperator, lsqr

from pyprox.operators import dual_prox
from pyprox.algorithms import douglas_rachford

from rsar.utils import null, l1ball_projection


def _ic_minimization(DJ, Omega_x0, maxiter):
    """
    Returns the solution of the problem
        min_{w \in Ker(D_J)} ||Omega sign(D_I^* x_0) - w||_\infty
    """
    proj = np.dot(np.dot(DJ.T,scipy.linalg.pinv(np.dot(DJ,DJ.T))), DJ)
    prox_indic = lambda w, la: w - np.dot(proj, w)

    proxinf = dual_prox(l1ball_projection)
    prox_obj = lambda x, la: -proxinf(Omega_x0 - x, la) + Omega_x0

    w = douglas_rachford(prox_indic, prox_obj, np.zeros((np.size(DJ,1),1)),
        maxiter=maxiter)
    return np.max(np.abs(Omega_x0 - w))


def criterions_dict(D, Phi, x, maxiter=75):
    # Dimensions of the problem
    N = np.size(D, 0)
    P = np.size(D, 1)
    Q = np.size(Phi, 0)

    # Generate sub-dict of given cosparsity
    I = (np.abs(np.dot(D.T, x)) > 1e-5).flatten()
    J = ~I
    DI = D[:,I]
    DJ = D[:,J]

    # Compute operators involved in criterions
    U = null(DJ.T)
    gram = np.dot(Phi.T, Phi)
    inside = np.dot(np.dot(U.T, gram), U)
    if np.prod(inside.shape) <> 0:
        inside = scipy.linalg.pinv(inside)
    A = np.dot(np.dot(U, inside), U.T)
    Omega = np.dot(scipy.linalg.pinv(DJ), np.dot((np.eye(N) -  np.dot(gram,
        A)), DI))

    # Compute wRC
    wRC = scipy.linalg.norm(Omega, np.inf)

    # D-sign
    ds = np.sign(np.dot(DI.T, x))

    # Compute IC-noker
    ic_noker = lambda s : np.max(np.abs(np.dot(Omega,s)))
    ICnoker = ic_noker(ds)

    # Compute IC
    if np.prod(null(DJ).shape) <> 0:
        ic_ker = lambda s: _ic_minimization(DJ, np.dot(Omega, s), maxiter)
    else:
        ic_ker = ic_noker

    IC = ic_ker(ds)

    res = {
        'wRC' : wRC,
        'IC_noker' : ICnoker,
        'IC' : IC,
        'ic_noker' : ic_noker,
        'ic' : ic_ker,
        'I' : I,
        'J' : J,
        'U' : U,
        'A' : A,
        'Omega' : Omega,
        'ds' : ds
    }

    return res

def ic(D, Phi, x):
    return criterions_dict(D, Phi, x)['IC']

#def _fast_ic(J, DJ, DJS, DI, DIS, Phi, PhiS, x):
#    N = DJ.shape[0]
#    block_fun = lambda v : np.concatenate(
#        (PhiS(Phi(v[:N,:])) + DJ(v[N:,:]),
#         DJS(v[:N,:])),
#        axis=0)
#    block_op = LinearOperator(
#        (N+len(J),N+len(J)),
#        block_fun,
#        block_fun
#    )
#    A_fun = lambda u : cgs(
#        block_op,
#        np.concatenate((u,np.zeros((len(J),1))),axis=0)
#    )
#    tilde_Omega = lambda u : DI(u) - PhiS(Phi(A_fun(DI(u))))
#    Omega = lambda v : lsqr(DJ, tilde_Omega(v))
#
#    ds = np.sign(DIS(x))
#    Omega_x0 = Omega(ds)
#
#    proj = lambda w: DJS(lsqr(
#        lambda u : DJ(DJS(u)), DJ(w)
#    ))
#    prox_indic = lambda w, la: w - proj(w)
#
#    proxinf = dual_prox(l1ball_projection)
#    prox_obj = lambda x, la: -proxinf(Omega_x0 - x, la) + Omega_x0
#
#    w = douglas_rachford(prox_indic, prox_obj, np.zeros((len(J),1)))
#    return np.max(np.abs(Omega_x0 - w))
#
#
#def fast_ic(D, DS, Phi, PhiS, x):
#    I = (np.abs(DS(x)) > 1e-5).flatten()
#    J = ~I
#    return _fast_ic(J, D, DS, D, DS, Phi, PhiS, x)
#

