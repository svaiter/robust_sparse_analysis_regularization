"""
Common operators definitions
"""
# Author: Samuel Vaiter <samuel.vaiter@ceremade.dauphine.fr>
from __future__ import division
import numpy as np

def _divergence(x):
    pass

def finite_diff_1d(n, bound='sym', mode='explicit'):
    if mode == 'explicit':
        D = np.eye(n) - np.diag(np.ones((n-1,)),1)
        if bound == 'sym':
            D = D[:,1:]
        elif bound == 'per':
            pass
        else:
            raise Exception('Not a valid boundary condition')
        return D
#    else:
#        if bound == 'sym':
#            adj = lambda v : -v[:-1,:] + v[1:,:]
#            direct = _divergence
#            #adj = np.diff(x,axis=0)
#            return (direct, adj)
#        else:
#            raise Exception('Not implemented')

def fused_lasso_dict(n, eps=0.5, mode='explicit'):
    if mode == 'explicit':
        return np.concatenate((finite_diff_1d(n), eps * np.eye(n)), 1)
    else:
        pass