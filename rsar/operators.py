"""
Common operators definitions
"""
# Author: Samuel Vaiter <samuel.vaiter@ceremade.dauphine.fr>
from __future__ import division
import numpy as np

def finite_diff_1d(n, bound='sym'):
    D = np.eye(n) - np.diag(np.ones((n-1,)),1)
    D = D[:,1:]
    return D