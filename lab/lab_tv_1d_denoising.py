"""
TODO write spec
"""
# Author: Samuel Vaiter <samuel.vaiter@ceremade.dauphine.fr>

# Settings
from __future__ import division
import numpy as np

from criterions import criterions_dict
from operators import finite_diff_1d

n = 16
n0 = n // 4
D = finite_diff_1d(n)
Phi = np.eye(n)

x = np.zeros((n,1))
x[n//2:,:] = 1
x[3*n//4:,:] = 2
crit = criterions_dict(D, Phi, x)

print crit['IC']
print crit['IC_noker']
print crit['wRC']