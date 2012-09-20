from __future__ import division
import numpy as np

from rsar.criterions import ic
from rsar.operators import fused_lasso_dict

n = 256

def sig(n,eta=0.2,rho=0.25):
    mu = 0.5 - eta - rho
    x = np.zeros((n,1))
    x[np.floor(mu*n):np.floor(n*(mu+eta)),:] = 1
    x[np.floor(n*(0.5 + rho)):np.floor(n*(0.5 + rho + eta))] = -1
    return x

def dummy(c):
    (eps, ratio, eta, trial) = c
    D = fused_lasso_dict(n, eps=eps)
    x = sig(n,eta,rho=0.1)
    q = round(ratio*n)
    Phi = np.random.randn(q,n)
    return ic(D, Phi, x)
    
def grid_ratio_eta(eps_candidates, ratio_candidates, eta_candidates, trials):
    import itertools
    import multiprocessing as mp
    pos = itertools.product(eps_candidates, ratio_candidates, eta_candidates, range(trials))
    pool = mp.Pool()
    res_tmp = pool.map_async(dummy, pos)
    return res_tmp

trials_per_settings = 1000
eta_candidates = np.arange(1,np.floor(0.2*n)) / n
ratio_candidates = np.arange(round(n/2),n+1,2) / n
eps_candidates = [50/n,]

restmp=grid_ratio_eta(n, eps_candidates, ratio_candidates, eta_candidates, trials_per_settings).get()
restarr = np.asarray(restmp)
np.save('fused_final_full',restarr)