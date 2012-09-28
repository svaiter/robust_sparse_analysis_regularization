# Robust Sparse Analysis Regularization #

This repository holds numerical experiments associated to the paper S. Vaiter, G. Peyré, C. Dossal, and J. Fadili, Robust Sparse Analysis Regularization, Submitted, 2011.

## Dependencies ##

The required dependencies to build the software are Python >= 2.6, Numpy >= 1.3, SciPy >= 0.7 and pyprox == 0.1.
IPython >=0.13 is required for the notebooks.
There is no install process.

## Retrieve grids ##

In the case you do not want to compute results for Fused Lasso Compressed Sensing (~2 days of computation each), you can retrieve them using the command

	cd npy/
	./download_npy_files