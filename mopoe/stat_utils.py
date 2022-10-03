# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Utility functions.
"""

# Imports
import os
import types
import numpy as np
from scipy.stats.stats import kendalltau
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


def data2cmat(data):
    """ Compute pairwise (dis)similarity matrices.
    """
    if data.ndim > 2:
        return np.array([squareform(pdist(data[idx], metric="euclidean"))
                         for idx in range(len(data))])
    else:
        return squareform(pdist(data, metric="euclidean"))


def cmat2triu(arr):
    """ Get similarity matrix upper triangular.
    """
    assert np.ndim(arr) == 2, "Expect 2 dim similarity!"
    assert arr.shape[0] == arr.shape[1], "Expect square similarity!"
    n_dims = arr.shape[0]
    triu_vec = arr[np.triu_indices(n=n_dims, k=1)]
    return triu_vec


def vec2cmat(vec, data_scale="ratio", metric="euclidean"):
    """ Compute pairwise (dis)similarity matrice for a specific clinical
    characteristic vector.
    """
    vec = (vec - vec.min()) / (vec.max() - vec.min())
    if vec.ndim == 1:
        vec = np.vstack((vec, np.zeros(vec.shape))).transpose()
    cmat = squareform(pdist(vec, metric=metric).transpose())
    if data_scale == "ordinal":
        cmat[cmat != 0] = 1
    return cmat


def fit_rsa(cmat, ref_cmat, idxs=None):
    """ Compare dissimilarity matrix to the matrices for each individual
    characteristic using the Kendall rank correlation coefficient.
    """
    if cmat.ndim > 2:
        r = np.array([
            kendalltau(cmat2triu(cmat[idx][idxs, :][:, idxs]),
                       cmat2triu(ref_cmat))[0]
            for idx in range(10)])
        r = np.arctan(r)
        return r
    else:
        tau, pval = kendalltau(cmat2triu(cmat), cmat2triu(ref_cmat))
        return tau, pval
