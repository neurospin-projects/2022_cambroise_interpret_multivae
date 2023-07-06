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
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import kendalltau, pearsonr
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


def vec2cmat(vec, categorical=False, metric="euclidean"):
    """ Compute pairwise (dis)similarity matrice for a specific clinical
    characteristic vector.
    """
    if not categorical:
        cmat = squareform(pdist(vec[:, None], metric=metric).transpose())
    else:
        cmat = (vec[:, None] != vec).astype(int)
    return cmat

def make_regression(df, x_name, y_name, other_cov_names=[], groups_name=None, method="fixed", other=None):
    """ Fit linear models with the wanted design
    """
    formula = "{} ~ {}".format(y_name, x_name)
    formula = " + ".join([formula] + other_cov_names)
    idx_of_beta = x_name
    subjects_betas = None
    if method == "fixed":
        est = sm.OLS.from_formula(formula, data=df)
    elif method == "mixed":
        est = sm.MixedLM.from_formula(formula, data=df, groups=groups_name)
    elif method == "hierarchical":
        lv1 = [[group_lab, sm.OLS.from_formula(formula, group_df).fit().params[x_name]]
               for group_lab, group_df in df.groupby(groups_name, sort=False)]
        # lv1_betas = [[group_lab, result.params[x_name]] for group_lab, result in lv1]
        # lv1_intercepts = [[group_lab, result.params["Intercept"]] for group_lab, result in lv1]
        lv1 = pd.DataFrame(lv1, columns=[groups_name, 'beta'])
        subjects_betas = lv1
        est = sm.OLS.from_formula("beta ~ 1", data=lv1)
        idx_of_beta = "Intercept"
    results = est.fit()
    # corr, pval = pearsonr([intercept for _, intercept in lv1_intercepts], other)
    # print([intercept for _, intercept in lv1_intercepts])
    # print(other.tolist())
    return results.pvalues[idx_of_beta], results.params[idx_of_beta], subjects_betas

def fit_rsa(cmat, ref_cmat, idxs=None):
    """ Compare dissimilarity matrix to the matrices for each individual
    characteristic using the Kendall rank correlation coefficient.
    """
    if cmat.ndim > 2:
        print("weird")
        r = np.array([
            kendalltau(cmat2triu(cmat[idx][idxs, :][:, idxs]),
                       cmat2triu(ref_cmat))[0]
            for idx in range(10)])
        r = np.arctan(r)
        return r
    else:
        tau, pval = kendalltau(cmat2triu(cmat), cmat2triu(ref_cmat))
        return tau, pval