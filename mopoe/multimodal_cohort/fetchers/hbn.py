# -*- coding: utf-8 -*-
########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
########################################################################

"""
Module provides functions to prepare different datasets from HBN.
"""

# Imports
from .multiblock_fetcher import fetch_multiblock_wrapper

# Global parameters
DEFAULTS = {
    "multiblock": {
        "test_size": 0.2, "seed": 42,
        "stratify": ["age", "sex", "site"],
        "discretize": ["age"],
        "blocks": ["clinical", "rois"],
        "allow_missing_blocks": False,
    }
}


def make_all_fetchers(datasetdir):
    """ Compiles all the fetchers into a dictionnary.
    The difference with "make_fetchers" is that it
    include the multiblock fetcher.

    Parameters
    ----------
    datasetdir: string, default SAVING_FOLDER
        path to the folder in which to save the data

    Returns
    -------
    fetchers: dict
        dictionnary with name of the modality as keys and
        fetchers as values

    """
    fetchers = {}

    fetchers["multiblock"] = fetch_multiblock_wrapper(
        datasetdir=datasetdir, defaults=DEFAULTS["multiblock"])
    return fetchers

