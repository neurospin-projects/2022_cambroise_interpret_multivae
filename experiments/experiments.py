#! /usr/bin/env python3
# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Organize the analysis steps.
"""

# System import
import fire
import workflow as wf
from analyze_avatars import analyze_avatars, assess_robustness, univariate_tests, evaluate_stability


fire.Fire({
    "train": wf.train_exp,
    "daa": wf.daa_exp,
    "anova": wf.anova_exp,
    "daa-plot-most-connected": wf.daa_plot_most_connected,
    "daa-plot-most-significant": wf.daa_plot_most_significant,
    "daa-plot-score-metric": wf.daa_plot_score_metric,
    "daa-plot-score-metric-strongest": wf.daa_plot_score_metric_strongest,
    "rsa": wf.rsa_exp,
    "rsa-plot": wf.rsa_plot_exp,
    "hist-plot": wf.hist_plot_exp,
    # "avatar-plot": wf.avatar_plot_exp,
    "daa-analysis": analyze_avatars,
    "daa-robustness": assess_robustness,
    "univariate-tests": univariate_tests,
    "daa-evaluate-stability": evaluate_stability
})