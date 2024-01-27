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
import analyze_avatars as aa
import stability
import stability_plots

fire.Fire({
    "train": wf.train_exp,
    "multiple-train": wf.multiple_train_exp,
    "retrain": wf.retrain_exp,
    "daa": wf.daa_exp,
    "anova": wf.anova_exp,
    "daa-plot-most-connected": wf.daa_plot_most_connected,
    "daa-plot-most-significant": wf.daa_plot_most_significant,
    "daa-plot-score-metric": wf.daa_plot_score_metric,
    "daa-plot-score-metric-strongest": wf.daa_plot_score_metric_strongest,
    "rsa": wf.rsa_exp,
    "multiple-rsa": wf.multiple_rsa_exps,
    "rsa-score-models": wf.score_models,
    "rsa-plot": wf.rsa_plot_exp,
    "hist-plot": wf.hist_plot_exp,
    # "avatar-plot": wf.avatar_plot_exp,
    "daa-analysis": aa.analyze_avatars,
    "daa-robustness": aa.assess_robustness,
    "univariate-tests": aa.univariate_tests,
    "daa-evaluate-stability": stability.evaluate_stability,
    "daa-validate-stability": stability.validate_stability,
    "daa-evaluate-stability-scaling": stability.evaluate_stability_scaling,
    "daa-study-heuristics": stability.study_heuristics,
    "daa-results-to-latex": stability_plots.generate_latex_associations_table,
    "daa-plot-most-connected-stable": stability_plots.daa_plot_most_connected_stable,
    "daa-plot-most-associated-stable": stability_plots.daa_plot_most_associated_stable,
    "daa-plot-metric-score-stable": stability_plots.daa_plot_metric_score_stable,
    "daa-plot-metric-score-sign-stable": stability_plots.daa_plot_metric_score_coefs_sign_stable,
    "daa-plot-metric-score-probs": stability_plots.plot_metric_score_stability_against_n_models,
    "daa-plot-metric-score-coefs": stability_plots.plot_metric_score_coefs_against_n_models,
    "permuted-daa": stability.permuted_daa_exp,
    "check-permuted-associations": stability.check_permutation_stable_associations,
})