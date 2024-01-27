import os
import glob
import time
import copy
import pickle
import itertools
import collections
import torch
from types import SimpleNamespace
import numpy as np
from numpy.lib.format import open_memmap
import matplotlib.pyplot as plt
from matplotlib import colors, lines
from joblib import Parallel, delayed
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from multimodal_cohort.experiment import MultimodalExperiment
from multimodal_cohort.utils import extract_and_order_by
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from scipy.stats import pearsonr, combine_pvalues
from color_utils import (print_title, print_subtitle, print_text, print_result)
from daa_functions import (compute_significativity, compute_all_stability,
                           compute_all_associations, Heuristic, make_digital_avatars,
                           compute_daa_statistics, compute_all_stability_fast,
                           compute_all_results_stability)
from workflow import score_models


def evaluate_stability(dataset, datasetdir, outdir, runs=[],
                       metrics=["thickness", "meancurv", "area"],
                       scores=None, model_score_thrs=None,
                       stability_measure="product",
                       n_subjects=301, sampling=None, sample_latents=None,
                       ensemble_models=False):
    assert len(runs) > int(not ensemble_models)
    heuristics_params = {
        # "pvalues_vote": {"strategy": ["vote_prop-trust_level"], "vote_prop": [0.8, 0.85, 0.9, 0.95, 1], "trust_level": [1]},
        # "pvalues_prod": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31)), "thr": [5e-3, 1e-5, 1e-20, 1e-50, 1e-100], "var": [1, 0.5, 0.25]},
        # "pvalues_min": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "pvalues_coefs": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 50)), "thr": [1e-250, 1e-200, 1e-150, 1e-100, 1e-50, 1e-20], "var": [1, 0.5, 0.25]},
        # "pvalues_combine_fisher": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "pvalues_combine_pearson": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "pvalues_combine_tippett": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "pvalues_combine_stouffer": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "pvalues_combine_mudholkar_george": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "pvalues_mean": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31)), "thr": [1e-3, 1e-5, 1e-8], "var": [1, 0.5, 0.25]},
        "coefs_mean": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        "coefs_max": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31)), "thr": [1e-3, 1e-5, 1e-8], "var": [1, 0.5, 0.25]},
        # "coefs_weighted_mean_score": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "coefs_weighted_mean_rank": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
    }
    heuristics_params = {
        # "pvalues_vote": {"strategy": ["vote_prop-trust_level"], "vote_prop": [0.8, 0.85, 0.9, 0.95, 1], "trust_level": [1]},
        # "pvalues_prod": {"strategy": ["num", "num-thr"], "num": list(range(1, 31, 3)), "thr": [1e-20, 1e-50]},
        # "pvalues_min": {"strategy": ["num", "num-thr"], "num": list(range(1, 31, 3)), "thr": [1e-10]},
        # "pvalues_vote": {"strategy": ["vote_prop-trust_level"], "vote_prop": [0.8, 0.85, 0.9, 0.95, 1], "trust_level": [1]},
        # "pvalues_min": {"strategy": ["num"], "num": list(range(1, 21, 3))},
        # "pvalues_mean": {"strategy": ["num"], "num": list(range(1, 21, 3))},
        # "pvalues_coefs": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 50)), "thr": [1e-250, 1e-200, 1e-150, 1e-100, 1e-50, 1e-20], "var": [1, 0.5, 0.25]},
        # "pvalues_combine_fisher": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "pvalues_combine_pearson": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "pvalues_combine_tippett": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "pvalues_combine_stouffer": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "pvalues_combine_mudholkar_george": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "pvalues_mean": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31)), "thr": [1e-3, 1e-5, 1e-8], "var": [1, 0.5, 0.25]},
        "coefs_mean": {"strategy": ["num"], "num": [12]},#list(range(0, 25, 3))},
        # "coefs_max": {"strategy": ["num"], "num": [12]},#list(range(0, 25, 3))},
        # "coefs_weighted_mean_score": {"strategy": ["num"], "num": list(range(1, 31, 2))},
        # "coefs_weighted_mean_rank": {"strategy": ["num"], "num": list(range(1, 31, 2))},
        # "coefs_mean": {"strategy": ["thr", "num-thr"], "num": list(range(1, 31, 2)), "thr": [1e-5]},
        # "coefs_max": {"strategy": ["thr", "num-thr"], "num": list(range(1, 31, 2)), "thr": [1e-3]},
        # "coefs_weighted_mean_score": {"strategy": ["thr", "num-thr"], "num": list(range(1, 31, 2)), "thr": [1e-5]},
        # "coefs_weighted_mean_rank": {"strategy": ["thr", "num-thr"], "num": list(range(1, 31, 2)), "thr": [1e-5]},
    }

    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings("ignore")

    clinical_names = np.load(
            os.path.join(datasetdir, "clinical_names.npy"), allow_pickle=True)
    clinical_names = clinical_names.tolist()
    if scores is None:
        scores = clinical_names
    rois_names = np.load(
        os.path.join(datasetdir, "rois_names.npy"), allow_pickle=True)
    rois = np.array(
        list(dict.fromkeys([name.rsplit("_", 1)[0] for name in rois_names])))
    metadata = pd.read_table(
        os.path.join(datasetdir, "metadata_train.tsv"))
    metadata_columns = metadata.columns.tolist()

    additional_data = SimpleNamespace(metadata_columns=metadata_columns,
                                    clinical_names=clinical_names,
                                    rois_names=rois_names)

    model_selections = ["no_selection"]
    if model_score_thrs is not None:
        if type(model_score_thrs) not in (list, tuple):
            model_score_thrs = [model_score_thrs]
        for selection_param in model_score_thrs:
            model_selections.append(f"weight_aggregation_{selection_param}")
        #     model_selections.append(f"thr_score_{selection_param}")

    stabdir = os.path.join(outdir, runs[0], f"stability_with_{len(runs) - 1}_other_runs")
    if not os.path.exists(stabdir):
        os.makedirs(stabdir)
    local_stabdir = os.path.join(
        stabdir,
        (f"n_subjects_{n_subjects}_sampling_{sampling}_sample_latents_"
        f"{sample_latents}"))
    if ensemble_models:
        local_stabdir += "_ensemble_final_simplest"
    if not os.path.exists(local_stabdir):
        os.makedirs(local_stabdir)
    best_heuristic_path = os.path.join(local_stabdir, "best_heuristic_prior")

    if os.path.exists(best_heuristic_path):
        print("Loading existing stability results")
        with open(best_heuristic_path, 'rb') as f:
            best_heuristic_prior = pickle.load(f)
    else:
        best_heuristic_prior = {}
    print(best_heuristic_prior)
    default_thr = None
    # Computing heuristics with various parameters for each metric / score
    t = time.time()
    model_selections_to_compute = []
    global_results = []
    all_heuristics_params = []
    if ("no_selection" not in best_heuristic_prior.keys()):# or
        # (model_score_thrs is not None and "weight_aggregation" not in 
        #  best_heuristic_prior.keys())):
        global_result = compute_all_associations(dataset, datasetdir, outdir, runs,
                                                    heuristics_params, metrics,
                                                    scores, default_thr,
                                                    n_subjects=n_subjects,
                                                    sampling=sampling,
                                                    sample_latents=sample_latents,
                                                    ensemble_models=ensemble_models)
        global_results.append(global_result)
        all_heuristics_params.append(heuristics_params)
        model_selections_to_compute.append("no_selection")
    if model_score_thrs is not None:
        # if "weight_aggregation" not in best_heuristic_prior.keys():
        #     new_heuristics_params = {
        #         "coefs_weighted_mean_score": {"strategy": ["num"], "num": [12]},#list(range(0, 25, 3))},
        #         # "coefs_weighted_mean_rank": {"strategy": ["num"], "num": [12]},#list(range(0, 25, 3))},
        #         # "coefs_weighted_mean_score": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        #         # "coefs_weighted_mean_rank": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        #         }
        #     global_result_new = compute_all_associations(
        #         dataset, datasetdir, outdir, runs, new_heuristics_params, metrics,
        #         scores, default_thr, n_subjects=n_subjects, sampling=sampling,
        #         sample_latents=sample_latents, ensemble_models=ensemble_models)
        #     global_results.append(global_result_new)
        #     all_heuristics_params.append(new_heuristics_params)
        #     model_selections_to_compute.append("weight_aggregation")
        for thr_idx, threshold in enumerate(model_score_thrs):
            model_selection = model_selections[thr_idx + 1]
            if model_selection not in best_heuristic_prior.keys():
                new_heuristics_params = {
                    f"coefs_weighted_mean_{threshold}": {"strategy": ["num"], "num": [12]}}
                
                global_result_select = compute_all_associations(dataset, datasetdir, outdir,
                                                                runs, new_heuristics_params,
                                                                metrics, scores, default_thr,
                                                                n_subjects=n_subjects,
                                                                sampling=sampling,
                                                                sample_latents=sample_latents,
                                                                ensemble_models=ensemble_models)
                global_results.append(global_result_select)
                all_heuristics_params.append(new_heuristics_params)
                model_selections_to_compute.append(model_selection)

        # for thr_idx, threshold in enumerate(model_score_thrs):
        #     if model_selections[thr_idx + 2] not in best_heuristic_prior.keys():
        #         global_result_select = compute_all_associations(dataset, datasetdir, outdir,
        #                                                         runs, heuristics_params,
        #                                                         metrics, scores, threshold,
        #                                                         n_subjects=n_subjects,
        #                                                         sampling=sampling,
        #                                                         sample_latents=sample_latents,
        #                                                         ensemble_models=ensemble_models)
        #         global_results.append(global_result_select)
        #         all_heuristics_params.append(heuristics_params)
        #         model_selections_to_compute.append(model_selections[thr_idx + 2])

    print(time.time() - t)
    print("Associations computed. Computing stability...")
    print(f"For the {len(global_results)} uncomputed settings.")
    # Computing stability
    ideal_Ns = np.array(list(range(5, 25)))#np.sqrt(len(rois))
    ideal_Ns = np.array([12])
    to_compare = runs
    if ensemble_models:
        to_compare = range(int(len(global_results[0])))# * 0.8))
    comparisons = list(itertools.combinations(to_compare, 2))
    best_values = {"stability" : np.empty((len(global_results), len(ideal_Ns))),
                   "penalized_stability": np.empty((len(global_results), len(ideal_Ns))),
                   "stability_std" : np.zeros((len(global_results), len(ideal_Ns))),
                    "penalized_stability_std": np.zeros((len(global_results), len(ideal_Ns))),
                   "heuristic": np.empty((len(global_results),len(ideal_Ns)), dtype=object),
                   "strat_param" : np.empty((len(global_results), len(ideal_Ns)), dtype=object),
                   "daa_params": np.empty((len(global_results), len(ideal_Ns)), dtype=object)}

    best_values_per_metric_score = {
        "stability" : np.zeros((len(metrics), len(scores), len(global_results), len(ideal_Ns))),
        "penalized_stability": np.zeros((len(metrics), len(scores), len(global_results), len(ideal_Ns))),
        "stability_std" : np.zeros((len(metrics), len(scores), len(global_results), len(ideal_Ns))),
        "penalized_stability_std": np.zeros((len(metrics), len(scores), len(global_results), len(ideal_Ns))),
        "heuristic": np.empty((len(metrics), len(scores), len(global_results), len(ideal_Ns)), dtype=object),
        "strat_param" : np.empty((len(metrics), len(scores), len(global_results), len(ideal_Ns)), dtype=object),
        "daa_params": np.empty((len(metrics), len(scores), len(global_results), len(ideal_Ns)), dtype=object)
    }

    best_values_per_metric = {
        "stability" : np.zeros((len(metrics), len(global_results), len(ideal_Ns))),
        "penalized_stability": np.zeros((len(metrics), len(global_results), len(ideal_Ns))),
        "stability_std" : np.zeros((len(metrics), len(global_results), len(ideal_Ns))),
        "penalized_stability_std": np.zeros((len(metrics), len(global_results), len(ideal_Ns))),
        "heuristic": np.empty((len(metrics), len(global_results), len(ideal_Ns)), dtype=object),
        "strat_param" : np.empty((len(metrics), len(global_results), len(ideal_Ns)), dtype=object),
        "daa_params": np.empty((len(metrics), len(global_results), len(ideal_Ns)), dtype=object)
    }

    best_values_per_score = {
        "stability" : np.zeros((len(scores), len(global_results), len(ideal_Ns))),
        "penalized_stability": np.zeros((len(scores), len(global_results), len(ideal_Ns))),
        "stability_std" : np.zeros((len(scores), len(global_results), len(ideal_Ns))),
        "penalized_stability_std": np.zeros((len(scores), len(global_results), len(ideal_Ns))),
        "heuristic": np.empty((len(scores), len(global_results), len(ideal_Ns)), dtype=object),
        "strat_param" : np.empty((len(scores), len(global_results), len(ideal_Ns)), dtype=object),
        "daa_params": np.empty((len(scores), len(global_results), len(ideal_Ns)), dtype=object)
    }

    global_std = False
    variables = list(best_values.keys())
    std_variables = ["stability_std", "penalized_stability_std"]
    for var in std_variables:
        variables.remove(var)
    # all_stability_results = []
    # Compute penalized stability for each ideal_N value
    iterator = zip(global_results, all_heuristics_params, model_selections_to_compute)
    for result_idx, (result, heuristics_params, model_selection) in enumerate(
        iterator):
        stability_per_score_metric = {
            "daa_params": [], "heuristic": [], "strat_param": [], "metric": [],
            "score": [], "stability": [], "penalized_stability": [], "comparison": []}
        all_stability_per_score_metric = [
            copy.deepcopy(stability_per_score_metric) for _ in ideal_Ns]
        # if result_idx == 1 and "weight_aggregation" in model_selections_to_compute:
            # all_stability_per_score_metric = [
            #     copy.deepcopy(stabilities) for stabilities in all_stability_results[0]
            # ]
            # del all_stability_results
        product_of_params = list(
            itertools.product(ideal_Ns, comparisons, list(heuristics_params.keys())))
        print(heuristics_params)
        stabilities_per_score_metric = Parallel(n_jobs=-2, verbose=1)(
            delayed(compute_all_results_stability)(
                copy.deepcopy(stability_per_score_metric),
                (result[to_compare.index(comparison[0])],
                result[to_compare.index(comparison[1])]),
                comparison, heuristics_params, heuristic, ideal_N,
                scores, metrics, stability_measure)
                for ideal_N, comparison, heuristic in product_of_params)

        for param_idx, (ideal_N, _, _) in enumerate(product_of_params):
            ideal_N_idx = ideal_Ns.tolist().index(ideal_N)
            stability_per_score_metric = all_stability_per_score_metric[ideal_N_idx]
            local_stability_per_metric_score = stabilities_per_score_metric[param_idx]
            for key, value in stability_per_score_metric.items():
                value += local_stability_per_metric_score[key]
        # if model_selection == "no_selection":
        #     all_stability_results.append(all_stability_per_score_metric)
        # Find out the best ensembling strategy for each granularity and each prior
        for N_idx, ideal_N in enumerate(tqdm(ideal_Ns)):
            stability_per_score_metric = pd.DataFrame.from_dict(
                all_stability_per_score_metric[N_idx])
            # print(stability_per_score_metric.sort_values("penalized_stability", ascending=False))
            # print(final_stability.sort_values("penalized_stability", ascending=False))

            # Compute best values per (metric, score), metric and score w.r.t.
            # penalized stability
            best_stability_per_comparison = []
            best_pen_stability_per_comparison = []
            grouped_stability = stability_per_score_metric.groupby([
                "daa_params", "heuristic", "strat_param", "metric", 
                "score"
            ], as_index=False)
            grouped_stability_mean = grouped_stability.mean()
            for metric_idx, metric in enumerate(metrics):
                local_metric_idx = (grouped_stability_mean["metric"] == metric)
                for score_idx, score in enumerate(scores):
                    idx = ((local_metric_idx) & (grouped_stability_mean["score"] == score))
                    sorted_local_stability = grouped_stability_mean[idx].sort_values(
                        "penalized_stability", ascending=False)
                    for variable in variables:
                        best_values_per_metric_score[variable][
                            metric_idx, score_idx, result_idx, N_idx] = (
                            sorted_local_stability[variable].to_list()[0])
                    if global_std:
                        local_stability_std = grouped_stability.std()[idx]
                        for variable in std_variables:
                            idx = sorted_local_stability.index[0]
                            best_values_per_metric_score[variable][
                                metric_idx, score_idx, result_idx, N_idx] = (
                                local_stability_std.loc[idx, variable.replace("_std", "")].item())
                    else:
                        best_daa_params, best_heuristic, best_strat_param = sorted_local_stability[["daa_params", "heuristic", "strat_param"]].values[0]
                        best_indices = ((stability_per_score_metric["daa_params"] == best_daa_params) &
                                        (stability_per_score_metric["heuristic"] == best_heuristic) &
                                        (stability_per_score_metric["strat_param"] == best_strat_param) &
                                        (stability_per_score_metric["metric"] == metric) &
                                        (stability_per_score_metric["score"] == score))
                        best_stability_per_comparison.append(stability_per_score_metric.loc[best_indices, "stability"].values.tolist())
                        best_pen_stability_per_comparison.append(stability_per_score_metric.loc[best_indices, "penalized_stability"].values.tolist())
            if not global_std:
                for metric_idx, metric in enumerate(metrics):
                    for score_idx, score in enumerate(metrics):
                        best_values_per_metric_score["stability_std"][
                            metric_idx, score_idx, result_idx, N_idx] = (
                                np.array(best_stability_per_comparison).mean(0).std(0))
                        best_values_per_metric_score["penalized_stability_std"][
                            metric_idx, score_idx, result_idx, N_idx] = (
                                np.array(best_pen_stability_per_comparison).mean(0).std(0))
            final_stability = stability_per_score_metric.groupby(
                ["daa_params", "heuristic", "strat_param", "comparison"],
                as_index=False).mean()
            final_stability_std = final_stability.groupby(
                ["daa_params", "heuristic", "strat_param"],
                as_index=False).std()
            final_stability_mean = final_stability.groupby(
                ["daa_params", "heuristic", "strat_param"],
                as_index=False).mean()
            sorted_stability = final_stability_mean.sort_values(
                "penalized_stability", ascending=False)
            for variable in variables:
                best_values[variable][result_idx, N_idx] = (
                    sorted_stability[variable].to_list()[0])
            for variable in std_variables:
                idx = sorted_stability.index[0]
                best_values[variable][result_idx, N_idx] = (
                    final_stability_std.loc[idx, variable.replace("_std", "")].item())
            best_stability_per_comparison = []
            best_pen_stability_per_comparison = []
            local_stability = stability_per_score_metric.groupby(
                ["daa_params", "heuristic", "strat_param", "metric", "comparison"],
                as_index=False).mean()
            for metric_idx, metric in enumerate(metrics):
                idx = (local_stability["metric"] == metric)
                local_stability_mean = local_stability[idx].groupby(
                    ["daa_params", "heuristic", "strat_param", "metric"],
                    as_index=False).mean()
                sorted_local_stability = local_stability_mean.sort_values(
                    "penalized_stability", ascending=False)
                for variable in variables:
                    best_values_per_metric[variable][metric_idx, result_idx, N_idx] = (
                        sorted_local_stability[variable].to_list()[0])
                if global_std:
                    std_idx = stability_per_score_metric["metric"] == metric
                    local_stability_std = stability_per_score_metric[std_idx].groupby(
                        ["daa_params", "heuristic", "strat_param", "metric"],
                        as_index=False).std()
                    idx = sorted_local_stability.index[0]
                    for variable in std_variables:
                        best_values_per_metric[variable][metric_idx, result_idx, N_idx] = (
                            local_stability_std.loc[idx, variable.replace("_std", "")].item())
                else:
                    best_daa_params, best_heuristic, best_strat_param = sorted_local_stability[["daa_params", "heuristic", "strat_param"]].values[0]
                    best_indices = ((local_stability["daa_params"] == best_daa_params) &
                                    (local_stability["heuristic"] == best_heuristic) &
                                    (local_stability["strat_param"] == best_strat_param) &
                                    (local_stability["metric"] == metric))
                    best_stability_per_comparison.append(local_stability.loc[best_indices, "stability"].values.tolist())
                    best_pen_stability_per_comparison.append(local_stability.loc[best_indices, "penalized_stability"].values.tolist())

            if not global_std:
                for metric_idx, metric in enumerate(metrics):
                    best_values_per_metric["stability_std"][metric_idx, result_idx, N_idx] = np.array(best_stability_per_comparison).mean(0).std(0)
                    best_values_per_metric["penalized_stability_std"][metric_idx, result_idx, N_idx] = np.array(best_pen_stability_per_comparison).mean(0).std(0)
            best_stability_per_comparison = []
            best_pen_stability_per_comparison = []
            local_stability = stability_per_score_metric.groupby(
                ["daa_params", "heuristic", "strat_param", "score", "comparison"],
                as_index=False).mean()
            for score_idx, score in enumerate(scores):
                idx = (local_stability["score"] == score)
                local_stability_mean = local_stability[idx].groupby(
                    ["daa_params", "heuristic", "strat_param", "score"],
                    as_index=False).mean()
                sorted_local_stability = local_stability_mean.sort_values(
                    "penalized_stability", ascending=False)
                for variable in variables:
                    best_values_per_score[variable][score_idx, result_idx, N_idx] = (
                        sorted_local_stability[variable].to_list()[0])
                if global_std:
                    std_idx = stability_per_score_metric["score"] == score
                    local_stability_std = stability_per_score_metric[std_idx].groupby(
                        ["daa_params", "heuristic", "strat_param", "score"],
                        as_index=False).std()
                    idx = sorted_stability.index[0]
                    for variable in std_variables:
                        best_values_per_score[variable][score_idx, result_idx, N_idx] = (
                            local_stability_std.loc[idx, variable.replace("_std", "")].item())
                else:
                    best_daa_params, best_heuristic, best_strat_param = sorted_local_stability[["daa_params", "heuristic", "strat_param"]].values[0]
                    best_indices = ((local_stability["daa_params"] == best_daa_params) &
                                    (local_stability["heuristic"] == best_heuristic) &
                                    (local_stability["strat_param"] == best_strat_param) &
                                    (local_stability["score"] == score))
                    best_stability_per_comparison.append(local_stability.loc[best_indices, "stability"].values.tolist())
                    best_pen_stability_per_comparison.append(local_stability.loc[best_indices, "penalized_stability"].values.tolist())

            if not global_std:
                for score_idx, score in enumerate(scores):
                    best_values_per_score["stability_std"][score_idx, result_idx, N_idx] = np.array(best_stability_per_comparison).mean(0).std(0)
                    best_values_per_score["penalized_stability_std"][score_idx, result_idx, N_idx] = np.array(best_pen_stability_per_comparison).mean(0).std(0)
    
        # Find out the best prior and ensembling strategy for each granularity, and store it
        best_heuristic_prior[model_selection] = {"metric_score": {}, "metric": {}, "score": {}}
        for metric_idx, metric in enumerate(metrics):
            for score_idx, score in enumerate(scores):
                local_values = {}
                for variable in variables + std_variables:
                    local_values[variable] = best_values_per_metric_score[variable][metric_idx, score_idx, result_idx]
                
                max_penalized_stability = round(local_values["penalized_stability"].max(), 5)
                best_pen_N_idx = np.argwhere(local_values["penalized_stability"].round(5) ==
                                            max_penalized_stability).flatten().max()
               
                best_pen_N = ideal_Ns[best_pen_N_idx]
                best_pen_stab_std = local_values["penalized_stability_std"][best_pen_N_idx]
                best_pen_heuristic = local_values["heuristic"][best_pen_N_idx]
                best_pen_param = local_values["strat_param"][best_pen_N_idx]
                best_stab = local_values["stability"][best_pen_N_idx]
                best_stab_std = local_values["stability_std"][best_pen_N_idx]
                best_heuristic_prior[model_selection]["metric_score"][
                    f"{metric}_{score}"] = (
                        max_penalized_stability, best_pen_stab_std, best_stab, best_stab_std, best_pen_N, best_pen_heuristic, best_pen_param)
                # print(f"Best penalized stability for {metric} and {score} : {best_pen_stab} for N_pen in {best_pen_N} and coef mean with {best_pen_param}.")
    
        for metric_idx, metric in enumerate(metrics):
            local_values = {}
            for variable in variables + std_variables:
                local_values[variable] = best_values_per_metric[variable][metric_idx, result_idx]

            max_penalized_stability = round(local_values["penalized_stability"].max(), 5)
            best_pen_N_idx = np.argwhere(local_values["penalized_stability"].round(5) ==
                                        max_penalized_stability).flatten().max()
          
            best_pen_N = ideal_Ns[best_pen_N_idx]
            best_pen_stab_std = local_values["penalized_stability_std"][best_pen_N_idx]
            best_pen_param = local_values["strat_param"][best_pen_N_idx]
            best_pen_heuristic = local_values["heuristic"][best_pen_N_idx]
            best_stab = local_values["stability"][best_pen_N_idx]
            best_stab_std = local_values["stability_std"][best_pen_N_idx]
            best_heuristic_prior[model_selection]["metric"][
                f"{metric}"] = (
                    max_penalized_stability, best_pen_stab_std, best_stab, best_stab_std, best_pen_N, best_pen_heuristic, best_pen_param)
            # print(f"Best average penalized stability for {metric} : {best_pen_stab} for N_pen in {best_pen_N} and {best_pen_heuristic} with {best_pen_param} and daa params {best_pen_daa}.")
        
        for score_idx, score in enumerate(scores):
            local_values = {}
            for variable in variables + std_variables:
                local_values[variable] = best_values_per_score[variable][score_idx, result_idx]

            max_penalized_stability = round(local_values["penalized_stability"].max(), 5)
            best_pen_N_idx = np.argwhere(local_values["penalized_stability"].round(5) ==
                                        max_penalized_stability).flatten().max()
           
            best_pen_N = ideal_Ns[best_pen_N_idx]
            best_pen_stab_std = local_values["penalized_stability_std"][best_pen_N_idx]
            best_pen_param = local_values["strat_param"][best_pen_N_idx]
            best_pen_heuristic = local_values["heuristic"][best_pen_N_idx]
            best_stab = local_values["stability"][best_pen_N_idx]
            best_stab_std = local_values["stability_std"][best_pen_N_idx]
            best_heuristic_prior[model_selection]["score"][
                f"{score}"] = (
                    max_penalized_stability, best_pen_stab_std, best_stab, best_stab_std, best_pen_N, best_pen_heuristic, best_pen_param)
            # print(f"Best average penalized stability for {score} : {best_pen_stab} for N_pen in {best_pen_N} and {best_pen_heuristic} with {best_pen_param} and daa params {best_pen_daa}.")

        local_values = {}
        for variable in variables + std_variables:
            local_values[variable] = best_values[variable][result_idx]
        
        max_penalized_stability = round(local_values["penalized_stability"].max(), 5)
        best_pen_N_idx = np.argwhere(local_values["penalized_stability"].round(5) ==
                                     max_penalized_stability).flatten().max()
        
        best_pen_N = ideal_Ns[best_pen_N_idx]
        best_pen_stab_std = local_values["penalized_stability_std"][best_pen_N_idx]
        best_pen_param = local_values["strat_param"][best_pen_N_idx]
        best_pen_heuristic = local_values["heuristic"][best_pen_N_idx]
        best_stab = local_values["stability"][best_pen_N_idx]
        best_stab_std = local_values["stability_std"][best_pen_N_idx]
        best_heuristic_prior[model_selection]["overall"] = (
            max_penalized_stability, best_pen_stab_std, best_stab, best_stab_std, best_pen_N, best_pen_heuristic, best_pen_param)
        best_heuristic_prior[model_selection]["overall_mix"] = (
            max_penalized_stability, best_pen_stab_std, best_stab, best_stab_std, best_pen_N, best_pen_heuristic, best_pen_param)
        with open(best_heuristic_path, 'wb+') as f:
            pickle.dump(best_heuristic_prior, f)
        print(f"Best heuristic saved for newly computed stability with "
              f"{model_selection}.")
    # Plot stability for each case
    plot_stability = True
    plot_heuristic_hist = False
    model_selection_idx = 0
    std_scaling = 1
    if plot_stability:
        fig, ax = plt.subplots(figsize=(12, 9))
        handles = []
        for select_idx, model_selection in enumerate(model_selections_to_compute):
            color = list(colors.TABLEAU_COLORS)[select_idx]
            handle = ax.plot(ideal_Ns, best_values["stability"][select_idx], label=model_selection, c=color)
            ax.fill_between(ideal_Ns, best_values["stability"][select_idx] - std_scaling * best_values["stability_std"][select_idx],
                            best_values["stability"][select_idx] + std_scaling * best_values["stability_std"][select_idx],
                            color=color, alpha=.1)
            ax.plot(ideal_Ns, best_values["penalized_stability"][select_idx], c=color, ls="--")
            handles += handle

        line_stab = lines.Line2D([0], [0], label='raw', color='k', ls="-")
        line_pen_stab = lines.Line2D([0], [0], label='penalized', color='k', ls="--")

        first_legend = ax.legend(handles=handles, loc='lower right', title="Setting")
        ax.add_artist(first_legend)
        ax.legend(handles=[line_stab, line_pen_stab], title="Stability", loc="upper right")
        ax.set_title("Best stability when varying N*")
        for metric_idx, metric in enumerate(metrics):
            fig, ax = plt.subplots(figsize=(12, 9))
            ax.set_title(f"Stability for {metric}")
            handles = []
            for score_idx, score in enumerate(scores):
                color = list(colors.TABLEAU_COLORS)[score_idx]
                handle = ax.plot(ideal_Ns, best_values_per_metric_score["stability"][metric_idx, score_idx, model_selection_idx],
                                label=score, ls="-", c=color)
                ax.fill_between(ideal_Ns, (best_values_per_metric_score["stability"][metric_idx, score_idx, model_selection_idx] 
                                    - std_scaling * best_values_per_metric_score["stability_std"][metric_idx, score_idx, model_selection_idx]),
                                (best_values_per_metric_score["stability"][metric_idx, score_idx, model_selection_idx] + std_scaling *
                                 best_values_per_metric_score["stability_std"][metric_idx, score_idx, model_selection_idx]),
                                color=color, alpha=.1)
                ax.plot(ideal_Ns, best_values_per_metric_score["penalized_stability"][metric_idx, score_idx, model_selection_idx],
                        ls="--", c=color)
                handles += handle
            
            # create manual symbols for legend
            line_stab = lines.Line2D([0], [0], label='raw', color='k', ls="-")
            line_pen_stab = lines.Line2D([0], [0], label='penalized', color='k', ls="--")

            handle = ax.plot(ideal_Ns, best_values_per_metric["stability"][metric_idx, model_selection_idx],
                            label="Average", ls="-", c="k", lw=3)
            ax.fill_between(ideal_Ns, (best_values_per_metric["stability"][metric_idx, model_selection_idx] 
                                    - std_scaling * best_values_per_metric["stability_std"][metric_idx, model_selection_idx]),
                                (best_values_per_metric["stability"][metric_idx, model_selection_idx] + std_scaling *
                                 best_values_per_metric["stability_std"][metric_idx, model_selection_idx]),
                                color="k", alpha=.1)
            ax.plot(ideal_Ns, best_values_per_metric["penalized_stability"][metric_idx, model_selection_idx],
                        ls="--", c="k", lw=3)
            handles += handle

            first_legend = ax.legend(handles=handles, loc='lower right', title="Score")
            ax.add_artist(first_legend)
            ax.legend(handles=[line_stab, line_pen_stab], title="Stability", loc="upper right")


        for score_idx, score in enumerate(scores):
            fig, ax = plt.subplots(figsize=(12, 9))
            ax.set_title(f"Stability for {score}")
            handles = []
            for metric_idx, metric in enumerate(metrics):
                color = list(colors.TABLEAU_COLORS)[metric_idx]
                handle = ax.plot(ideal_Ns, best_values_per_metric_score["stability"][metric_idx, score_idx, model_selection_idx],
                                 label=metric, ls="-", c=color)
                ax.fill_between(ideal_Ns, (best_values_per_metric_score["stability"][metric_idx, score_idx, model_selection_idx] 
                                    - std_scaling * best_values_per_metric_score["stability_std"][metric_idx, score_idx, model_selection_idx]),
                                (best_values_per_metric_score["stability"][metric_idx, score_idx, model_selection_idx] + std_scaling *
                                 best_values_per_metric_score["stability_std"][metric_idx, score_idx, model_selection_idx]),
                                color=color, alpha=.1)
                ax.plot(ideal_Ns, best_values_per_metric_score["penalized_stability"][metric_idx, score_idx, model_selection_idx],
                        ls="--", c=color)
                handles += handle

            # create manual symbols for legend
            line_stab = lines.Line2D([0], [0], label='raw', color='k', ls="-")
            line_pen_stab = lines.Line2D([0], [0], label='penalized', color='k', ls="--")

            handle = ax.plot(ideal_Ns, best_values_per_score["stability"][score_idx, model_selection_idx],
                            label="Average", ls="-", c="k", lw=3)
            ax.fill_between(ideal_Ns, (best_values_per_score["stability"][score_idx, model_selection_idx] 
                                    - std_scaling * best_values_per_score["stability_std"][score_idx, model_selection_idx]),
                                (best_values_per_score["stability"][score_idx, model_selection_idx] + std_scaling *
                                 best_values_per_score["stability_std"][score_idx, model_selection_idx]),
                                color="k", alpha=.1)
            ax.plot(ideal_Ns, best_values_per_score["penalized_stability"][score_idx, model_selection_idx],
                        ls="--", c="k", lw=3)
            handles += handle
            
            first_legend = ax.legend(handles=handles, title="Metric", loc="lower right")
            ax.add_artist(first_legend)
            ax.legend(handles=[line_stab, line_pen_stab], title="Stability", loc="upper right")
    
    if plot_heuristic_hist:
        plt.figure(figsize=(24, 16))
        plt.hist(best_values["strat_param"])
        plt.xticks(rotation = 315)
        plt.subplots_adjust(bottom=0.3)
        plt.title("Histogram of best heuristics and params on average")
        for metric_idx, metric in enumerate(metrics):
            plt.figure(figsize=(24, 16))
            plt.title(f"Histogram of best heuristic and param for {metric} on average")
            plt.hist(best_values_per_metric["strat_param"][metric_idx])
            plt.xticks(rotation = 315)
            plt.subplots_adjust(bottom=0.3)
        for score_idx, score in enumerate(scores):
            plt.figure(figsize=(24, 16))
            plt.title(f"Histogram of best heuristic and param for {score} on average")
            plt.hist(best_values_per_score["strat_param"][score_idx])
            plt.xticks(rotation = 315)
            plt.subplots_adjust(bottom=0.3)
        plt.figure(figsize=(24, 16))
        plt.title(f"Histogram of best heuristic and param for {score} accross metrics and scores")
        plt.hist(best_values_per_metric_score["strat_param"].reshape(
            (-1, best_values_per_metric_score["strat_param"].shape[-1])))
        plt.xticks(rotation = 315)
        plt.subplots_adjust(bottom=0.3)
    plt.show()
    # Print best result for each case

    import pprint
    # Prints the nicely formatted dictionary
    pprint.pprint(best_heuristic_prior)


def validate_stability(dataset, datasetdir, outdir, validation_runs=[], runs=[], 
                       metrics=["thickness", "meancurv", "area"],
                       scores=None, model_score_thrs=None,
                       stability_measure="product",
                       n_subjects=301, sampling=None, sample_latents=None,
                       ensemble_models=False):

    clinical_names = np.load(
            os.path.join(datasetdir, "clinical_names.npy"), allow_pickle=True)
    clinical_names = clinical_names.tolist()
    if scores is None:
        scores = clinical_names
    rois_names = np.load(
        os.path.join(datasetdir, "rois_names.npy"), allow_pickle=True)
    rois = np.array(
        list(dict.fromkeys([name.rsplit("_", 1)[0] for name in rois_names])))
    metadata = pd.read_table(
        os.path.join(datasetdir, "metadata_train.tsv"))
    metadata_columns = metadata.columns.tolist()

    ref_run = runs[0]
    other_runs = list(runs).copy()
    other_runs.remove(ref_run)
    stabdir = os.path.join(outdir, ref_run, f"stability_with_{len(other_runs)}_other_runs")
    if not os.path.exists(stabdir):
        raise ValueError("the runs you provide need to have stability computed with each other")
    local_stabdir = os.path.join(
        stabdir,
        (f"n_subjects_{n_subjects}_sampling_{sampling}_sample_latents_"
            f"{sample_latents}"))
    if ensemble_models:
        local_stabdir += "_ensemble_final_simplest"
    if not os.path.exists(local_stabdir):
        raise ValueError("the two runs you provide need to have stability computed with each other with provided arguments")
    with open(os.path.join(local_stabdir, "best_heuristic_prior"), 'rb') as f:
        best_heuristic_prior = pickle.load(f)
    best_heuristic_priors = best_heuristic_prior
    global_results = []
    to_compare = validation_runs
    if len(to_compare) == 0:
        to_compare = runs
    # if ensemble_models:
    #     to_compare = range(int(len(global_results[0]) * 0.8), len(global_results[0]))
    comparisons = list(itertools.combinations(to_compare, 2))

    heuristics_params = {
        # "pvalues_vote": {"strategy": ["vote_prop-trust_level"], "vote_prop": [0.8, 0.85, 0.9, 0.95, 1], "trust_level": [1]},
        # "pvalues_min": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "pvalues_mean": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31))},
        "coefs_mean": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        "coefs_max": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31)), "thr": [1e-3, 1e-5, 1e-8], "var": [1, 0.5, 0.25]},
    }
    # Computing heuristics with various parameters for each metric / score

    model_selections = ["no_selection"]
    if model_score_thrs is not None:
        if type(model_score_thrs) not in (list, tuple):
            model_score_thrs = [model_score_thrs]
        for selection_param in model_score_thrs:
            model_selections.append(f"weight_aggregation_{selection_param}")
        #     model_selections.append(f"thr_score_{selection_param}")
    if not ensemble_models:
        global_result = compute_all_associations(dataset, datasetdir, outdir, to_compare,
                                                heuristics_params, metrics,
                                                scores, None,
                                                n_subjects=n_subjects,
                                                sampling=sampling,
                                                sample_latents=sample_latents,
                                                ensemble_models=ensemble_models)
        global_results = [global_result]
        if model_score_thrs is not None:
            for threshold in model_score_thrs:
                # model_indices = np.array(model_indices).T
                # print(model_indices.shape)
                global_result_select = compute_all_associations(dataset, datasetdir, outdir,
                                                                local_runs, heuristics_params,
                                                                metrics, scores, threshold,
                                                                n_subjects=n_subjects,
                                                                sampling=sampling,
                                                                sample_latents=sample_latents,
                                                                ensemble_models=ensemble_models)
                global_results.append(global_result_select)

            heuristics_params.update({
                "coefs_weighted_mean_score": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
                "coefs_weighted_mean_rank": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
                })
            global_result_new = compute_all_associations(
                dataset, datasetdir, outdir, local_runs, heuristics_params, metrics,
                scores, None, n_subjects=n_subjects, sampling=sampling,
                sample_latents=sample_latents, ensemble_models=ensemble_models)
            global_results.append(global_result_new)

    values = {"stability": np.zeros((len(comparisons), len(model_selections))),
              "penalized_stability" : np.empty((len(comparisons), len(model_selections))),
              "stability_std": np.zeros((len(comparisons), len(model_selections))),
              "penalized_stability_std": np.zeros((len(comparisons), len(model_selections))),
              "validation_stability": np.empty((len(comparisons), len(model_selections))),
              "validation_penalized_stability":np.empty((len(comparisons), len(model_selections))),
              "validation_stability_std": np.zeros((len(model_selections))),
              "validation_penalized_stability_std": np.zeros((len(model_selections))),
              "heuristic": np.empty((len(comparisons), len(model_selections)), dtype=object),
              "strat_param" : np.empty((len(comparisons), len(model_selections)), dtype=object),
              "prior": np.empty((len(comparisons), len(model_selections)), dtype=int),
    }

    values_per_metric_score = {
        "stability": np.zeros((len(metrics), len(scores), len(comparisons), len(model_selections))),
        "penalized_stability" : np.zeros((len(metrics), len(scores), len(comparisons), len(model_selections))),
        "stability_std": np.zeros((len(metrics), len(scores), len(comparisons), len(model_selections))),
        "penalized_stability_std": np.zeros((len(metrics), len(scores), len(comparisons), len(model_selections))),
        "validation_stability": np.zeros((len(metrics), len(scores), len(comparisons), len(model_selections))),
        "validation_penalized_stability":np.zeros((len(metrics), len(scores), len(comparisons), len(model_selections))),
        "validation_stability_std": np.zeros((len(model_selections))),
        "validation_penalized_stability_std": np.zeros((len(model_selections))),
        "heuristic": np.empty((len(metrics), len(scores), len(comparisons), len(model_selections)), dtype=object),
        "strat_param" : np.empty((len(metrics), len(scores), len(comparisons), len(model_selections)), dtype=object),
        "prior": np.empty((len(metrics), len(scores), len(comparisons), len(model_selections)), dtype=int),
    }

    values_per_metric = {
        "stability": np.zeros((len(metrics), len(comparisons), len(model_selections))),
        "penalized_stability" : np.zeros((len(metrics), len(comparisons), len(model_selections))),
        "stability_std": np.zeros((len(metrics), len(comparisons), len(model_selections))),
        "penalized_stability_std": np.zeros((len(metrics), len(comparisons), len(model_selections))),
        "validation_stability": np.zeros((len(metrics), len(comparisons), len(model_selections))),
        "validation_penalized_stability": np.zeros((len(metrics), len(comparisons), len(model_selections))),
        "validation_stability_std": np.zeros((len(model_selections))),
        "validation_penalized_stability_std": np.zeros((len(model_selections))),
        "heuristic": np.empty((len(metrics), len(comparisons), len(model_selections)), dtype=object),
        "strat_param" : np.empty((len(metrics), len(comparisons), len(model_selections)), dtype=object),
        "prior": np.empty((len(metrics), len(comparisons), len(model_selections)), dtype=int),
    }

    values_per_score = {
        "stability": np.zeros((len(scores), len(comparisons), len(model_selections))),
        "penalized_stability" : np.zeros((len(scores), len(comparisons), len(model_selections))),
        "stability_std": np.zeros((len(scores), len(comparisons), len(model_selections))),
        "penalized_stability_std": np.zeros((len(scores), len(comparisons), len(model_selections))),
        "validation_stability": np.zeros((len(scores), len(comparisons), len(model_selections))),
        "validation_penalized_stability": np.zeros((len(scores), len(comparisons), len(model_selections))),
        "validation_stability_std": np.zeros((len(model_selections))),
        "validation_penalized_stability_std": np.zeros((len(model_selections))),
        "heuristic": np.empty((len(scores), len(comparisons), len(model_selections)), dtype=object),
        "strat_param" : np.empty((len(scores), len(comparisons), len(model_selections)), dtype=object),
        "prior": np.empty((len(scores), len(comparisons), len(model_selections)), dtype=int),
    }

    variables = list(values.keys())

    # Compute penalized stability for each ideal_N value
    all_stabilities = {"overall": [[] for _ in model_selections],
                       "metric": [[] for _ in model_selections],
                       "score": [[] for _ in model_selections]}
    all_pen_stabilities = {"overall": [[] for _ in model_selections],
                           "metric": [[] for _ in model_selections],
                           "score": [[] for _ in model_selections]}
    for comparison_idx, comparison in enumerate(comparisons):
        run_0, run_1 = comparison
        run_0_idx = runs.index(run_0)
        run_1_idx = runs.index(run_1)
        local_results = []
        for result_idx, model_selection in enumerate(model_selections):
            if not ensemble_models:
                local_result = global_results[result_idx]
                local_results.append((local_result[run_0_idx], local_result[run_1_idx]))
            for metric_idx, metric in enumerate(metrics):
                for score_idx, score in enumerate(scores):
                    score_metric_heuristic_prior = best_heuristic_priors[
                        model_selections[result_idx]]["metric_score"][f"{metric}_{score}"]
                    pen_stab, pen_stab_std, stab, stab_std, prior, heuristic, strat_param = score_metric_heuristic_prior
                    values_per_metric_score["stability"][metric_idx, score_idx, comparison_idx, result_idx] = stab
                    values_per_metric_score["penalized_stability"][metric_idx, score_idx, comparison_idx, result_idx] = pen_stab
                    values_per_metric_score["stability_std"][metric_idx, score_idx, comparison_idx, result_idx] = stab_std
                    values_per_metric_score["penalized_stability_std"][metric_idx, score_idx, comparison_idx, result_idx] = pen_stab_std
                    values_per_metric_score["prior"][metric_idx, score_idx, comparison_idx, result_idx] = prior
                    values_per_metric_score["heuristic"][metric_idx, score_idx, comparison_idx, result_idx] = heuristic
                    values_per_metric_score["strat_param"][metric_idx, score_idx, comparison_idx, result_idx] = strat_param
                    if not ensemble_models:
                        daa_params = f"{sampling}_{sample_latents}"
                        local_stability_per_metric_score = compute_all_stability_fast(
                            local_results, daa_params, heuristic, strat_param,
                            prior, [metric], [score], stability_measure)
                        values_per_metric_score["validation_stability"][
                            metric_idx, score_idx, comparison_idx, result_idx] = (
                                round(local_stability_per_metric_score[
                                    "stability"][0], 5)
                            )
                        values_per_metric_score["validation_penalized_stability"][
                            metric_idx, score_idx, comparison_idx, result_idx] = (
                                round(local_stability_per_metric_score[
                                    "penalized_stability"][0], 5)
                            )
            for metric_idx, metric in enumerate(metrics):
                metric_heuristic_prior = best_heuristic_priors[
                    model_selections[result_idx]]["metric"][f"{metric}"]
                pen_stab, pen_stab_std, stab, stab_std, prior, heuristic, strat_param = metric_heuristic_prior
                values_per_metric["stability"][metric_idx, comparison_idx, result_idx] = stab
                values_per_metric["penalized_stability"][metric_idx, comparison_idx, result_idx] = pen_stab
                values_per_metric["stability_std"][metric_idx, comparison_idx, result_idx] = stab_std
                values_per_metric["penalized_stability_std"][metric_idx, comparison_idx, result_idx] = pen_stab_std
                values_per_metric["prior"][metric_idx, comparison_idx, result_idx] = prior
                values_per_metric["heuristic"][metric_idx, comparison_idx, result_idx] = heuristic
                values_per_metric["strat_param"][metric_idx, comparison_idx, result_idx] = strat_param
                if not ensemble_models:
                    daa_params = f"{sampling}_{sample_latents}"
                    local_stability_per_metric = compute_all_stability_fast(
                        local_results, daa_params, heuristic, strat_param,
                        prior, [metric], scores, stability_measure)
                    values_per_metric["validation_stability"][
                        metric_idx, comparison_idx, result_idx] = (
                            np.mean(local_stability_per_metric[
                                "stability"]).round(5)
                        )
                    values_per_metric["validation_penalized_stability"][
                        metric_idx, comparison_idx, result_idx] = (
                            np.mean(local_stability_per_metric[
                                "penalized_stability"]).round(5)
                        )
                    all_stabilities["metric"][result_idx].append(
                        local_stability_per_metric["stability"])
                    all_pen_stabilities["metric"][result_idx].append(
                        local_stability_per_metric["penalized_stability"])

            for score_idx, score in enumerate(scores):
                score_heuristic_prior = best_heuristic_priors[
                    model_selections[result_idx]]["score"][f"{score}"]
                pen_stab, pen_stab_std, stab, stab_std, prior, heuristic, strat_param = score_heuristic_prior
                values_per_score["stability"][score_idx, comparison_idx, result_idx] = stab
                values_per_score["penalized_stability"][score_idx, comparison_idx, result_idx] = pen_stab
                values_per_score["stability_std"][score_idx, comparison_idx, result_idx] = stab_std
                values_per_score["penalized_stability_std"][score_idx, comparison_idx, result_idx] = pen_stab_std
                values_per_score["prior"][score_idx, comparison_idx, result_idx] = prior
                values_per_score["heuristic"][score_idx, comparison_idx, result_idx] = heuristic
                values_per_score["strat_param"][score_idx, comparison_idx, result_idx] = strat_param
                if not ensemble_models:
                    daa_params = f"{sampling}_{sample_latents}"
                    local_stability_per_score = compute_all_stability_fast(
                        local_results, daa_params, heuristic, strat_param,
                        prior, metrics, [score], stability_measure)
                    values_per_score["validation_stability"][
                        score_idx, comparison_idx, result_idx] = (
                            np.mean(local_stability_per_score[
                                "stability"]).round(5)
                        )
                    values_per_score["validation_penalized_stability"][
                        score_idx, comparison_idx, result_idx] = (
                            np.mean(local_stability_per_score[
                                "penalized_stability"]).round(5)
                        )
                    all_stabilities["score"][result_idx].append(
                        local_stability_per_score["stability"])
                    all_pen_stabilities["score"][result_idx].append(
                        local_stability_per_score["penalized_stability"])
                
            overall_heuristic_prior = best_heuristic_priors[
                    model_selections[result_idx]]["overall"]
            pen_stab, pen_stab_std, stab, stab_std, prior, heuristic, strat_param = overall_heuristic_prior
            values["stability"][comparison_idx, result_idx] = stab
            values["penalized_stability"][comparison_idx, result_idx] = pen_stab
            values["stability_std"][comparison_idx, result_idx] = stab_std
            values["penalized_stability_std"][comparison_idx, result_idx] = pen_stab_std
            values["prior"][comparison_idx, result_idx] = prior
            values["heuristic"][comparison_idx, result_idx] = heuristic
            values["strat_param"][comparison_idx, result_idx] = strat_param
            if not ensemble_models:
                daa_params = f"{sampling}_{sample_latents}"
                local_stability = compute_all_stability_fast(
                    local_results, daa_params, heuristic, strat_param,
                    prior, metrics, scores, stability_measure)
                values["validation_penalized_stability"][comparison_idx, result_idx] = (
                        np.mean(local_stability["penalized_stability"]).round(5)
                    )
                values["validation_stability"][comparison_idx, result_idx] = (
                        np.mean(local_stability["stability"]).round(5)
                    )
                all_stabilities["overall"][result_idx].append(
                    local_stability["stability"])
                all_pen_stabilities["overall"][result_idx].append(
                    local_stability["penalized_stability"])
    if not ensemble_models:
        for select_idx, _ in enumerate(model_selections):
            values_per_metric_score["validation_stability_std"][select_idx] = np.std(values_per_metric_score["validation_stability"][:, :, :, select_idx])
            values_per_metric_score["validation_penalized_stability_std"][select_idx] = np.std(values_per_metric_score["validation_penalized_stability"][:, :, :, select_idx])

            values_per_metric["validation_stability_std"][select_idx] = np.std(all_stabilities["metric"][select_idx])
            values_per_metric["validation_penalized_stability_std"][select_idx] = np.std(all_pen_stabilities["metric"][select_idx])

            values_per_score["validation_stability_std"][select_idx] = np.std(all_stabilities["score"][select_idx])
            values_per_score["validation_penalized_stability_std"][select_idx] = np.std(all_pen_stabilities["score"][select_idx])

            values["validation_stability_std"][select_idx] = np.std(all_stabilities["overall"][select_idx])
            values["validation_penalized_stability_std"][select_idx] = np.std(all_pen_stabilities["overall"][select_idx])

    global_std = False
    all_penalized_stability = {}
    all_stability = {}
    print("Validating penalized stability :")
    penalized_stab = values["penalized_stability"]
    penalized_stab_std = values["penalized_stability_std"]
    validation_pen_stab = values["validation_penalized_stability"]
    stab = values["stability"]
    stab_std = values["stability_std"]
    validation_stab = values["validation_stability"]
    # print(f"Without selection : {penalized_stab.mean(0)[0]} "
    #       f"and validation {validation_pen_stab.mean(0)[0]} with "
    #       f"std {validation_pen_stab.std(0)[0]}")
    # print(f"With model selection thr : {penalized_stab.mean(0)[1]} "
    #       f"and validation {validation_pen_stab.mean(0)[1]} with "
    #       f"std {validation_pen_stab.std(0)[1]}")
    # print(f"With weighted heuristics : {penalized_stab.mean(0)[2]} "
    #       f"and validation {validation_pen_stab.mean(0)[2]} with "
    #       f"std {validation_pen_stab.std(0)[2]})")
    # array_of_pen_stabs = np.concatenate(([penalized_stab[0]], [validation_pen_stab[0]], [validation_pen_stab[1]]), axis=0)
    # array_of_stabs = np.concatenate(([stab[0]], [validation_stab[0]], [validation_stab[1]]), axis=0)
    # array_of_pen_stabs = np.concatenate(([validation_pen_stab[comp_idx]] for comp_idx in range(len(comparisons))), axis=0)
    # array_of_stabs = np.concatenate(([validation_stab[comp_idx]] for comp_idx in range(len(comparisons))), axis=0)
    # print(f"Average stability without selection : {validation_stab.mean(0)[0]} +- {validation_stab.std(0)[0]} (pen : {validation_pen_stab.mean(0)[0]} +- {validation_pen_stab.std(0)[0]})")
    # print(f"Average stability with selection thr : {validation_stab.mean(0)[1]} +- {validation_stab.std(0)[1]} (pen : {validation_pen_stab.mean(0)[1]} +- {validation_pen_stab.std(0)[1]})")
    # print(f"Average stability with weighted heuristics : {validation_stab.mean(0)[2]} +- {validation_stab.std(0)[2]} (pen : {validation_pen_stab.mean(0)[2]} +- {validation_pen_stab.std(0)[2]})")
    print()
    all_penalized_stability["overall"] = {}
    all_penalized_stability["overall"]["mean"] = validation_pen_stab.mean(0)
    all_penalized_stability["overall"]["std"] = validation_pen_stab.std(0)
    if global_std:
        all_penalized_stability["overall"]["std"] = values["validation_penalized_stability_std"]
    all_penalized_stability["overall"]["initial"] = penalized_stab[0]
    all_penalized_stability["overall"]["initial_std"] = penalized_stab_std[0]
    all_stability["overall"] = {}
    all_stability["overall"]["mean"] = validation_stab.mean(0)
    all_stability["overall"]["std"] = validation_stab.std(0)
    if global_std:
        all_stability["overall"]["std"] = values["validation_stability_std"]
    all_stability["overall"]["initial"] = stab[0]
    all_stability["overall"]["initial_std"] = stab_std[0]

    print("Validation penalized stability per metric")
    average_stab = [[] for _ in model_selections]
    deviations = [[] for _ in model_selections]
    average_pen_stab = [[] for _ in model_selections]
    pen_deviations = [[] for _ in model_selections]
    initial_stab = [[] for _ in model_selections]
    initial_stab_std = [[] for _ in model_selections]
    initial_pen_stab = [[] for _ in model_selections]
    initial_pen_stab_std = [[] for _ in model_selections]
    for metric_idx, metric in enumerate(metrics):
        penalized_stab = values_per_metric["penalized_stability"][metric_idx]
        penalized_stab_std = values_per_metric["penalized_stability_std"][metric_idx]
        validation_pen_stab = values_per_metric["validation_penalized_stability"][metric_idx]
        stab = values_per_metric["stability"][metric_idx]
        stab_std = values_per_metric["stability_std"][metric_idx]
        validation_stab = values_per_metric["validation_stability"][metric_idx]
        # print(f"Validating penalized stability for {metric}:")
        # print(f"Without selection : {penalized_stab.mean(0)[0]} and validation"
        #       f" {validation_pen_stab.mean(0)[0]} with std "
        #       f"{validation_pen_stab.std(0)[0]}")
        # print(f"With model selection thr : {penalized_stab.mean(0)[1]} "
        #       f"and validation {validation_pen_stab.mean(0)[1]} with "
        #       f"std {validation_pen_stab.std(0)[1]}")
        # print(f"With weighted heuristics : {penalized_stab.mean(0)[2]} "
        #       f"and validation {validation_pen_stab.mean(0)[2]} with "
        #       f"std {validation_pen_stab.std(0)[2]}")
        # print()
        # array_of_pen_stabs = np.concatenate(([penalized_stab[0]], [validation_pen_stab[0]], [validation_pen_stab[1]]), axis=0)
        # array_of_stabs = np.concatenate(([stab[0]], [validation_stab[0]], [validation_stab[1]]), axis=0)
        # array_of_pen_stabs = np.concatenate(([validation_pen_stab[comp_idx]] for comp_idx in range(len(comparisons))), axis=0)
        # array_of_stabs = np.concatenate(([validation_pen_stab[comp_idx]] for comp_idx in range(len(comparisons))), axis=0)
        for select_idx in range(len(model_selections)):
            if not ensemble_models:
                average_stab[select_idx].append(validation_stab.mean(0)[select_idx])
            # deviations[select_idx].append(validation_stab.std(0)[select_idx])
                average_pen_stab[select_idx].append(validation_pen_stab.mean(0)[select_idx])
                pen_deviations[select_idx].append(validation_pen_stab.std(0)[select_idx])
            initial_stab[select_idx].append(stab[0, select_idx])
            initial_pen_stab[select_idx].append(penalized_stab[0, select_idx])
            initial_stab_std[select_idx].append(stab_std[0, select_idx])
            initial_pen_stab_std[select_idx].append(penalized_stab_std[0, select_idx])
    # print(f"Average stability without selection : {np.mean(average_stab[0])} +- {np.mean(deviations[0])} (pen {np.mean(average_pen_stab[0])} +- {np.mean(pen_deviations[0])}")
    # print(f"Average stability with selection thr : {np.mean(average_stab[1])} +- {np.mean(deviations[1])} (pen {np.mean(average_pen_stab[1])} +- {np.mean(pen_deviations[1])}")
    # print(f"Average stability with weighted heuristics : {np.mean(average_stab[2])} +- {np.mean(deviations[2])} (pen {np.mean(average_pen_stab[2])} +- {np.mean(pen_deviations[2])}")
    # print()
    all_stability["metric"] = {}
    if not ensemble_models:
        all_stability["metric"]["mean"] = np.mean(average_stab, 1)
    # all_stability["metric"]["std"] = np.mean(deviations, 1)
    # all_stability["metric"]["std"] = np.std(average_stab, 1)
        all_stability["metric"]["std"] = values_per_metric["validation_stability"].mean(0).std(0)
        if global_std:
            all_stability["metric"]["std"] = values_per_metric["validation_stability_std"]
    all_stability["metric"]["initial"] = np.mean(initial_stab, 1)
    all_stability["metric"]["initial_std"] = np.mean(initial_stab_std, 1)
    all_penalized_stability["metric"] = {}
    if not ensemble_models:
        all_penalized_stability["metric"]["mean"] = np.mean(average_pen_stab, 1)
    # all_penalized_stability["metric"]["std"] = np.mean(pen_deviations, 1)
    # all_penalized_stability["metric"]["std"] = np.std(average_pen_stab, 1)
        all_penalized_stability["metric"]["std"] = values_per_metric["validation_penalized_stability"].mean(0).std(0)
        if global_std:
            all_penalized_stability["metric"]["std"] = values_per_metric["validation_penalized_stability_std"]
    all_penalized_stability["metric"]["initial"] = np.mean(initial_pen_stab, 1)
    all_penalized_stability["metric"]["initial_std"] = np.mean(initial_pen_stab_std, 1)
    
    # print("Validation penalized stability per score")
    average_stab = [[] for _ in model_selections]
    deviations = [[] for _ in model_selections]
    average_pen_stab = [[] for _ in model_selections]
    pen_deviations = [[] for _ in model_selections]
    initial_stab = [[] for _ in model_selections]
    initial_stab_std = [[] for _ in model_selections]
    initial_pen_stab = [[] for _ in model_selections]
    initial_pen_stab_std = [[] for _ in model_selections]
    for score_idx, score in enumerate(scores):
        penalized_stab = values_per_score["penalized_stability"][score_idx]
        penalized_stab_std = values_per_score["penalized_stability_std"][score_idx]
        validation_pen_stab = values_per_score["validation_penalized_stability"][score_idx]
        stab = values_per_score["stability"][score_idx]
        stab_std = values_per_score["stability_std"][score_idx]
        validation_stab = values_per_score["validation_stability"][score_idx]
        # print(f"Validating penalized stability for {score}:")
        # print(f"Without selection : {penalized_stab.mean(0)[0]} "
        #   f"and validation {validation_pen_stab.mean(0)[0]} with "
        #   f"std {validation_pen_stab.std(0)[0]}")
        # print(f"With model selection thr : {penalized_stab.mean(0)[1]} "
        #     f"and validation {validation_pen_stab.mean(0)[1]} with "
        #     f"std {validation_pen_stab.std(0)[1]}")
        # print(f"With weighted heuristics : {penalized_stab.mean(0)[2]} "
        #     f"and validation {validation_pen_stab.mean(0)[2]} with "
        #     f"std {validation_pen_stab.std(0)[2]}")
        # print()
        # array_of_pen_stabs = np.concatenate(([penalized_stab[0]], [validation_pen_stab[0]], [validation_pen_stab[1]]), axis=0)
        # array_of_stabs = np.concatenate(([stab[0]], [validation_stab[0]], [validation_stab[1]]), axis=0)
        # array_of_pen_stabs = np.concatenate(([validation_pen_stab[0]], [validation_pen_stab[1]]), axis=0)
        # array_of_stabs = np.concatenate(([validation_stab[0]], [validation_stab[1]]), axis=0)
        for select_idx in range(len(model_selections)):
            if not ensemble_models:
                average_stab[select_idx].append(validation_stab.mean(0)[select_idx])
                deviations[select_idx].append(validation_stab.std(0)[select_idx])
                average_pen_stab[select_idx].append(validation_pen_stab.mean(0)[select_idx])
                pen_deviations[select_idx].append(validation_pen_stab.std(0)[select_idx])
            initial_stab[select_idx].append(stab[0][select_idx])
            initial_pen_stab[select_idx].append(penalized_stab[0][select_idx])
            initial_stab_std[select_idx].append(stab_std[0][select_idx])
            initial_pen_stab_std[select_idx].append(penalized_stab_std[0][select_idx])

    # print(f"Average stability without selection : {np.mean(average_stab[0])} +- {np.mean(deviations[0])} (pen {np.mean(average_pen_stab[0])} +- {np.mean(pen_deviations[0])}")
    # print(f"Average stability with selection thr : {np.mean(average_stab[1])} +- {np.mean(deviations[1])} (pen {np.mean(average_pen_stab[1])} +- {np.mean(pen_deviations[1])}")
    # print(f"Average stability with weighted heuristics : {np.mean(average_stab[2])} +- {np.mean(deviations[2])} (pen {np.mean(average_pen_stab[2])} +- {np.mean(pen_deviations[2])}")
    # print()
    all_stability["score"] = {}
    if not ensemble_models:
        all_stability["score"]["mean"] = np.mean(average_stab, 1)
    # all_stability["score"]["std"] = np.mean(deviations, 1)
    # all_stability["score"]["std"] = np.std(average_stab, 1)
        all_stability["score"]["std"] = values_per_score["validation_stability"].mean(0).std(0)
        if global_std:
            all_stability["score"]["std"] = values_per_score["validation_stability_std"]
    all_stability["score"]["initial"] = np.mean(initial_stab, 1)
    all_stability["score"]["initial_std"] = np.mean(initial_stab_std, 1)
    all_penalized_stability["score"] = {}
    if not ensemble_models:
        all_penalized_stability["score"]["mean"] = np.mean(average_pen_stab, 1)
        # all_penalized_stability["score"]["std"] = np.mean(pen_deviations, 1)
        # all_penalized_stability["score"]["std"] = np.std(average_pen_stab, 1)
        all_penalized_stability["score"]["std"] = values_per_score["validation_penalized_stability"].mean(0).std(0)
        if global_std:
            all_penalized_stability["score"]["std"] = values_per_score["validation_penalized_stability_std"]
    all_penalized_stability["score"]["initial"] = np.mean(initial_pen_stab, 1)
    all_penalized_stability["score"]["initial_std"] = np.mean(initial_pen_stab_std, 1)

    # print("Validation penalized stability per metric-score")
    average_stab = [[] for _ in model_selections]
    deviations = [[] for _ in model_selections]
    average_pen_stab = [[] for _ in model_selections]
    pen_deviations = [[] for _ in model_selections]
    initial_stab = [[] for _ in model_selections]
    initial_stab_std = [[] for _ in model_selections]
    initial_pen_stab = [[] for _ in model_selections]
    initial_pen_stab_std = [[] for _ in model_selections]
    for metric_idx, metric in enumerate(metrics):
        for score_idx, score in enumerate(scores):
            penalized_stab = values_per_metric_score["penalized_stability"][metric_idx, score_idx]
            penalized_stab_std = values_per_metric_score["penalized_stability_std"][metric_idx, score_idx]
            validation_pen_stab = values_per_metric_score["validation_penalized_stability"][metric_idx, score_idx]
            stab = values_per_metric_score["stability"][metric_idx, score_idx]
            stab_std = values_per_metric_score["stability_std"][metric_idx, score_idx]
            validation_stab = values_per_metric_score["validation_stability"][metric_idx, score_idx]
            # print(f"Validating penalized stability for {metric} and {score}:")
            # print(f"Without selection : {penalized_stab.mean(0)[0]} "
            #     f"and validation {validation_pen_stab.mean(0)[0]} with "
            #     f"std {validation_pen_stab.std(0)[0]}")
            # print(f"With model selection thr : {penalized_stab.mean(0)[1]} "
            #     f"and validation {validation_pen_stab.mean(0)[1]} with "
            #     f"std {validation_pen_stab.std(0)[1]}")
            # print(f"With weighted heuristics : {penalized_stab.mean(0)[2]} "
            #     f"and validation {validation_pen_stab.mean(0)[2]} with "
            #     f"std {validation_pen_stab.std(0)[2]}")
            # print()
            # array_of_pen_stabs = np.concatenate(([penalized_stab[0]], [validation_pen_stab[0]], [validation_pen_stab[1]]), axis=0)
            # array_of_stabs = np.concatenate(([stab[0]], [validation_stab[0]], [validation_stab[1]]), axis=0)
            # array_of_pen_stabs = np.concatenate(([validation_pen_stab[0]], [validation_pen_stab[1]]), axis=0)
            # array_of_stabs = np.concatenate(([validation_stab[0]], [validation_stab[1]]), axis=0)
            for select_idx in range(len(model_selections)):
                if not ensemble_models:
                    average_stab[select_idx].append(validation_stab.mean(0)[select_idx])
                    deviations[select_idx].append(validation_stab.std(0)[select_idx])
                    average_pen_stab[select_idx].append(validation_pen_stab.mean(0)[select_idx])
                    pen_deviations[select_idx].append(validation_pen_stab.std(0)[select_idx])
                initial_stab[select_idx].append(stab[0][select_idx])
                initial_pen_stab[select_idx].append(penalized_stab[0][select_idx])
                initial_stab_std[select_idx].append(stab_std[0][select_idx])
                initial_pen_stab_std[select_idx].append(penalized_stab_std[0][select_idx])
    # print(f"Average stability without selection : {np.mean(average_stab[0])} +- {np.mean(deviations[0])} (pen {np.mean(average_pen_stab[0])} +- {np.mean(pen_deviations[0])}")
    # print(f"Average stability with selection thr : {np.mean(average_stab[1])} +- {np.mean(deviations[1])} (pen {np.mean(average_pen_stab[1])} +- {np.mean(pen_deviations[1])}")
    # print(f"Average stability with weighted heuristics : {np.mean(average_stab[2])} +- {np.mean(deviations[2])} (pen {np.mean(average_pen_stab[2])} +- {np.mean(pen_deviations[2])}")
    # print()
    all_stability["metric_score"] = {}
    if not ensemble_models:
        all_stability["metric_score"]["mean"] = np.mean(average_stab, 1)
        # all_stability["metric_score"]["std"] = np.mean(deviations, 1)
        # all_stability["metric_score"]["std"] = np.std(average_stab, 1)
        all_stability["metric_score"]["std"] = values_per_metric_score["validation_stability"].mean((0, 1)).std(0)
        if global_std:
            all_stability["metric_score"]["std"] = values_per_metric_score["validation_stability_std"]
    all_stability["metric_score"]["initial"] = np.mean(initial_stab, 1)
    all_stability["metric_score"]["initial_std"] = np.mean(initial_stab_std, 1)
    all_penalized_stability["metric_score"] = {}
    if not ensemble_models:
        all_penalized_stability["metric_score"]["mean"] = np.mean(average_pen_stab, 1)
        # all_penalized_stability["metric_score"]["std"] = np.mean(pen_deviations, 1)
        # all_penalized_stability["metric_score"]["std"] = np.std(average_pen_stab, 1)
        all_penalized_stability["metric_score"]["std"] = values_per_metric_score["validation_penalized_stability"].mean((0, 1)).std(0)
        if global_std:
            all_penalized_stability["metric_score"]["std"] = values_per_metric_score["validation_penalized_stability_std"]
    all_penalized_stability["metric_score"]["initial"] = np.mean(initial_pen_stab, 1)
    all_penalized_stability["metric_score"]["initial_std"] = np.mean(initial_pen_stab_std, 1)

    groups = ["overall", "metric", "score", "metric_score"]
    groups = ["metric_score"]
    x = np.arange(len(groups))  # the label locations
    width = 0.18  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')

    colors = ['#ff7f0e', '#d62728', '#8c564b', '#1f77b4', '#2ca02c', '#9467bd',
              '#e377c2','#7f7f7f', '#bcbd22', '#17becf']
    for selection_idx, selection in enumerate(model_selections):
        offset = width * multiplier
        means = []
        stds = []
        initials = []
        initial_stds = []
        for granularity in groups:
            if not ensemble_models:
                means.append(np.round(np.mean(all_stability[granularity]["mean"][selection_idx]), 3))
                stds.append(np.mean(all_stability[granularity]["std"][selection_idx]))
            initials.append(np.round(np.mean(all_stability[granularity]["initial"][selection_idx]), 3))
            initial_stds.append(np.mean(all_stability[granularity]["initial_std"][selection_idx]))
        if ensemble_models:
            rects = ax.bar(x + offset, initials, width, label=selection,
                           yerr=initial_stds, edgecolor="white",
                           color=colors[selection_idx])
        else:
            rects = ax.bar(x + offset, means, width, label=selection, yerr=stds, edgecolor="white")

        ax.bar_label(rects, padding=3)
        # ax.bar(x + offset, initials, width, label="initial", fill=False)
        if not ensemble_models:
            ymin = x + offset - width / 2
            ymax = x + offset + width / 2
            label = "train" if selection_idx == 0 else None
            ax.hlines(initials, ymin, ymax, linestyles="dashed", colors="k", label=label)
            ax.vlines(x + offset + width / 4, np.array(initials) - np.array(initial_stds),
                    np.array(initials) + np.array(initial_stds),
                    linestyles="dashed", colors="k")
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Stability")
    ax.set_title("Stability per granularity")
    ax.set_xticks(x + 2*width, groups)
    ax.legend(loc="upper left", ncols=3, title="Model selection strategy")
    ax.set_ylim(0, 0.9)

    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')

    for selection_idx, selection in enumerate(model_selections):
        offset = width * multiplier
        means = []
        stds = []
        initials = []
        initial_stds = []
        for granularity in groups:
            if not ensemble_models:
                means.append(np.round(np.mean(all_penalized_stability[granularity]["mean"][selection_idx]), 3))
                stds.append(np.mean(all_penalized_stability[granularity]["std"][selection_idx]))
            initials.append(np.round(np.mean(all_penalized_stability[granularity]["initial"][selection_idx]), 3))
            initial_stds.append(np.mean(all_penalized_stability[granularity]["initial_std"][selection_idx]))
        if ensemble_models:
            rects = ax.bar(x + offset, initials, width, label=selection,
                           yerr=initial_stds, edgecolor="white",
                           color=colors[selection_idx])
        else:
            rects = ax.bar(x + offset, means, width, label=selection, yerr=stds, edgecolor="white")
        ax.bar_label(rects, padding=3)
        # ax.bar(x + offset, initials, width, label="initial", fill=False)#, edgecolor="white")
        if not ensemble_models:
            ymin = x + offset - width / 2
            ymax = x + offset + width / 2
            label = "train" if selection_idx == 0 else None
            ax.hlines(initials, ymin, ymax, linestyles="dashed", colors="k", label=label)
            ax.vlines(x + offset + width / 4, np.array(initials) - np.array(initial_stds),
                    np.array(initials) + np.array(initial_stds),
                    linestyles="dashed", colors="k")
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Penalized stability")
    ax.set_title("Penalized stability per granularity")
    ax.set_xticks(x + 2*width, groups)
    ax.legend(loc="upper left", ncols=3, title="Model selection strategy")
    ax.set_ylim(0, 0.9)

    plt.show()

def evaluate_stability_scaling(dataset, datasetdir, outdir, runs=[],
                               metrics=["thickness", "meancurv", "area"],
                               scores=None, vary_models=True, 
                               select_good_models=0,
                               stability_measure="product", n_subjects=150,
                               sampling=None, sample_latents=None,
                               scaling_params=None):
    assert len(runs) >= 2
    heuristics_params = {
        "pvalues_vote": {"strategy": ["vote_prop-trust_level"], "vote_prop": [0.95, 1], "trust_level": [0.95, 1]},
        # "pvalues_prod": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 30)), "thr": [5e-3, 1e-5, 1e-20, 1e-50, 1e-100], "var": [1, 0.5, 0.25]},
        "pvalues_min": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 25)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "pvalues_coefs": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 25)), "thr": [1e-250, 1e-200, 1e-150, 1e-100, 1e-50, 1e-20], "var": [1, 0.5, 0.25]},
        # "pvalues_combine_fisher": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 25)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "pvalues_combine_pearson": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 25)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "pvalues_combine_tippett": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 25)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "pvalues_combine_stouffer": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 25)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "pvalues_combine_mudholkar_george": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 25)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        "pvalues_mean": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 25)), "thr": [1e-3, 1e-5, 1e-8], "var": [1, 0.5, 0.25]},
        "coefs_mean": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 25)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        "coefs_max": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 25)), "thr": [1e-3, 1e-5, 1e-8], "var": [1, 0.5, 0.25]},
        # "coefs_weighted_mean_score": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 25)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "coefs_weighted_mean_rank": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 25)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        
        #"composite": {"strategy": ["thr", "num", "var"], "num": [10], "thr": [1e-10]}
    }

    # heuristics_params = {
    #     "coefs_mean": {"strategy": ["num-var"], "num": [17], "var": [0.25]},
    # }

    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings("ignore")

    clinical_names = np.load(
            os.path.join(datasetdir, "clinical_names.npy"), allow_pickle=True)
    clinical_names = clinical_names.tolist()
    if scores is None:
        scores = clinical_names
    rois_names = np.load(
        os.path.join(datasetdir, "rois_names.npy"), allow_pickle=True)
    rois = np.array(
        list(dict.fromkeys([name.rsplit("_", 1)[0] for name in rois_names])))
    metadata = pd.read_table(
        os.path.join(datasetdir, "metadata_train.tsv"))
    metadata_columns = metadata.columns.tolist()

    # Computing heuristics with various parameters for each metric / score
    if scaling_params is None:
        scaling_params = list(range(100))
        if not vary_models:
            scaling_params = list(range(20))

    n_params = len(scaling_params)

    if select_good_models > 0:
        assert select_good_models < n_params
        scaling_params = []
        if int(select_good_models) == select_good_models:
            num_remaining = len(model_scores) - select_good_models
            for run in runs:
                scaling_params.append([])
                model_scores = score_models(dataset, datasetdir, outdir, run, scores=scores)
                selected_models = np.argsort(model_scores)[::-1][:num_remaining]
                scaling_params[-1] = np.sort(selected_models)
        else:
            all_scores = []
            num_models_bellow = []
            for run in runs:
                model_scores = score_models(dataset, datasetdir, outdir, run, scores=scores)
                all_scores.append(model_scores)
                num_models_bellow.append((model_scores < select_good_models).sum())
            min_to_remove = min(num_models_bellow)
            num_remaining = len(model_scores) - min_to_remove
            for run_idx, run in enumerate(runs):
                scaling_params.append([])
                model_scores = all_scores[run_idx]
                selected_models = np.argsort(model_scores)[::-1][:num_remaining]
                scaling_params[-1] = np.sort(selected_models)
        n_params = num_remaining

    scaled_global_results = []
    for param_idx in range(1, n_params + 1):
        validation_indices = None
        if select_good_models > 0:
            run0_indices = scaling_params[0][:param_idx]
            run1_indices = scaling_params[1][:param_idx]
            model_indices = np.array([run0_indices, run1_indices]).T
        else:
            model_indices = scaling_params[:param_idx]
            if not vary_models:
                validation_indices = scaling_params[:param_idx]
                model_indices = None
        global_results = compute_all_associations(dataset, datasetdir, outdir, runs,
                                                  heuristics_params,
                                                  metrics, scores, model_indices=model_indices,
                                                  validation_indices=validation_indices,
                                                  n_subjects=n_subjects, sampling=sampling,
                                                  sample_latents=sample_latents)
        scaled_global_results.append(global_results)


    # Computing stability
    ideal_Ns = np.array(list(range(1, 21))) # np.sqrt(len(rois))
    comparisons = list(itertools.combinations(runs, 2))
    best_values = {"stability" : np.empty((n_params, len(ideal_Ns))),
                   "penalized_stability": np.empty((n_params, len(ideal_Ns))),
                   "stability_std" : np.zeros((n_params, len(ideal_Ns))),
                   "penalized_stability_std": np.zeros((n_params, len(ideal_Ns))),
                   "heuristic": np.empty((n_params, len(ideal_Ns)), dtype=object),
                   "strat_param" : np.empty((n_params, len(ideal_Ns)), dtype=object),
                   "daa_params": np.empty((n_params, len(ideal_Ns)), dtype=object)}

    best_values_per_metric_score = {
        "stability" : np.zeros((len(metrics), len(scores), n_params, len(ideal_Ns))),
        "penalized_stability": np.zeros((len(metrics), len(scores), n_params, len(ideal_Ns))),
        "stability_std" : np.zeros((len(metrics), len(scores), n_params, len(ideal_Ns))),
        "penalized_stability_std": np.zeros((len(metrics), len(scores), n_params, len(ideal_Ns))),
        "heuristic": np.empty((len(metrics), len(scores), n_params, len(ideal_Ns)), dtype=object),
        "strat_param" : np.empty((len(metrics), len(scores), n_params, len(ideal_Ns)), dtype=object),
        "daa_params": np.empty((len(metrics), len(scores), n_params, len(ideal_Ns)), dtype=object)
    }

    best_values_per_metric = {
        "stability" : np.zeros((len(metrics), n_params, len(ideal_Ns))),
        "penalized_stability": np.zeros((len(metrics), n_params, len(ideal_Ns))),
        "stability_std" : np.zeros((len(metrics), n_params, len(ideal_Ns))),
        "penalized_stability_std": np.zeros((len(metrics), n_params, len(ideal_Ns))),
        "heuristic": np.empty((len(metrics), n_params, len(ideal_Ns)), dtype=object),
        "strat_param" : np.empty((len(metrics), n_params, len(ideal_Ns)), dtype=object),
        "daa_params": np.empty((len(metrics), n_params, len(ideal_Ns)), dtype=object)
    }

    best_values_per_score = {
        "stability" : np.zeros((len(scores), n_params, len(ideal_Ns))),
        "penalized_stability": np.zeros((len(scores), n_params, len(ideal_Ns))),
        "stability_std" : np.zeros((len(scores), n_params, len(ideal_Ns))),
        "penalized_stability_std": np.zeros((len(scores), n_params, len(ideal_Ns))),
        "heuristic": np.empty((len(scores), n_params, len(ideal_Ns)), dtype=object),
        "strat_param" : np.empty((len(scores), n_params, len(ideal_Ns)), dtype=object),
        "daa_params": np.empty((len(scores), n_params, len(ideal_Ns)), dtype=object)
    }

    variables = list(best_values.keys())
    std_variables = ["stability_std", "penalized_stability_std"]
    (variables.remove(var) for var in std_variables)

    # Compute penalized stability for each ideal_N value
    for param_idx in tqdm(range(n_params)):
        global_results = scaled_global_results[param_idx]
        for N_idx, ideal_N in enumerate(ideal_Ns):
            stability_per_score_metric = {
                "daa_params": [], "heuristic": [], "strat_param": [],
                "metric": [], "score": [], "stability": [],
                "penalized_stability": [], "comparison": []}
            for comparison_idx, comparison in enumerate(comparisons):
                run_0, run_1 = comparison
                run_0_idx = runs.index(run_0)
                run_1_idx = runs.index(run_1)
                local_result = (global_results[run_0_idx], global_results[run_1_idx])
                for daa_params in set(list(local_result[0].keys())).intersection(
                    local_result[1].keys()):
                    for heuristic in heuristics_params.keys():
                        if "strategy" in heuristics_params[heuristic]:
                            for strategy in heuristics_params[heuristic]["strategy"]:
                                if "-" not in strategy:
                                    for strat_param in heuristics_params[heuristic][strategy]:
                                        strat_param_name = f"strategy_{strategy}_value_{strat_param}"

                                        local_stability_per_metric_score = (
                                            compute_all_stability(local_result,
                                                                daa_params,
                                                                heuristic,
                                                                strat_param_name,
                                                                ideal_N, metrics,
                                                                scores, stability_measure))
                                        for key, value in stability_per_score_metric.items():
                                            value += local_stability_per_metric_score[key]

                                else:
                                    first_param, second_param = strategy.split("-")
                                    for first_value, second_value in itertools.product(
                                        heuristics_params[heuristic][first_param],
                                        heuristics_params[heuristic][second_param]):
                                        strat_param_name = f"strategy_{strategy}_values_{first_value}_{second_value}"

                                        local_stability_per_metric_score = (
                                            compute_all_stability(local_result,
                                                                daa_params,
                                                                heuristic,
                                                                strat_param_name,
                                                                ideal_N, metrics,
                                                                scores, stability_measure))
                                        for key, value in stability_per_score_metric.items():
                                            value += local_stability_per_metric_score[key]

            stability_per_score_metric = pd.DataFrame.from_dict(stability_per_score_metric)
            # print(stability_per_score_metric.sort_values("penalized_stability", ascending=False))
            # print(final_stability.sort_values("penalized_stability", ascending=False))

            # Compute best values per (metric, score), metric and score w.r.t.
            # penalized stability
            for metric_idx, metric in enumerate(metrics):
                for score_idx, score in enumerate(scores):
                    idx = ((stability_per_score_metric["metric"] == metric) &
                        (stability_per_score_metric["score"] == score))
                    local_stability = stability_per_score_metric[idx].groupby([
                        "daa_params", "heuristic", "strat_param", "metric", 
                        "score"
                    ], as_index=False).mean()
                    sorted_local_stability = local_stability.sort_values(
                        "penalized_stability", ascending=False)
                    for variable in variables:
                        best_values_per_metric_score[variable][
                            metric_idx, score_idx, param_idx, N_idx] = (
                            sorted_local_stability[variable].to_list()[0])

            final_stability = stability_per_score_metric.groupby(
                ["daa_params", "heuristic", "strat_param"],
                as_index=False).mean()
            sorted_stability = final_stability.sort_values(
                "penalized_stability", ascending=False)
            for variable in variables:
                best_values[variable][param_idx, N_idx] = (
                    sorted_stability[variable].to_list()[0])

            for metric_idx, metric in enumerate(metrics):
                idx = (stability_per_score_metric["metric"] == metric)
                local_stability = stability_per_score_metric[idx].groupby(
                    ["daa_params", "heuristic", "strat_param", "metric"],
                    as_index=False).mean()
                sorted_local_stability = local_stability.sort_values(
                    "penalized_stability", ascending=False)
                for variable in variables:
                    best_values_per_metric[variable][metric_idx,param_idx, N_idx] = (
                        sorted_local_stability[variable].to_list()[0])

            for score_idx, score in enumerate(scores):
                idx = (stability_per_score_metric["score"] == score)
                local_stability = stability_per_score_metric[idx].groupby(
                    ["daa_params", "heuristic", "strat_param", "score"],
                    as_index=False).mean()
                sorted_local_stability = local_stability.sort_values(
                    "penalized_stability", ascending=False)
                for variable in variables:
                    best_values_per_score[variable][score_idx, param_idx, N_idx] = (
                        sorted_local_stability[variable].to_list()[0])

    # Plot stability for each case
    plot_stability = True
    plot_heuristic_hist = False
    if plot_stability:
        max_pen_stab_idx = best_values["penalized_stability"].argmax(axis=1)
        plt.plot(range(1, n_params + 1), best_values["stability"][range(n_params), max_pen_stab_idx], label="raw")
        plt.plot(range(1, n_params + 1), best_values["penalized_stability"][range(n_params), max_pen_stab_idx], label="penalized")
        plt.legend(title="Stability")
        plt.title("Best stability when varying M models")
        
        best_heuristics = best_values["heuristic"][range(n_params), max_pen_stab_idx]
        best_strat_params = best_values["strat_param"][range(n_params), max_pen_stab_idx]

        heuristics = np.unique(best_heuristics)
        strat_params = np.unique(best_strat_params)
        bins = 10
        hist_heuristics_by_bin = []
        hist_strat_params_by_bin = []
        for b in range(bins):
            bin_min_idx = int(b * n_params / bins)
            bin_max_idx = int((b + 1) * n_params / bins)
            local_heuristics = best_heuristics[bin_min_idx:bin_max_idx]
            local_strat_params = best_strat_params[bin_min_idx:bin_max_idx]
            hist_heuristics_by_bin.append([])
            hist_strat_params_by_bin.append([])
            for heuristic in heuristics:
                hist_heuristics_by_bin[b].append((local_heuristics == heuristic).sum())
            for strat_param in strat_params:
                hist_strat_params_by_bin[b].append((local_strat_params == strat_param).sum())
        bin_values = np.array([int((b+1) * n_params / bins) for b in range(bins)])

        fig, ax = plt.subplots(layout='constrained', figsize=(24, 18))
        width = n_params / bins / len(heuristic)
        multiplier = 0
        for h_idx, heuristic in enumerate(heuristics):
            offset = width * multiplier
            values = [hist_heuristics_by_bin[b][h_idx] for b in range(bins)]
            rects = ax.bar(bin_values + offset, values, width, label=heuristic, color=list(colors.TABLEAU_COLORS)[h_idx])
            # ax.bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Num')
        ax.set_title('Heuristic when augmenting M models')
        # ax.set_xticks(x + width, species)
        ax.legend(loc='upper left', ncols=4)
        # ax.set_ylim(0, 250)

        fig, ax = plt.subplots(layout='constrained', figsize=(24, 18))
        width = n_params / bins / len(strat_params)
        multiplier = 0
        for s_idx, strat_param in enumerate(strat_params):
            offset = width * multiplier
            values = [hist_strat_params_by_bin[b][s_idx] for b in range(bins)]
            rects = ax.bar(bin_values + offset, values, width, label=strat_param, color=list(colors.XKCD_COLORS)[s_idx])
            # ax.bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Num')
        ax.set_title('Strat param when augmenting M models')
        # ax.set_xticks(x + width, species)
        ax.legend(loc='upper left', ncols=3)

        plt.figure(figsize=(24, 18))
        plt.plot(range(1, n_params + 1), ideal_Ns[max_pen_stab_idx])
        plt.title("Best N* when varying M models")
        
        for metric_idx, metric in enumerate(metrics):
            fig, ax = plt.subplots(figsize=(12, 9))
            ax.set_title(f"Stability for {metric}")
            handles = []
            for score_idx, score in enumerate(scores):
                local_best_values = best_values_per_metric_score["penalized_stability"][metric_idx, score_idx]
                max_pen_stab_idx = local_best_values.argmax(axis=1)
                handle = ax.plot(range(1, n_params + 1), best_values_per_metric_score["stability"][metric_idx, score_idx, range(n_params), max_pen_stab_idx],
                                label=score, ls="-", c=list(colors.TABLEAU_COLORS)[score_idx])
                ax.plot(range(1, n_params + 1), local_best_values[range(n_params), max_pen_stab_idx],
                        ls="--", c=list(colors.TABLEAU_COLORS)[score_idx])
                handles += handle
            
            # create manual symbols for legend
            line_stab = lines.Line2D([0], [0], label='raw', color='k', ls="-")
            line_pen_stab = lines.Line2D([0], [0], label='penalized', color='k', ls="--")

            local_best_values = best_values_per_metric["penalized_stability"][metric_idx]
            max_pen_stab_idx = local_best_values.argmax(axis=1)
            handle = ax.plot(range(1, n_params + 1), best_values_per_metric["stability"][metric_idx, range(n_params), max_pen_stab_idx],
                            label="Average", ls="-", c="k", lw=3)
            ax.plot(range(1, n_params + 1), local_best_values[range(n_params), max_pen_stab_idx],
                        ls="--", c="k", lw=3)
            handles += handle

            first_legend = ax.legend(handles=handles, loc='lower right', title="Score")
            ax.add_artist(first_legend)
            ax.legend(handles=[line_stab, line_pen_stab], title="Stability", loc="upper right")


        for score_idx, score in enumerate(scores):
            fig, ax = plt.subplots(figsize=(12, 9))
            ax.set_title(f"Stability for {score}")
            handles = []
            for metric_idx, metric in enumerate(metrics):
                local_best_values = best_values_per_metric_score["penalized_stability"][metric_idx, score_idx]
                max_pen_stab_idx = local_best_values.argmax(axis=1)
                handle = ax.plot(range(1, n_params + 1), best_values_per_metric_score["stability"][metric_idx, score_idx, range(n_params), max_pen_stab_idx],
                                label=metric, ls="-", c=list(colors.TABLEAU_COLORS)[metric_idx])
                ax.plot(range(1, n_params + 1), local_best_values[range(n_params), max_pen_stab_idx],
                        ls="--", c=list(colors.TABLEAU_COLORS)[metric_idx])
                handles += handle

            # create manual symbols for legend
            line_stab = lines.Line2D([0], [0], label='raw', color='k', ls="-")
            line_pen_stab = lines.Line2D([0], [0], label='penalized', color='k', ls="--")

            local_best_values = best_values_per_score["penalized_stability"][score_idx]
            max_pen_stab_idx = local_best_values.argmax(axis=1)
            handle = ax.plot(range(1, n_params + 1), best_values_per_score["stability"][score_idx, range(n_params), max_pen_stab_idx],
                            label="Average", ls="-", c="k", lw=3)
            ax.plot(range(1, n_params + 1), local_best_values[range(n_params), max_pen_stab_idx],
                        ls="--", c="k", lw=3)
            handles += handle
            
            first_legend = ax.legend(handles=handles, title="Metric", loc="lower right")
            ax.add_artist(first_legend)
            ax.legend(handles=[line_stab, line_pen_stab], title="Stability", loc="upper right")
    
    if plot_heuristic_hist:
        plt.figure(figsize=(24, 16))
        plt.hist(best_values["strat_param"])
        plt.xticks(rotation = 315)
        plt.subplots_adjust(bottom=0.3)
        plt.title("Histogram of best heuristics and params on average")
        for metric_idx, metric in enumerate(metrics):
            plt.figure(figsize=(24, 16))
            plt.title(f"Histogram of best heuristic and param for {metric} on average")
            plt.hist(best_values_per_metric["strat_param"][metric_idx])
            plt.xticks(rotation = 315)
            plt.subplots_adjust(bottom=0.3)
        for score_idx, score in enumerate(scores):
            plt.figure(figsize=(24, 16))
            plt.title(f"Histogram of best heuristic and param for {score} on average")
            plt.hist(best_values_per_score["strat_param"][score_idx])
            plt.xticks(rotation = 315)
            plt.subplots_adjust(bottom=0.3)
        plt.figure(figsize=(24, 16))
        plt.title(f"Histogram of best heuristic and param for {score} accross metrics and scores")
        plt.hist(best_values_per_metric_score["strat_param"].reshape(
            (-1, best_values_per_metric_score["strat_param"].shape[-1])))
        plt.xticks(rotation = 315)
        plt.subplots_adjust(bottom=0.3)

    # Print best result for each case

    # for metric_idx, metric in enumerate(metrics):
    #     for score_idx, score in enumerate(scores):
    #         local_pen_stab = best_penalized_stability_per_metric_score[metric_idx, score_idx]
    #         local_stab = best_stability_per_metric_score[metric_idx, score_idx]
    #         local_params = best_heuristic_params_per_metric_score[metric_idx, score_idx]
    #         best_pen_N_idx = np.argwhere(local_pen_stab == np.amax(local_pen_stab)).flatten()
    #         best_N_idx = np.argwhere(local_stab == np.amax(local_stab)).flatten()
    #         best_pen_stab = local_pen_stab[best_pen_N_idx]
    #         best_stab = local_stab[best_N_idx]
    #         best_pen_N = ideal_Ns[best_pen_N_idx]
    #         best_N = ideal_Ns[best_N_idx]
    #         best_pen_param = local_params[best_pen_N_idx]
    #         best_params = local_params[best_N_idx]
    #         print(f"Best penalized stability for {metric} and {score} : {best_pen_stab} for N_pen in {best_pen_N} and coef mean with {best_pen_param}.")
    #         print(f"Best stability for {metric} and {score}: {best_stab} for N_pen in {best_N} and coef mean with {best_params}.")
    
    # for metric_idx, metric in enumerate(metrics):
    #     local_values = {}
    #     for variable in variables:
    #         local_values[variable] = best_values_per_metric[variable][metric_idx]

    #     best_pen_N_idx = np.argwhere(local_values["penalized_stability"] == 
    #                                  np.amax(local_values["penalized_stability"])
    #                                 ).flatten()
    #     best_N_idx = np.argwhere(local_values["stability"] ==
    #                              np.amax(local_values["stability"])).flatten()
    #     best_pen_stab = local_values["penalized_stability"][best_pen_N_idx]
    #     best_stab = local_values["stability"][best_N_idx]
    #     best_pen_N = ideal_Ns[best_pen_N_idx]
    #     best_N = ideal_Ns[best_N_idx]
    #     best_pen_param = local_values["strat_param"][best_pen_N_idx]
    #     best_params = local_values["strat_param"][best_N_idx]
    #     best_pen_daa = local_values["daa_params"][best_pen_N_idx]
    #     best_daa = local_values["daa_params"][best_N_idx]
    #     best_pen_heuristic = local_values["heuristic"][best_pen_N_idx]
    #     best_heuristic = local_values["heuristic"][best_N_idx]
    #     print(f"Best average penalized stability for {metric} : {best_pen_stab} for N_pen in {best_pen_N} and {best_pen_heuristic} with {best_pen_param} and daa params {best_pen_daa}.")
    #     print(f"Best average stability for {metric} : {best_stab} for N_pen in {best_N} and {best_heuristic} with {best_params} and daa params {best_daa}.")
    
    # for score_idx, score in enumerate(scores):
    #     local_values = {}
    #     for variable in variables:
    #         local_values[variable] = best_values_per_score[variable][score_idx]

    #     best_pen_N_idx = np.argwhere(local_values["penalized_stability"] == 
    #                                  np.amax(local_values["penalized_stability"])
    #                                 ).flatten()
    #     best_N_idx = np.argwhere(local_values["stability"] ==
    #                              np.amax(local_values["stability"])).flatten()
    #     best_pen_stab = local_values["penalized_stability"][best_pen_N_idx]
    #     best_stab = local_values["stability"][best_N_idx]
    #     best_pen_N = ideal_Ns[best_pen_N_idx]
    #     best_N = ideal_Ns[best_N_idx]
    #     best_pen_param = local_values["strat_param"][best_pen_N_idx]
    #     best_params = local_values["strat_param"][best_N_idx]
    #     best_pen_daa = local_values["daa_params"][best_pen_N_idx]
    #     best_daa = local_values["daa_params"][best_N_idx]
    #     best_pen_heuristic = local_values["heuristic"][best_pen_N_idx]
    #     best_heuristic = local_values["heuristic"][best_N_idx]
    #     print(f"Best average penalized stability for {score} : {best_pen_stab} for N_pen in {best_pen_N} and {best_pen_heuristic} with {best_pen_param} and daa params {best_pen_daa}.")
    #     print(f"Best average stability for {score} : {best_stab} for N_pen in {best_N} and {best_heuristic} with {best_params} and daa params {best_daa}.")


    # best_pen_N_idx = np.argwhere(best_values["penalized_stability"] == 
    #                                 np.amax(best_values["penalized_stability"])
    #                             ).flatten()
    # best_N_idx = np.argwhere(best_values["stability"] ==
    #                             np.amax(best_values["stability"])).flatten()
    # best_pen_stab = best_values["penalized_stability"][best_pen_N_idx]
    # best_stab = best_values["stability"][best_N_idx]
    # best_pen_N = ideal_Ns[best_pen_N_idx]
    # best_N = ideal_Ns[best_N_idx]
    # best_pen_param = best_values["strat_param"][best_pen_N_idx]
    # best_params = best_values["strat_param"][best_N_idx]
    # best_pen_daa = best_values["daa_params"][best_pen_N_idx]
    # best_daa = best_values["daa_params"][best_N_idx]
    # best_pen_heuristic = best_values["heuristic"][best_pen_N_idx]
    # best_heuristic = best_values["heuristic"][best_N_idx]
    # print(f"Best average penalized stability overall : {best_pen_stab} for N_pen in {best_pen_N} and {best_pen_heuristic} with {best_pen_param} and daa params {best_pen_daa}.")
    # print(f"Best average stability overall : {best_stab} for N_pen in {best_N} and {best_heuristic} with {best_params} and daa params {best_daa}.")
    plt.show()


def study_heuristics(dataset, datasetdir, outdir, runs=[],
                     metrics=["thickness", "meancurv", "area"],
                     scores=None):
    # assert len(runs) == 2
    heuristics_params = {
        # "pvalues_vote": {"vote_prop": [0.95, 1], "trust_level": [0.95, 1]},
        # "pvalues_prod": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [5e-3, 1e-5, 1e-20, 1e-50, 1e-100], "var": [1, 0.5, 0.25]},
        # "pvalues_min": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        "pvalues_combine_fisher": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        "pvalues_combine_pearson": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        "pvalues_combine_tippett": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        "pvalues_combine_stouffer": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        "pvalues_combine_mudholkar_george": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "pvalues_mean": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [1e-3, 1e-5, 1e-8], "var": [1, 0.5, 0.25]},
        # "coefs_mean": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "coefs_max": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [1e-3, 1e-5, 1e-8], "var": [1, 0.5, 0.25]},
        #"composite": {"strategy": ["thr", "num", "var"], "num": [10], "thr": [1e-10]}
    }

    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    clinical_names = np.load(
            os.path.join(datasetdir, "clinical_names.npy"), allow_pickle=True)
    clinical_names = clinical_names.tolist()
    if scores is None:
        scores = clinical_names
    rois_names = np.load(
        os.path.join(datasetdir, "rois_names.npy"), allow_pickle=True)
    rois = np.array(
        list(dict.fromkeys([name.rsplit("_", 1)[0] for name in rois_names])))

    daa_params = "likelihood_False"


    # Computing heuristics with various parameters for each metric / score
    values_to_plot = []
    for run in runs:
        run_results = {}
        expdir = os.path.join(outdir, run)
        daadir = os.path.join(expdir, "daa")
        # print_text(f"experimental directory: {expdir}")
        # print_text(f"DAA directory: {daadir}")
        simdirs = [path for path in glob.glob(os.path.join(daadir, "*"))
                if os.path.isdir(path)]
        # print_text(f"Simulation directories: {','.join(simdirs)}")

        for dirname in simdirs:
            # print_text(dirname)
            if not os.path.exists(os.path.join(dirname, "coefs.npy")):
                continue
            coefs = np.load(os.path.join(dirname, "coefs.npy"))
            pvalues = np.load(os.path.join(dirname, "pvalues.npy"))


            sampling = dirname.split("sampling_")[1].split("_sample")[0]
            sample_latent = dirname.split("latents_")[1].split("_seed")[0]
            n_subjects = int(dirname.split("subjects_")[1].split("_M")[0])
            if f"{sampling}_{sample_latent}" != daa_params or n_subjects != 301:
                continue

            # Aggregation
            # combine_pvalues_heuristics = [heuristic for heuristic in heuristics
            #                               if "pvalues_combine" in heuristic]
            # print("Origin pvalues stats")
            # print(np.sort(pvalues.flatten()).tolist()[:10])
            # print(np.sort(pvalues.flatten())[0])
            # print(np.isnan(pvalues).sum())
            # print(pvalues.flatten().mean())
            # print(pvalues.flatten().std())
            # print(pvalues.flatten().max())
            # print(np.median(pvalues.flatten()))
            # plt.figure()
            # plt.hist(pvalues.flatten())
            # method = heuristic.split("combine_")[-1]
            # combined_pvalues = non_nullity_coef(coefs)
            # for score_idx, score in enumerate(scores):
            #     for metric_idx, metric in enumerate(rois_names):
            #         value = combined_pvalues[score_idx, metric_idx]
            #         if value == 0:
            #             print(score, metric)
            # x = combined_pvalues.flatten()
            x = pvalues.flatten()
                # print(x.tolist()[:10])
                # print(np.sort(x).tolist()[:10])
            print("Combine pvalues stats")
            print(np.sort(x)[0])
            print(x.shape)
            print((x == 0).sum())
            print(np.isnan(x).sum())
            print(x.mean())
            print(np.median(x))
            hist, bins = np.histogram(np.log10(x[x!=0]), bins=20)
            # print(bins[0], bins[-1])

            # # histogram on log scale. 
            # # Use non-equal bin sizes, such that they look equal on log scale.
            # logbins = np.logspace(np.log10(x[x != 0].min()),np.log10(bins[-1]),len(bins))
            # plt.subplot(212)
            # plt.figure()
            values_to_plot.append((run, x[x != 0]))
            # plt.hist(x[x != 0], bins=np.power(10, bins), label=f"{run}", stacked=True, edgecolor="white")#logbins)
            # plt.xscale('log')
            # plt.title(f"Combine coefs pvalues histogram for run {run} with {daa_params}")
            # plt.title("Histogram of pvalues")
            # plt.legend(title="Run")
            # plt.xlabel("log10 pvalues")
            # for heuristic in combine_pvalues_heuristics:
            #     method = heuristic.split("combine_")[-1]
            #     x = combine_all_pvalues(pvalues.astype("double"), method).flatten()
            #     # print(x.tolist()[:10])
            #     # print(np.sort(x).tolist()[:10])
            #     print("Combine pvalues stats")
            #     print(np.sort(x)[0])
            #     print(x.shape)
            #     print((x == 0).sum())
            #     print(np.isnan(x).sum())
            #     print(x.mean())
            #     print(np.median(x))
            #     hist, bins = np.histogram(np.log10(x), bins=10)

                # # histogram on log scale. 
                # # Use non-equal bin sizes, such that they look equal on log scale.
                # logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
                # # plt.subplot(212)
                # plt.figure()
                # plt.hist(x, bins=logbins)
                # plt.xscale('log')
                # plt.title(f"Combine {method} pvalues histogram for run {run} with {daa_params}")
                # plt.xlabel("log10 pvalues")                
                # plt.hist(np.log10(other_agg_pvalues[method].flatten()), bins=20)

    all_values = np.concatenate([value for _, value in values_to_plot])
    hist, bins = np.histogram(np.log10(all_values), bins=20)
    # print(bins[0], bins[-1])
    # # histogram on log scale. 
    # # Use non-equal bin sizes, such that they look equal on log scale.
    # logbins = np.logspace(np.log10(x[x != 0].min()),np.log10(bins[-1]),len(bins))
    # plt.subplot(212)
    # plt.figure()
    values = [value for _, value in values_to_plot]
    labels = [key for key, _ in values_to_plot]
    plt.hist(values, bins=np.power(10, bins), label=labels, stacked=True, edgecolor="white")#logbins)
    plt.xscale('log')
    # plt.title(f"Combine coefs pvalues histogram for run {run} with {daa_params}")
    plt.title("Histogram of $p$-values")
    plt.legend(title="Run")
    plt.xlabel("log10 pvalues")
    plt.show()


    coefs_mean = coefs.mean((0, 1))
    pvalues_mean = pvalues.mean((0, 1))
    coefs_std = coefs.std((0, 1))
    pvalues_std = pvalues.std((0, 1))
    df = {"metric": [], "score": [], "coefs_mean": [], "coefs_std": [], "pvalues_mean": [], "pvalues_std": []}
    for roi_idx, roi_name in enumerate(rois_names):
        for score_idx, score in enumerate(scores):
            metric = roi_name.split("_")[-1]
            df["metric"].append(metric)
            df["score"].append(score)
            df["coefs_mean"].append(np.absolute(coefs_mean[score_idx, roi_idx]))
            df["pvalues_mean"].append(pvalues_mean[score_idx, roi_idx])
            df["coefs_std"].append(coefs_std[score_idx, roi_idx])
            df["pvalues_std"].append(pvalues_std[score_idx, roi_idx])
    
    textfont = dict(
        size=20,
        family="Droid Serif")
    fig = go.Figure()

    fig.add_trace(go.Violin(x=["<i>p</i>-values"] * len(df["pvalues_mean"]),
                            y=df["pvalues_mean"],
                            # legendgroup="Pvalues", scalegroup="Pvalues",
                             name="<i>p</i>-values", line_color="blue")
                )
    fig.add_trace(go.Violin(x=["coefficients"] * len(df["coefs_mean"]),
                            y=df["coefs_mean"],
                            #legendgroup="Coefficients", scalegroup="Coefficients",
                            name="coefficients", line_color="orange")
                )

    fig.update_traces(meanline_visible=True, box_visible=True, showlegend=False)
    fig.update_layout(font=textfont)
    fig.update_yaxes(range=[-0.01, 0.1])
    fig.show()

    fig = go.Figure()

    fig.add_trace(go.Violin(x=["<i>p</i>-values"] * len(df["pvalues_std"]),
                            y=df["pvalues_std"],
                            # legendgroup="Pvalues", scalegroup="Pvalues",
                             name="<i>p</i>-values", line_color="blue")
                )
    fig.add_trace(go.Violin(x=["coefficients"] * len(df["coefs_std"]),
                            y=df["coefs_std"],
                            #legendgroup="Coefficients", scalegroup="Coefficients",
                            name="coefficients", line_color="orange")
                )

    fig.update_traces(meanline_visible=True, showlegend=False, box_visible=True)
    fig.update_layout(font=textfont)
    # fig.update_layout(violinmode="group")
    # fig.update_yaxes(type="log")
    filename = os.path.join(
                dirname, f"pvaluesvscoefsstds.png")
    fig.write_image(filename, scale=5)
    fig.show()


def compute_stable_associations(dataset, datasetdir, outdir, best_heuristic_prior, 
                                runs, additional_data,
                                granularity="overall",
                                model_selection="no_selection",
                                n_subjects=301,
                                sampling="likelihood", sample_latents=False,
                                permuted=False, ensemble_models=False,
                                fast=False):
    """ Display specified score histogram across different cohorts.
    Parameters
    ----------
    dataset: str
        the dataset name: euaims or hbn.
    datasetdir: str
        the path to the dataset associated data.
    outdir: str
        the destination folder.
    run: str
        the name of the experiment in the destination folder:
        `<dataset>_<timestamp>'.
    """

    assert model_selection in best_heuristic_prior.keys()
    assert granularity in best_heuristic_prior[model_selection].keys()

    heuristic_param = {}
    if "overall" not in granularity:
        heuristic_param[granularity] = {}
        for key, value in best_heuristic_prior[model_selection][granularity].items():
            _, _, _, _, _, heuristic_name, strat_param = value
            strategy = strat_param.split("strategy_")[-1].split("_value")[0]
            params = {"strategy": strategy}
            if "-" not in strategy:
                _, param_value = strat_param.rsplit("_", 1)
                params[strategy] = float(param_value) if strategy != "num" else int(param_value)
            else:
                first_param, second_param = strategy.split("-")
                _, first_value, second_value = strat_param.rsplit("_", 2)
                params[first_param] = int(first_value) if first_param == "num" else float(first_value)
                params[second_param] = float(second_value)
            heuristic_param[granularity][key] = {heuristic_name: params}
    else:
        value = best_heuristic_prior[model_selection][granularity]
        _, _, _, _, _, heuristic_name, strat_param = value
        strategy = strat_param.split("strategy_")[-1].split("_value")[0]
        params = {"strategy": strategy}
        if "-" not in strategy:
            _, param_value = strat_param.rsplit("_", 1)
            params[strategy] = float(param_value) if strategy != "num" else int(param_value)
        else:
            first_param, second_param = strategy.split("-")
            _, first_value, second_value = strat_param.rsplit("_", 2)
            params[first_param] = int(first_value) if first_param == "num" else float(first_value)
            params[second_param] = float(second_value)
        heuristic_param[heuristic_name] = params
        heuristic_param = {granularity: heuristic_param}
    # print(heuristic_param)
    heuristic = Heuristic(heuristic_param, additional_data, fast=fast)

    scores = additional_data.clinical_names
    model_thr = None
    if model_selection.startswith("thr"):
        model_thr = float(model_selection.rsplit("_", 1)[-1])
    associations = []
    all_agg_values = []
    all_coefs = []
    all_pvalues = []
    all_model_scores = []
    for run in runs:
        model_scores = score_models(dataset, datasetdir, outdir, run, scores=scores)

        expdir = os.path.join(outdir, run)
        daadir = os.path.join(expdir, "daa")
        simdirs = [path for path in glob.glob(os.path.join(daadir, "*"))
                if os.path.isdir(path)]
        for dirname in simdirs:
            local_sampling = dirname.split("sampling_")[1].split("_sample")[0]
            local_sample_latents = dirname.split("latents_")[1].split("_seed")[0]
            local_n_subjects = int(dirname.split("subjects_")[1].split("_M")[0])
            if not os.path.exists(os.path.join(dirname, "coefs.npy")) or (
                local_sampling != sampling or 
                local_sample_latents != str(sample_latents) or
                local_n_subjects != n_subjects or
                permuted != ("permuted" in dirname)):
                continue
            # print_subtitle(f"Computing stable associations for {dirname}")
            coefs = np.load(os.path.join(dirname, "coefs.npy"))
            pvalues = np.load(os.path.join(dirname, "pvalues.npy"))

            all_coefs.append(coefs)
            all_pvalues.append(pvalues)
            all_model_scores.append(model_scores)

    all_coefs = np.array(all_coefs)
    all_pvalues = np.array(all_pvalues)
    all_model_scores = np.array(all_model_scores)
    ensemble_across = range(len(runs))
    if ensemble_models:
        ensemble_across = range(len(all_coefs[0]))
    for ens_idx in ensemble_across:
        if ensemble_models:
            coefs = all_coefs[:, ens_idx]
            pvalues = all_pvalues[:, ens_idx]
            model_scores = all_model_scores[:, ens_idx]
            if model_thr is not None:
                good_model_idx = model_scores >= model_thr
                coefs = coefs[good_model_idx]
                pvalues = pvalues[good_model_idx]
                model_scores = model_scores[good_model_idx]
        else:
            coefs = all_coefs[ens_idx]
            pvalues = all_pvalues[ens_idx]
            model_scores = all_model_scores[ens_idx]
        df, agg_values = heuristic(coefs, pvalues, model_scores,
                                    return_agg=True)
        associations.append(df)
        # all_agg_values.append(agg_values)
        all_agg_values.append(agg_values)

    return associations, all_agg_values


def select_stable_associations(associations, agg_values, min_occurence, additional_data):
    print_text(f"Number of associations : {len(associations)}")
    rois_names = additional_data.rois_names.tolist()
    clinical_names = additional_data.clinical_names
    df = associations[0]
    counts = []
    all_associations = []
    for idx in range(1, len(associations)):
        local_associations = list(associations[idx].itertuples(index=False))
        all_associations.append(local_associations)
        counts += local_associations
        # df = df.merge(associations[idx], how="outer")
    counter = collections.Counter(counts)
    kept_records = [item[0] for item in counter.most_common(len(counts)) if item[1] >= min_occurence]
    agg_values = np.array(agg_values)
    coefs = agg_values.mean(0)
    # for record in kept_records:
    #     roi_name = f"{record.roi}_{record.metric}"
    #     roi_idx = rois_names.index(roi_name)
    #     score_idx = clinical_names.index(record.score)
    #     sample_idx_with_record = [idx for idx, records in enumerate(all_associations) if record in records]
    #     coefs[score_idx, roi_idx] = agg_values[sample_idx_with_record, score_idx, roi_idx].mean(0)

    return pd.DataFrame(kept_records), coefs

def compute_associations_probs(associations):
    return associations.sum(0) / len(associations)


def permuted_daa_exp(dataset, datasetdir, outdir, run, sampling="likelihood",
                     n_validation=1, n_samples=200, n_subjects=301,
                     M=1000, seed=1037, reg_method="hierarchical",
                     sample_latents=False):
    """ Perform the digital avatars analysis using clinical scores taverses
    to influence the imaging part.
    Parameters
    ----------
    dataset: str
        the dataset name: euaims or hbn.
    datasetdir: str
        the path to the dataset associated data.
    outdir: str
        the destination folder.
    run: str
        the name of the experiment in the destination folder:
        `<dataset>_<timestamp>'.
    sampling_strategy: str, default likelihood
        way to sample realistic value for the variable to explain. Must be
        either "linear", "uniform", "gaussian" or "likelihood".
    n_validation: int, default 5
        the number of times we repeat the experiments.
    n_samples: int, default 200
        the number of samples per subject.
    n_subjects: int, default 50
        the number of subjects used in each validation step.
    M: int, default 1000
        estimate the distribution per clinical scores from M reconstructions
    trust_level: float, default 0.75
        after thresholding the Bonferoni-corrected p-values at 0.05, apply
        a voting threshold at `trust_level * n_validation`.
    seed: int, default 1037
        optionally specify a seed to control expriment reproducibility, set
        to None for randomization.
    """
    if sampling not in ["linear", "uniform", "gaussian", "likelihood"]:
        raise ValueError("sampling_strategy must be either linear, uniform"
                         "gaussian or likelihood")

    print_title(f"DIGITAL AVATARS ANALYSIS: {dataset}")
    expdir = os.path.join(outdir, run)
    daadir = os.path.join(expdir, "daa")
    if not os.path.isdir(daadir):
        os.mkdir(daadir)
    print_text(f"experimental directory: {expdir}")
    print_text(f"DAA directory: {daadir}")

    print_subtitle("Loading data...")
    flags_file = os.path.join(expdir, "flags.rar")
    if not os.path.isfile(flags_file):
        raise ValueError("You need first to train the model.")    
    checkpoints_dir = os.path.join(expdir, "checkpoints")
    experiment, flags = MultimodalExperiment.get_experiment(
        flags_file, checkpoints_dir, datasetdir=datasetdir,
        outdir=outdir)
    n_models = experiment.flags.num_models
    # print_flags(flags)

    clinical_names = np.load(
        os.path.join(datasetdir, "clinical_names.npy"), allow_pickle=True)
    rois_names = np.load(
        os.path.join(datasetdir, "rois_names.npy"), allow_pickle=True)
    metadata = pd.read_table(
        os.path.join(datasetdir, "metadata_train.tsv"))
    metadata_columns = metadata.columns.tolist()
    modalities = ["clinical", "rois"]
    print_text(f"modalities: {modalities}")

    n_scores = len(clinical_names)
    n_rois = len(rois_names)

    additional_data = SimpleNamespace(metadata_columns=metadata_columns,
                                      clinical_names=clinical_names,
                                      rois_names=rois_names)

    # Creating folders and path to content
    params = SimpleNamespace(
        n_validation=n_validation, n_subjects=n_subjects, M=M,
        n_samples=n_samples, reg_method=reg_method,
        sampling=sampling, sample_latents=sample_latents, seed=seed)
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    name = "_".join(["_".join([key, str(val)])
                     for key, val in params.__dict__.items()])
    name += "_permuted"
    resdir = os.path.join(daadir, name)
    if not os.path.isdir(resdir):
        os.mkdir(resdir)

    da_file = os.path.join(resdir, "rois_digital_avatars.npy")
    pvals_file = os.path.join(resdir, "pvalues.npy")

    if not os.path.exists(da_file):
        make_digital_avatars(outdir, run, params, additional_data, permuted=True)
    if not os.path.exists(pvals_file):
        compute_daa_statistics(outdir, run, params, additional_data, permuted=True)


def check_permutation_stable_associations(dataset, datasetdir, outdir, runs,
                                        validation_runs, granularity="overall",
                                        model_selection="no_selection",
                                        metrics=["thickness", "meancurv", "area"],
                                        scores=None, min_occurence=5, n_subjects=301,
                                        sampling="likelihood", sample_latents=False):
    """ Display specified score and metric associations
    Parameters
    ----------
    dataset: str
        the dataset name: euaims or hbn.
    datasetdir: str
        the path to the dataset associated data.
    outdir: str
        the destination folder.
    run: str
        the name of the experiment in the destination folder:
        `<dataset>_<timestamp>'.
    """
    from plotting import plot_areas, plot_coefs
    import matplotlib.pyplot as plt

    print_title(f"PLOT DAA results: {dataset}")

    clinical_names = np.load(
        os.path.join(datasetdir, "clinical_names.npy"), allow_pickle=True)
    clinical_names = clinical_names.tolist()
    if scores is None:
        scores = clinical_names.copy()
    rois_names = np.load(
        os.path.join(datasetdir, "rois_names.npy"), allow_pickle=True)
    metadata = pd.read_table(
        os.path.join(datasetdir, "metadata_train.tsv"))
    metadata_columns = metadata.columns.tolist()

    additional_data = SimpleNamespace(metadata_columns=metadata_columns,
                                      clinical_names=clinical_names,
                                      rois_names=rois_names, scores=scores,
                                      metrics=metrics)

    rois_names = rois_names.tolist()

    stabdir = os.path.join(outdir, runs[0], f"stability_with_{'-'.join(runs[1:])}")
    local_stabdir = os.path.join(
        stabdir,
        (f"n_subjects_{n_subjects}_sampling_{sampling}_sample_latents_"
            f"{sample_latents}"))
    if not os.path.exists(local_stabdir):
        raise ValueError("You must compute the stability between runs before "
                         "ploting final results with validation runs.")
    with open(os.path.join(local_stabdir, "best_heuristic_prior"), 'rb') as f:
        best_heuristic_prior = pickle.load(f)

    df, coefs = compute_stable_associations(dataset, datasetdir, outdir,
        best_heuristic_prior, validation_runs, additional_data,
        granularity, model_selection, min_occurence, n_subjects,
        sampling, sample_latents, permuted=True)
    print(df.shape[0])
    print(df[(df["metric"] == "thickness") & (df["score"] == "SRS_Total")])

    