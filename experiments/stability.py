import os
import glob
import pickle
import itertools
import collections
import torch
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, lines
from joblib import Parallel, delayed
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from multimodal_cohort.experiment import MultimodalExperiment
from multimodal_cohort.constants import short_clinical_names
from multimodal_cohort.utils import extract_and_order_by
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from scipy.stats import pearsonr, combine_pvalues
from color_utils import (print_title, print_subtitle, print_text, print_result)
from daa_functions import (compute_significativity, compute_all_stability,
                           compute_all_associations, Heuristic, make_digital_avatars,
                           compute_daa_statistics)
from workflow import score_models


def evaluate_stability(dataset, datasetdir, outdir, runs=[],
                       metrics=["thickness", "meancurv", "area"],
                       scores=None, select_good_models=None,
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
        "coefs_weighted_mean_score": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        "coefs_weighted_mean_rank": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
    }
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
        "coefs_mean": {"strategy": ["num"], "num": list(range(1, 51, 3))},
        # "coefs_max": {"strategy": ["num"], "num": list(range(1, 31, 2))},
        # "coefs_weighted_mean_score": {"strategy": ["num"], "num": list(range(1, 31, 2))},
        # "coefs_weighted_mean_rank": {"strategy": ["num"], "num": list(range(1, 31, 2))},
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
                list(set([name.rsplit("_", 1)[0] for name in rois_names])))
    metadata = pd.read_table(
        os.path.join(datasetdir, "metadata_train.tsv"))
    metadata_columns = metadata.columns.tolist()

    additional_data = SimpleNamespace(metadata_columns=metadata_columns,
                                    clinical_names=clinical_names,
                                    rois_names=rois_names)

    model_indices = None
    # Computing heuristics with various parameters for each metric / score
    if ensemble_models:
        global_results = []
        number_of_models = range(1, len(runs) + 2, 2)
        for n_models in number_of_models:
            global_result = compute_all_associations(dataset, datasetdir, outdir, runs[:n_models],
                                                    heuristics_params, metrics,
                                                    scores, model_indices,
                                                    n_subjects=n_subjects,
                                                    sampling=sampling,
                                                    sample_latents=sample_latents,
                                                    ensemble_models=ensemble_models)
            global_results.append(global_result)
    else:
        global_result = compute_all_associations(dataset, datasetdir, outdir, runs,
                                                    heuristics_params, metrics,
                                                    scores, model_indices,
                                                    n_subjects=n_subjects,
                                                    sampling=sampling,
                                                    sample_latents=sample_latents,
                                                    ensemble_models=ensemble_models)
        global_results = [global_result]
    if select_good_models is not None:
        if type(select_good_models) not in (list, tuple):
            select_good_models = [select_good_models]
        if int(select_good_models[0]) == select_good_models[0]:
            # n_worsts = range(1, select_good_models + 1)
            for n_worst in select_good_models:
                model_indices = []
                for run in runs:
                    model_scores = score_models(dataset, datasetdir, outdir, run, scores=scores)
                    model_indices.append(list(range(len(model_scores))))
                    worst_models = np.argsort(model_scores)[:n_worst]
                    for model_idx in worst_models:
                        model_indices[-1].remove(model_idx)
                model_indices = np.array(model_indices).T
                global_result_select = compute_all_associations(dataset, datasetdir, outdir,
                                                                runs, heuristics_params,
                                                                metrics, scores, model_indices,
                                                                n_subjects=n_subjects,
                                                                sampling=sampling,
                                                                sample_latents=sample_latents,
                                                                ensemble_models=ensemble_models)
                global_results.append(global_result_select)
        else:
            for threshold in select_good_models:
                model_indices = []
                for run in runs:
                    model_scores = score_models(dataset, datasetdir, outdir, run, scores=scores)
                    model_indices.append(list(range(len(model_scores))))
                    model_indices[-1] = np.array(model_indices[-1])[model_scores >= threshold]
                    print(f"Number of removed models for run {run} :"
                          f"{len(model_scores) - len(model_indices[-1])} "
                          f"with threshold {threshold}")
                # model_indices = np.array(model_indices).T
                # print(model_indices.shape)
                global_result_select = compute_all_associations(dataset, datasetdir, outdir,
                                                                runs, heuristics_params,
                                                                metrics, scores, model_indices,
                                                                n_subjects=n_subjects,
                                                                sampling=sampling,
                                                                sample_latents=sample_latents,
                                                                ensemble_models=ensemble_models)
                global_results.append(global_result_select)

        heuristics_params.update({
            "coefs_weighted_mean_score": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
            "coefs_weighted_mean_rank": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
            # "coefs_weighted_mean_score_softmax": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
            # "coefs_weighted_mean_rank_softmax": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
            # "coefs_weighted_mean_score_log": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
            })
        global_result_new = compute_all_associations(
            dataset, datasetdir, outdir, runs, heuristics_params, metrics,
            scores, None, n_subjects=n_subjects, sampling=sampling,
            sample_latents=sample_latents, ensemble_models=ensemble_models)
        global_results.append(global_result_new)

    print("Associations computed. Computing stability...")
    # Computing stability
    ideal_Ns = np.array(list(range(5, 21)))#np.sqrt(len(rois))
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
    # Compute penalized stability for each ideal_N value
    for result_idx, result in enumerate(global_results):
        for N_idx, ideal_N in enumerate(tqdm(ideal_Ns)):
            stability_per_score_metric = {
                "daa_params": [], "heuristic": [], "strat_param": [], "metric": [],
                "score": [], "stability": [], "penalized_stability": [], "comparison": []}
            product_of_params = []
            for comparison_idx, comparison in enumerate(comparisons):
                run_0, run_1 = comparison
                run_0_idx = to_compare.index(run_0)
                run_1_idx = to_compare.index(run_1)
                local_result = (result[run_0_idx], result[run_1_idx])
                for daa_params in set(list(local_result[0].keys())).intersection(local_result[1].keys()):
                    for heuristic in heuristics_params.keys():
                        if not heuristic in local_result[0][daa_params]:
                            continue
                        for strategy in heuristics_params[heuristic]["strategy"]:
                            if "-" not in strategy:
                                for strat_param in heuristics_params[heuristic][strategy]:
                                    strat_param_name = f"strategy_{strategy}_value_{strat_param}"

                                    # product_of_params.append((
                                    #     (run_0_idx, run_1_idx), daa_params,
                                    #     heuristic, strat_param_name))
                                    local_stability_per_metric_score = (
                                        compute_all_stability(local_result,
                                                            daa_params,
                                                            heuristic,
                                                            strat_param_name,
                                                            ideal_N, metrics,
                                                            scores,
                                                            stability_measure))
                                    local_stability_per_metric_score["comparison"] = [comparison for _ in range(len(metrics) * len(scores))]
                                    for key, value in stability_per_score_metric.items():
                                        value += local_stability_per_metric_score[key]
                            else:
                                first_param, second_param = strategy.split("-")
                                for first_value, second_value in itertools.product(
                                    heuristics_params[heuristic][first_param],
                                    heuristics_params[heuristic][second_param]):
                                    strat_param_name = f"strategy_{strategy}_values_{first_value}_{second_value}"
                                    # product_of_params.append(
                                    #     ((run_0_idx, run_1_idx), daa_params,
                                    #     heuristic, strat_param_name))
                                    local_stability_per_metric_score = (
                                        compute_all_stability(local_result,
                                                            daa_params,
                                                            heuristic,
                                                            strat_param_name,
                                                            ideal_N, metrics,
                                                            scores,
                                                            stability_measure))
                                    local_stability_per_metric_score["comparison"] = [comparison for _ in range(len(metrics) * len(scores))]
                                    for key, value in stability_per_score_metric.items():
                                        value += local_stability_per_metric_score[key]
            # delayed_results = Parallel(n_jobs=-2, verbose=1)(
            #     delayed(compute_all_stability)(
            #         (result[res_idx[0]], result[res_idx[1]]),
            #         daa_params, heuristic, strat_param_name,
            #         ideal_N, metrics, scores, stability_measure)
            #         for res_idx, daa_params, heuristic, strat_param_name in
            #         product_of_params)

            # for params_idx, all_params in enumerate(product_of_params):
            #     local_stability_per_metric_score = delayed_results[params_idx]
            #     local_stability_per_metric_score["comparison"] = [comparison for _ in range(len(metrics) * len(scores))]
            #     for key, value in stability_per_score_metric.items():
            #         value += local_stability_per_metric_score[key]
            
            stability_per_score_metric = pd.DataFrame.from_dict(stability_per_score_metric)
            # print(stability_per_score_metric.sort_values("penalized_stability", ascending=False))
            # print(final_stability.sort_values("penalized_stability", ascending=False))

            # Compute best values per (metric, score), metric and score w.r.t.
            # penalized stability
            best_stability_per_comparison = []
            best_pen_stability_per_comparison = []
            for metric_idx, metric in enumerate(metrics):
                for score_idx, score in enumerate(scores):
                    idx = ((stability_per_score_metric["metric"] == metric) &
                        (stability_per_score_metric["score"] == score))
                    local_stability_mean = stability_per_score_metric[idx].groupby([
                        "daa_params", "heuristic", "strat_param", "metric", 
                        "score"
                    ], as_index=False).mean()
                    sorted_local_stability = local_stability_mean.sort_values(
                        "penalized_stability", ascending=False)
                    for variable in variables:
                        best_values_per_metric_score[variable][
                            metric_idx, score_idx, result_idx, N_idx] = (
                            sorted_local_stability[variable].to_list()[0])
                    if global_std:
                        local_stability_std = stability_per_score_metric[idx].groupby([
                            "daa_params", "heuristic", "strat_param", "metric", 
                            "score"
                        ], as_index=False).std()
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

    # Plot stability for each case
    plot_stability = True
    plot_heuristic_hist = False
    model_selection_idx = -1 if ensemble_models else 0 if select_good_models is None else 1
    std_scaling = 1
    if plot_stability:
        fig, ax = plt.subplots(figsize=(12, 9))
        if ensemble_models:
            handles = []
            for model_idx, n_models in enumerate(number_of_models):
                label = f"with {min(n_models, len(runs))} models"
                color = list(colors.XKCD_COLORS)[model_idx]
            
                handle = ax.plot(ideal_Ns, best_values["stability"][model_idx], label=label, c=color)
                ax.fill_between(ideal_Ns, best_values["stability"][model_idx] - std_scaling * best_values["stability_std"][model_idx],
                                best_values["stability"][model_idx] + std_scaling * best_values["stability_std"][model_idx],
                                color=color, alpha=.1)
                ax.plot(ideal_Ns, best_values["penalized_stability"][model_idx], c=color, ls="--")
                handles += handle
        else:
            handles = []
            handle = ax.plot(ideal_Ns, best_values["stability"][0], label="all models", c="blue")
            ax.fill_between(ideal_Ns, best_values["stability"][0] - std_scaling * best_values["stability_std"][0],
                            best_values["stability"][0] + std_scaling * best_values["stability_std"][0],
                            color="blue", alpha=.1)
            ax.plot(ideal_Ns, best_values["penalized_stability"][0], c="blue", ls="--")
            handles += handle
        if select_good_models is not None:
            for selector_idx, selector in enumerate(select_good_models):
                label = f"without {selector} worst models"
                if int(selector) != selector:
                    label = f"without models with score bellow {selector}"
                color = list(colors.XKCD_COLORS)[selector_idx]
                handle = ax.plot(ideal_Ns, best_values["stability"][1 + selector_idx], label=label, c=color)
                ax.fill_between(ideal_Ns, best_values["stability"][1 + selector_idx] - std_scaling * best_values["stability_std"][1 + selector_idx],
                                best_values["stability"][1 + selector_idx] + std_scaling * best_values["stability_std"][1 + selector_idx],
                                color=color, alpha=.1)
                ax.plot(ideal_Ns, best_values["penalized_stability"][1 + selector_idx], c=color, ls="--")
                handles += handle
            handle = ax.plot(ideal_Ns, best_values["stability"][-1], label="with heuristics using score", c="m")
            ax.plot(ideal_Ns, best_values["penalized_stability"][-1], c="m", ls="--")
            ax.fill_between(ideal_Ns, best_values["stability"][-1] - std_scaling * best_values["stability_std"][-1],
                        best_values["stability"][-1] + std_scaling * best_values["stability_std"][-1],
                        color="m", alpha=.1)
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

    
    if ensemble_models:
        model_selections = []
        for n_models in number_of_models:
            model_selections.append(f"with {min(n_models, len(runs))} models")
    else:
        model_selections = ["no_selection"]
    if select_good_models is not None:
        for selection_param in select_good_models:
            if int(selection_param) == selection_param:
                model_selections.append(f"num_score_{selection_param}")
            else:
                model_selections.append(f"thr_score_{selection_param}")       
        model_selections.append("weight_heuristic")
    best_heuristic_prior = {
        selection : {"metric_score": {}, "metric": {}, "score": {}} 
        for selection in model_selections}
    for model_selection_idx, model_selection in enumerate(model_selections):
        for metric_idx, metric in enumerate(metrics):
            for score_idx, score in enumerate(scores):
                local_values = {}
                for variable in variables + std_variables:
                    local_values[variable] = best_values_per_metric_score[variable][metric_idx, score_idx, model_selection_idx]
                
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
                local_values[variable] = best_values_per_metric[variable][metric_idx, model_selection_idx]

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
                local_values[variable] = best_values_per_score[variable][score_idx, model_selection_idx]

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
            local_values[variable] = best_values[variable][model_selection_idx]
        
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

    import pprint
    # Prints the nicely formatted dictionary
    pprint.pprint(best_heuristic_prior)
    for run_idx, run in enumerate(runs):
        local_runs = list(runs).copy()
        local_runs.remove(run)
        stabdir = os.path.join(outdir, run, f"stability_with_{'-'.join(local_runs)}")
        if not os.path.exists(stabdir):
            os.makedirs(stabdir)
        local_stabdir = os.path.join(
            stabdir,
            (f"n_subjects_{n_subjects}_sampling_{sampling}_sample_latents_"
            f"{sample_latents}"))
        if ensemble_models:
            local_stabdir += "_ensemble"
        if not os.path.exists(local_stabdir):
            os.makedirs(local_stabdir)
        with open(os.path.join(local_stabdir, "best_heuristic_prior"), 'wb') as f:
            pickle.dump(best_heuristic_prior, f)
    # print(json.dumps(best_heuristic_prior, indent=4))
    # return best_heur
        # print(f"Best average penalized stability overall : {best_pen_stab} for N_pen in {best_pen_N} and {best_pen_heuristic} with {best_pen_param} and daa params {best_pen_daa}.")
        # print(f"Best average stability overall : {best_stab} for N_pen in {best_N} and {best_heuristic} with {best_params} and daa params {best_daa}.")


def validate_stability(dataset, datasetdir, outdir, validation_runs=[], runs=[], 
                       metrics=["thickness", "meancurv", "area"],
                       scores=None, select_good_models=None,
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
                list(set([name.rsplit("_", 1)[0] for name in rois_names])))
    metadata = pd.read_table(
        os.path.join(datasetdir, "metadata_train.tsv"))
    metadata_columns = metadata.columns.tolist()

    ref_run = runs[0]
    other_runs = list(runs).copy()
    other_runs.remove(ref_run)
    stabdir = os.path.join(outdir, ref_run, f"stability_with_{'-'.join(other_runs)}")
    if not os.path.exists(stabdir):
        raise ValueError("the runs you provide need to have stability computed with each other")
    local_stabdir = os.path.join(
        stabdir,
        (f"n_subjects_{n_subjects}_sampling_{sampling}_sample_latents_"
            f"{sample_latents}"))
    if ensemble_models:
        local_stabdir += "_ensemble"
    if not os.path.exists(local_stabdir):
        raise ValueError("the two runs you provide need to have stability computed with each other with provided arguments")
    with open(os.path.join(local_stabdir, "best_heuristic_prior"), 'rb') as f:
        best_heuristic_prior = pickle.load(f)
    best_heuristic_priors = best_heuristic_prior

    global_results = []
    to_compare = validation_runs
    if ensemble_models:
        to_compare = range(int(len(global_results[0]) * 0.8), len(global_results[0]))
    comparisons = list(itertools.combinations(to_compare, 2))
    for compare_idx in comparisons:
        model_indices = None

        heuristics_params = {
            "pvalues_vote": {"strategy": ["vote_prop-trust_level"], "vote_prop": [0.8, 0.85, 0.9, 0.95, 1], "trust_level": [1]},
            "pvalues_min": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
            "pvalues_mean": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31))},
            "coefs_mean": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
            "coefs_max": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31)), "thr": [1e-3, 1e-5, 1e-8], "var": [1, 0.5, 0.25]},
        }
        # Computing heuristics with various parameters for each metric / score
        global_result = compute_all_associations(dataset, datasetdir, outdir, local_runs,
                                                heuristics_params, metrics,
                                                scores, model_indices,
                                                n_subjects=n_subjects,
                                                sampling=sampling,
                                                sample_latents=sample_latents,
                                                ensemble_models=ensemble_models)
        global_results.append([global_result])
        if select_good_models is not None:
            if type(select_good_models) not in (list, tuple):
                select_good_models = [select_good_models]
            if int(select_good_models[0]) == select_good_models[0]:
                # n_worsts = range(1, select_good_models + 1)
                for n_worst in select_good_models:
                    model_indices = []
                    for run in local_runs:
                        model_scores = score_models(dataset, datasetdir, outdir, run, scores=scores)
                        model_indices.append(list(range(len(model_scores))))
                        worst_models = np.argsort(model_scores)[:n_worst]
                        for model_idx in worst_models:
                            model_indices[-1].remove(model_idx)
                    model_indices = np.array(model_indices).T
                    global_result_select = compute_all_associations(dataset, datasetdir, outdir,
                                                                    local_runs, heuristics_params,
                                                                    metrics, scores, model_indices,
                                                                    n_subjects=n_subjects,
                                                                    sampling=sampling,
                                                                    sample_latents=sample_latents,
                                                                    ensemble_models=ensemble_models)
                    global_results[-1].append(global_result_select)
            else:
                for threshold in select_good_models:
                    model_indices = []
                    for run in local_runs:
                        model_scores = score_models(dataset, datasetdir, outdir, run, scores=scores)
                        model_indices.append(list(range(len(model_scores))))
                        model_indices[-1] = np.array(model_indices[-1])[model_scores >= threshold]
                        print(f"Number of removed models for run {run} :"
                            f"{len(model_scores) - len(model_indices[-1])} "
                            f"with threshold {threshold}")
                    # model_indices = np.array(model_indices).T
                    # print(model_indices.shape)
                    global_result_select = compute_all_associations(dataset, datasetdir, outdir,
                                                                    local_runs, heuristics_params,
                                                                    metrics, scores, model_indices,
                                                                    n_subjects=n_subjects,
                                                                    sampling=sampling,
                                                                    sample_latents=sample_latents,
                                                                    ensemble_models=ensemble_models)
                    global_results[-1].append(global_result_select)

            heuristics_params.update({
                "coefs_weighted_mean_score": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
                "coefs_weighted_mean_rank": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 31)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
                })
            global_result_new = compute_all_associations(
                dataset, datasetdir, outdir, local_runs, heuristics_params, metrics,
                scores, None, n_subjects=n_subjects, sampling=sampling,
                sample_latents=sample_latents)
            global_results[-1].append(global_result_new)

    values = {"stability": np.zeros((len(comparisons), len(global_results[0]))),
              "penalized_stability" : np.empty((len(comparisons), len(global_results[0]))),
              "stability_std": np.zeros((len(comparisons), len(global_results[0]))),
              "penalized_stability_std": np.zeros((len(comparisons), len(global_results[0]))),
              "validation_stability": np.empty((len(comparisons), len(global_results[0]))),
              "validation_penalized_stability":np.empty((len(comparisons), len(global_results[0]))),
              "validation_stability_std": np.zeros((len(global_results[0]))),
              "validation_penalized_stability_std": np.zeros((len(global_results[0]))),
              "heuristic": np.empty((len(comparisons), len(global_results[0])), dtype=object),
              "strat_param" : np.empty((len(comparisons), len(global_results[0])), dtype=object),
              "prior": np.empty((len(comparisons), len(global_results[0])), dtype=int),
    }

    values_per_metric_score = {
        "stability": np.zeros((len(metrics), len(scores), len(comparisons), len(global_results[0]))),
        "penalized_stability" : np.zeros((len(metrics), len(scores), len(comparisons), len(global_results[0]))),
        "stability_std": np.zeros((len(metrics), len(scores), len(comparisons), len(global_results[0]))),
        "penalized_stability_std": np.zeros((len(metrics), len(scores), len(comparisons), len(global_results[0]))),
        "validation_stability": np.zeros((len(metrics), len(scores), len(comparisons), len(global_results[0]))),
        "validation_penalized_stability":np.zeros((len(metrics), len(scores), len(comparisons), len(global_results[0]))),
        "validation_stability_std": np.zeros((len(global_results[0]))),
        "validation_penalized_stability_std": np.zeros((len(global_results[0]))),
        "heuristic": np.empty((len(metrics), len(scores), len(comparisons), len(global_results[0])), dtype=object),
        "strat_param" : np.empty((len(metrics), len(scores), len(comparisons), len(global_results[0])), dtype=object),
        "prior": np.empty((len(metrics), len(scores), len(comparisons), len(global_results[0])), dtype=int),
    }

    values_per_metric = {
        "stability": np.zeros((len(metrics), len(comparisons), len(global_results[0]))),
        "penalized_stability" : np.zeros((len(metrics), len(comparisons), len(global_results[0]))),
        "stability_std": np.zeros((len(metrics), len(comparisons), len(global_results[0]))),
        "penalized_stability_std": np.zeros((len(metrics), len(comparisons), len(global_results[0]))),
        "validation_stability": np.zeros((len(metrics), len(comparisons), len(global_results[0]))),
        "validation_penalized_stability": np.zeros((len(metrics), len(comparisons), len(global_results[0]))),
        "validation_stability_std": np.zeros((len(global_results[0]))),
        "validation_penalized_stability_std": np.zeros((len(global_results[0]))),
        "heuristic": np.empty((len(metrics), len(comparisons), len(global_results[0])), dtype=object),
        "strat_param" : np.empty((len(metrics), len(comparisons), len(global_results[0])), dtype=object),
        "prior": np.empty((len(metrics), len(comparisons), len(global_results[0])), dtype=int),
    }

    values_per_score = {
        "stability": np.zeros((len(scores), len(comparisons), len(global_results[0]))),
        "penalized_stability" : np.zeros((len(scores), len(comparisons), len(global_results[0]))),
        "stability_std": np.zeros((len(scores), len(comparisons), len(global_results[0]))),
        "penalized_stability_std": np.zeros((len(scores), len(comparisons), len(global_results[0]))),
        "validation_stability": np.zeros((len(scores), len(comparisons), len(global_results[0]))),
        "validation_penalized_stability": np.zeros((len(scores), len(comparisons), len(global_results[0]))),
        "validation_stability_std": np.zeros((len(global_results[0]))),
        "validation_penalized_stability_std": np.zeros((len(global_results[0]))),
        "heuristic": np.empty((len(scores), len(comparisons), len(global_results[0])), dtype=object),
        "strat_param" : np.empty((len(scores), len(comparisons), len(global_results[0])), dtype=object),
        "prior": np.empty((len(scores), len(comparisons), len(global_results[0])), dtype=int),
    }

    variables = list(values.keys())

    model_selections = ["no_selection"]
    if select_good_models is not None:
        for selection_param in select_good_models:
            if int(selection_param) == selection_param:
                model_selections.append(f"num_score_{selection_param}")
            else:
                model_selections.append(f"thr_score_{selection_param}")       
        model_selections.append("weight_heuristic")

    # Compute penalized stability for each ideal_N value
    all_stabilities = {"overall": [[] for _ in model_selections],
                       "metric": [[] for _ in model_selections],
                       "score": [[] for _ in model_selections]}
    all_pen_stabilities = {"overall": [[] for _ in model_selections],
                           "metric": [[] for _ in model_selections],
                           "score": [[] for _ in model_selections]}
    for comparison_idx in range(len(comparisons)):
        for result_idx, result in enumerate(global_results[comparison_idx]):
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
                    daa_params = f"{sampling}_{sample_latents}"
                    local_stability_per_metric_score = compute_all_stability(
                        result, daa_params, heuristic, strat_param,
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
                daa_params = f"{sampling}_{sample_latents}"
                local_stability_per_metric = compute_all_stability(
                    result, daa_params, heuristic, strat_param,
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
                daa_params = f"{sampling}_{sample_latents}"
                local_stability_per_score = compute_all_stability(
                    result, daa_params, heuristic, strat_param,
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
            daa_params = f"{sampling}_{sample_latents}"
            local_stability = compute_all_stability(
                result, daa_params, heuristic, strat_param,
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
    print(f"Without selection : {penalized_stab.mean(0)[0]} "
          f"and validation {validation_pen_stab.mean(0)[0]} with "
          f"std {validation_pen_stab.std(0)[0]}")
    print(f"With model selection thr : {penalized_stab.mean(0)[1]} "
          f"and validation {validation_pen_stab.mean(0)[1]} with "
          f"std {validation_pen_stab.std(0)[1]}")
    print(f"With weighted heuristics : {penalized_stab.mean(0)[2]} "
          f"and validation {validation_pen_stab.mean(0)[2]} with "
          f"std {validation_pen_stab.std(0)[2]})")
    # array_of_pen_stabs = np.concatenate(([penalized_stab[0]], [validation_pen_stab[0]], [validation_pen_stab[1]]), axis=0)
    # array_of_stabs = np.concatenate(([stab[0]], [validation_stab[0]], [validation_stab[1]]), axis=0)
    # array_of_pen_stabs = np.concatenate(([validation_pen_stab[comp_idx]] for comp_idx in range(len(comparisons))), axis=0)
    # array_of_stabs = np.concatenate(([validation_stab[comp_idx]] for comp_idx in range(len(comparisons))), axis=0)
    print(f"Average stability without selection : {validation_stab.mean(0)[0]} +- {validation_stab.std(0)[0]} (pen : {validation_pen_stab.mean(0)[0]} +- {validation_pen_stab.std(0)[0]})")
    print(f"Average stability with selection thr : {validation_stab.mean(0)[1]} +- {validation_stab.std(0)[1]} (pen : {validation_pen_stab.mean(0)[1]} +- {validation_pen_stab.std(0)[1]})")
    print(f"Average stability with weighted heuristics : {validation_stab.mean(0)[2]} +- {validation_stab.std(0)[2]} (pen : {validation_pen_stab.mean(0)[2]} +- {validation_pen_stab.std(0)[2]})")
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
    average_stab = [[], [], []]
    deviations = [[], [], []]
    average_pen_stab = [[], [], []]
    pen_deviations = [[], [], []]
    initial_stab = [[], [], []]
    initial_stab_std = [[], [], []]
    initial_pen_stab = [[], [], []]
    initial_pen_stab_std = [[], [], []]
    for metric_idx, metric in enumerate(metrics):
        penalized_stab = values_per_metric["penalized_stability"][metric_idx]
        penalized_stab_std = values_per_metric["penalized_stability_std"][metric_idx]
        validation_pen_stab = values_per_metric["validation_penalized_stability"][metric_idx]
        stab = values_per_metric["stability"][metric_idx]
        stab_std = values_per_metric["stability_std"][metric_idx]
        validation_stab = values_per_metric["validation_stability"][metric_idx]
        print(f"Validating penalized stability for {metric}:")
        print(f"Without selection : {penalized_stab.mean(0)[0]} and validation"
              f" {validation_pen_stab.mean(0)[0]} with std "
              f"{validation_pen_stab.std(0)[0]}")
        print(f"With model selection thr : {penalized_stab.mean(0)[1]} "
              f"and validation {validation_pen_stab.mean(0)[1]} with "
              f"std {validation_pen_stab.std(0)[1]}")
        print(f"With weighted heuristics : {penalized_stab.mean(0)[2]} "
              f"and validation {validation_pen_stab.mean(0)[2]} with "
              f"std {validation_pen_stab.std(0)[2]}")
        print()
        # array_of_pen_stabs = np.concatenate(([penalized_stab[0]], [validation_pen_stab[0]], [validation_pen_stab[1]]), axis=0)
        # array_of_stabs = np.concatenate(([stab[0]], [validation_stab[0]], [validation_stab[1]]), axis=0)
        # array_of_pen_stabs = np.concatenate(([validation_pen_stab[comp_idx]] for comp_idx in range(len(comparisons))), axis=0)
        # array_of_stabs = np.concatenate(([validation_pen_stab[comp_idx]] for comp_idx in range(len(comparisons))), axis=0)
        for select_idx in range(len(model_selections)):
            average_stab[select_idx].append(validation_stab.mean(0)[select_idx])
            # deviations[select_idx].append(validation_stab.std(0)[select_idx])
            average_pen_stab[select_idx].append(validation_pen_stab.mean(0)[select_idx])
            pen_deviations[select_idx].append(validation_pen_stab.std(0)[select_idx])
            initial_stab[select_idx].append(stab[0][select_idx])
            initial_pen_stab[select_idx].append(penalized_stab[0][select_idx])
            initial_stab_std[select_idx].append(stab_std[0][select_idx])
            initial_pen_stab_std[select_idx].append(penalized_stab_std[0][select_idx])
    print(f"Average stability without selection : {np.mean(average_stab[0])} +- {np.mean(deviations[0])} (pen {np.mean(average_pen_stab[0])} +- {np.mean(pen_deviations[0])}")
    print(f"Average stability with selection thr : {np.mean(average_stab[1])} +- {np.mean(deviations[1])} (pen {np.mean(average_pen_stab[1])} +- {np.mean(pen_deviations[1])}")
    print(f"Average stability with weighted heuristics : {np.mean(average_stab[2])} +- {np.mean(deviations[2])} (pen {np.mean(average_pen_stab[2])} +- {np.mean(pen_deviations[2])}")
    print()
    all_stability["metric"] = {}
    all_stability["metric"]["mean"] = np.mean(average_stab, 1)
    # all_stability["metric"]["std"] = np.mean(deviations, 1)
    # all_stability["metric"]["std"] = np.std(average_stab, 1)
    all_stability["metric"]["std"] = values_per_metric["validation_stability"].mean(0).std(0)
    if global_std:
        all_stability["metric"]["std"] = values_per_metric["validation_stability_std"]
    all_stability["metric"]["initial"] = np.mean(initial_stab, 1)
    all_stability["metric"]["initial_std"] = np.mean(initial_stab_std, 1)
    all_penalized_stability["metric"] = {}
    all_penalized_stability["metric"]["mean"] = np.mean(average_pen_stab, 1)
    # all_penalized_stability["metric"]["std"] = np.mean(pen_deviations, 1)
    # all_penalized_stability["metric"]["std"] = np.std(average_pen_stab, 1)
    all_penalized_stability["metric"]["std"] = values_per_metric["validation_penalized_stability"].mean(0).std(0)
    if global_std:
        all_penalized_stability["metric"]["std"] = values_per_metric["validation_penalized_stability_std"]
    all_penalized_stability["metric"]["initial"] = np.mean(initial_pen_stab, 1)
    all_penalized_stability["metric"]["initial_std"] = np.mean(initial_pen_stab_std, 1)
    
    print("Validation penalized stability per score")
    average_stab = [[], [], []]
    deviations = [[], [], []]
    average_pen_stab = [[], [], []]
    pen_deviations = [[], [], []]
    initial_stab = [[], [], []]
    initial_stab_std = [[], [], []]
    initial_pen_stab = [[], [], []]
    initial_pen_stab_std = [[], [], []]
    for score_idx, score in enumerate(scores):
        penalized_stab = values_per_score["penalized_stability"][score_idx]
        penalized_stab_std = values_per_score["penalized_stability_std"][score_idx]
        validation_pen_stab = values_per_score["validation_penalized_stability"][score_idx]
        stab = values_per_score["stability"][score_idx]
        stab_std = values_per_score["stability_std"][score_idx]
        validation_stab = values_per_score["validation_stability"][score_idx]
        print(f"Validating penalized stability for {score}:")
        print(f"Without selection : {penalized_stab.mean(0)[0]} "
          f"and validation {validation_pen_stab.mean(0)[0]} with "
          f"std {validation_pen_stab.std(0)[0]}")
        print(f"With model selection thr : {penalized_stab.mean(0)[1]} "
            f"and validation {validation_pen_stab.mean(0)[1]} with "
            f"std {validation_pen_stab.std(0)[1]}")
        print(f"With weighted heuristics : {penalized_stab.mean(0)[2]} "
            f"and validation {validation_pen_stab.mean(0)[2]} with "
            f"std {validation_pen_stab.std(0)[2]}")
        print()
        # array_of_pen_stabs = np.concatenate(([penalized_stab[0]], [validation_pen_stab[0]], [validation_pen_stab[1]]), axis=0)
        # array_of_stabs = np.concatenate(([stab[0]], [validation_stab[0]], [validation_stab[1]]), axis=0)
        # array_of_pen_stabs = np.concatenate(([validation_pen_stab[0]], [validation_pen_stab[1]]), axis=0)
        # array_of_stabs = np.concatenate(([validation_stab[0]], [validation_stab[1]]), axis=0)
        for select_idx in range(len(model_selections)):
            average_stab[select_idx].append(validation_stab.mean(0)[select_idx])
            deviations[select_idx].append(validation_stab.std(0)[select_idx])
            average_pen_stab[select_idx].append(validation_pen_stab.mean(0)[select_idx])
            pen_deviations[select_idx].append(validation_pen_stab.std(0)[select_idx])
            initial_stab[select_idx].append(stab[0][select_idx])
            initial_pen_stab[select_idx].append(penalized_stab[0][select_idx])
            initial_stab_std[select_idx].append(stab_std[0][select_idx])
            initial_pen_stab_std[select_idx].append(penalized_stab_std[0][select_idx])

    print(f"Average stability without selection : {np.mean(average_stab[0])} +- {np.mean(deviations[0])} (pen {np.mean(average_pen_stab[0])} +- {np.mean(pen_deviations[0])}")
    print(f"Average stability with selection thr : {np.mean(average_stab[1])} +- {np.mean(deviations[1])} (pen {np.mean(average_pen_stab[1])} +- {np.mean(pen_deviations[1])}")
    print(f"Average stability with weighted heuristics : {np.mean(average_stab[2])} +- {np.mean(deviations[2])} (pen {np.mean(average_pen_stab[2])} +- {np.mean(pen_deviations[2])}")
    print()
    all_stability["score"] = {}
    all_stability["score"]["mean"] = np.mean(average_stab, 1)
    # all_stability["score"]["std"] = np.mean(deviations, 1)
    # all_stability["score"]["std"] = np.std(average_stab, 1)
    all_stability["score"]["std"] = values_per_score["validation_stability"].mean(0).std(0)
    if global_std:
        all_stability["score"]["std"] = values_per_score["validation_stability_std"]
    all_stability["score"]["initial"] = np.mean(initial_stab, 1)
    all_stability["score"]["initial_std"] = np.mean(initial_stab_std, 1)
    all_penalized_stability["score"] = {}
    all_penalized_stability["score"]["mean"] = np.mean(average_pen_stab, 1)
    # all_penalized_stability["score"]["std"] = np.mean(pen_deviations, 1)
    # all_penalized_stability["score"]["std"] = np.std(average_pen_stab, 1)
    all_penalized_stability["score"]["std"] = values_per_score["validation_penalized_stability"].mean(0).std(0)
    if global_std:
        all_penalized_stability["score"]["std"] = values_per_score["validation_penalized_stability_std"]
    all_penalized_stability["score"]["initial"] = np.mean(initial_pen_stab, 1)
    all_penalized_stability["score"]["initial_std"] = np.mean(initial_pen_stab_std, 1)

    print("Validation penalized stability per metric-score")
    average_stab = [[], [], []]
    deviations = [[], [], []]
    average_pen_stab = [[], [], []]
    pen_deviations = [[], [], []]
    initial_stab = [[], [], []]
    initial_stab_std = [[], [], []]
    initial_pen_stab = [[], [], []]
    initial_pen_stab_std = [[], [], []]
    for metric_idx, metric in enumerate(metrics):
        for score_idx, score in enumerate(scores):
            penalized_stab = values_per_metric_score["penalized_stability"][metric_idx, score_idx]
            penalized_stab_std = values_per_metric_score["penalized_stability_std"][metric_idx, score_idx]
            validation_pen_stab = values_per_metric_score["validation_penalized_stability"][metric_idx, score_idx]
            stab = values_per_metric_score["stability"][metric_idx, score_idx]
            stab_std = values_per_metric_score["stability_std"][metric_idx, score_idx]
            validation_stab = values_per_metric_score["validation_stability"][metric_idx, score_idx]
            print(f"Validating penalized stability for {metric} and {score}:")
            print(f"Without selection : {penalized_stab.mean(0)[0]} "
                f"and validation {validation_pen_stab.mean(0)[0]} with "
                f"std {validation_pen_stab.std(0)[0]}")
            print(f"With model selection thr : {penalized_stab.mean(0)[1]} "
                f"and validation {validation_pen_stab.mean(0)[1]} with "
                f"std {validation_pen_stab.std(0)[1]}")
            print(f"With weighted heuristics : {penalized_stab.mean(0)[2]} "
                f"and validation {validation_pen_stab.mean(0)[2]} with "
                f"std {validation_pen_stab.std(0)[2]}")
            print()
            # array_of_pen_stabs = np.concatenate(([penalized_stab[0]], [validation_pen_stab[0]], [validation_pen_stab[1]]), axis=0)
            # array_of_stabs = np.concatenate(([stab[0]], [validation_stab[0]], [validation_stab[1]]), axis=0)
            # array_of_pen_stabs = np.concatenate(([validation_pen_stab[0]], [validation_pen_stab[1]]), axis=0)
            # array_of_stabs = np.concatenate(([validation_stab[0]], [validation_stab[1]]), axis=0)
            for select_idx in range(len(model_selections)):
                average_stab[select_idx].append(validation_stab.mean(0)[select_idx])
                deviations[select_idx].append(validation_stab.std(0)[select_idx])
                average_pen_stab[select_idx].append(validation_pen_stab.mean(0)[select_idx])
                pen_deviations[select_idx].append(validation_pen_stab.std(0)[select_idx])
                initial_stab[select_idx].append(stab[0][select_idx])
                initial_pen_stab[select_idx].append(penalized_stab[0][select_idx])
                initial_stab_std[select_idx].append(stab_std[0][select_idx])
                initial_pen_stab_std[select_idx].append(penalized_stab_std[0][select_idx])
    print(f"Average stability without selection : {np.mean(average_stab[0])} +- {np.mean(deviations[0])} (pen {np.mean(average_pen_stab[0])} +- {np.mean(pen_deviations[0])}")
    print(f"Average stability with selection thr : {np.mean(average_stab[1])} +- {np.mean(deviations[1])} (pen {np.mean(average_pen_stab[1])} +- {np.mean(pen_deviations[1])}")
    print(f"Average stability with weighted heuristics : {np.mean(average_stab[2])} +- {np.mean(deviations[2])} (pen {np.mean(average_pen_stab[2])} +- {np.mean(pen_deviations[2])}")
    print()
    all_stability["metric_score"] = {}
    all_stability["metric_score"]["mean"] = np.mean(average_stab, 1)
    # all_stability["metric_score"]["std"] = np.mean(deviations, 1)
    # all_stability["metric_score"]["std"] = np.std(average_stab, 1)
    all_stability["metric_score"]["std"] = values_per_metric_score["validation_stability"].mean((0, 1)).std(0)
    if global_std:
        all_stability["metric_score"]["std"] = values_per_metric_score["validation_stability_std"]
    all_stability["metric_score"]["initial"] = np.mean(initial_stab, 1)
    all_stability["metric_score"]["initial_std"] = np.mean(initial_stab_std, 1)
    all_penalized_stability["metric_score"] = {}
    all_penalized_stability["metric_score"]["mean"] = np.mean(average_pen_stab, 1)
    # all_penalized_stability["metric_score"]["std"] = np.mean(pen_deviations, 1)
    # all_penalized_stability["metric_score"]["std"] = np.std(average_pen_stab, 1)
    all_penalized_stability["metric_score"]["std"] = values_per_metric_score["validation_penalized_stability"].mean((0, 1)).std(0)
    if global_std:
        all_penalized_stability["metric_score"]["std"] = values_per_metric_score["validation_penalized_stability_std"]
    all_penalized_stability["metric_score"]["initial"] = np.mean(initial_pen_stab, 1)
    all_penalized_stability["metric_score"]["initial_std"] = np.mean(initial_pen_stab_std, 1)

    groups = ["overall", "metric", "score", "metric_score"]
    x = np.arange(len(groups))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')

    for selection_idx, selection in enumerate(model_selections):
        offset = width * multiplier
        means = []
        stds = []
        initials = []
        initial_stds = []
        for granularity in groups:
            means.append(np.round(np.mean(all_stability[granularity]["mean"][selection_idx]), 3))
            stds.append(np.mean(all_stability[granularity]["std"][selection_idx]))
            initials.append(np.round(np.mean(all_stability[granularity]["initial"][selection_idx]), 3))
            initial_stds.append(np.mean(all_stability[granularity]["initial_std"][selection_idx]))
        rects = ax.bar(x + offset, means, width, label=selection, yerr=stds, edgecolor="white")
        ax.bar_label(rects, padding=3)
        # ax.bar(x + offset, initials, width, label="initial", fill=False)
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
    ax.set_xticks(x + width, groups)
    ax.legend(loc="upper left", ncols=3, title="Model selection strategy")
    ax.set_ylim(0, 1.1)

    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')

    for selection_idx, selection in enumerate(model_selections):
        offset = width * multiplier
        means = []
        stds = []
        initials = []
        initial_stds = []
        for granularity in groups:
            means.append(np.round(np.mean(all_penalized_stability[granularity]["mean"][selection_idx]), 3))
            stds.append(np.mean(all_penalized_stability[granularity]["std"][selection_idx]))
            initials.append(np.round(np.mean(all_penalized_stability[granularity]["initial"][selection_idx]), 3))
            initial_stds.append(np.mean(all_penalized_stability[granularity]["initial_std"][selection_idx]))
        rects = ax.bar(x + offset, means, width, label=selection, yerr=stds, edgecolor="white")
        ax.bar_label(rects, padding=3)
        # ax.bar(x + offset, initials, width, label="initial", fill=False)#, edgecolor="white")
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
    ax.set_xticks(x + width, groups)
    ax.legend(loc="upper left", ncols=3, title="Model selection strategy")
    ax.set_ylim(0, 1.1)

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
                list(set([name.rsplit("_", 1)[0] for name in rois_names])))
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
        list(set([name.rsplit("_", 1)[0] for name in rois_names])))

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
                                min_occurence=5, n_subjects=301,
                                sampling="likelihood", sample_latents=False,
                                permuted=False, ensemble_models=False):
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
    if granularity != "overall":
        heuristic_param[granularity] = {}
        for key, value in best_heuristic_prior[model_selection][granularity].items():
            _, _, _, _, _, heuristic_name, strat_param = value
            strategy = strat_param.split("strategy_")[-1].split("_")[0]
            params = {"strategy": strategy}
            if "-" not in strategy:
                _, param_value = strat_param.rsplit("_", 1)
                params[strategy] = float(param_value) if strategy != "num" else int(param_value)
            else:
                first_param, second_param = strategy.split("-")
                _, first_value, second_value = strat_param.rsplit("_", 2)
                params[first_param] = int(first_value)
                params[second_param] = float(second_value)
            heuristic_param[granularity][key] = {heuristic_name: params}
    else:
        value = best_heuristic_prior[model_selection]["overall"]
        _, _, _, _, _, heuristic_name, strat_param = value
        strategy = strat_param.split("strategy_")[-1].split("_")[0]
        params = {"strategy": strategy}
        if "-" not in strategy:
            _, param_value = strat_param.rsplit("_", 1)
            params[strategy] = float(param_value) if strategy != "num" else int(param_value)
        else:
            first_param, second_param = strategy.split("-")
            _, first_value, second_value = strat_param.rsplit("_", 2)
            params[first_param] = int(first_value)
            params[second_param] = float(second_value)
        heuristic_param[heuristic_name] = params
    print(heuristic_param)
    heuristic = Heuristic(heuristic_param, additional_data)

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
        model_indices = list(range(len(model_scores)))
        if model_thr is not None:
            model_indices = np.array(model_indices)[model_scores >= model_thr]
            print(f"Number of removed models for run {run} :"
                f"{len(model_scores) - len(model_indices)} "
                f"with threshold {model_thr}")

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
            print_subtitle(f"Computing stable associations for {dirname}")
            coefs = np.load(os.path.join(dirname, "coefs.npy"))
            pvalues = np.load(os.path.join(dirname, "pvalues.npy"))

            if model_thr is not None:
                coefs = coefs[model_indices]
                pvalues = pvalues[model_indices]
                model_scores = model_scores[model_indices]
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
        else:
            coefs = all_coefs[ens_idx]
            pvalues = all_pvalues[ens_idx]
            model_scores = all_model_scores[ens_idx]
        df, agg_values = heuristic(coefs, pvalues, model_scores,
                                    return_agg=True)
        associations.append(df)
        # all_agg_values.append(agg_values)
        all_agg_values.append(coefs.mean((0, 1)))

    print_text(f"Number of associations : {len(associations)}")
    df = associations[0]
    print(df.shape[0])
    counts = []
    for idx in range(1, len(associations)):
        counts += list(associations[idx].itertuples(index=False))
        print(associations[idx].shape[0])
        # df = df.merge(associations[idx], how="outer")
    counter = collections.Counter(counts)
    kept_records = [item[0] for item in counter.most_common(len(counts)) if item[1] >= min_occurence]
    coefs = np.array(all_agg_values).mean(0)

    return pd.DataFrame(kept_records), coefs


def daa_plot_most_connected_stable(dataset, datasetdir, outdir, runs, validation_runs=[],
                                granularity="overall", model_selection="no_selection",
                                metrics=["thickness", "meancurv", "area"],
                                scores=None, plot_associations=False,
                                n_connections=4, min_occurence=5, n_subjects=301,
                                sampling="likelihood", sample_latents=False,
                                ensemble_models=False):
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
    from plotting import plot_surf_mosaic, plot_areas
    from multimodal_cohort.constants import short_clinical_names
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    from nilearn import datasets
    import seaborn as sns
    from color_utils import plt_to_plotly_rgb, get_color_list

    print_title(f"PLOT DAA most associated rois across scores with stability.")

    clinical_names = np.load(
        os.path.join(datasetdir, "clinical_names.npy"), allow_pickle=True)
    clinical_names = clinical_names.tolist()
    if scores is None:
        scores = clinical_names
    rois_names = np.load(
        os.path.join(datasetdir, "rois_names.npy"), allow_pickle=True)
    metadata = pd.read_table(
        os.path.join(datasetdir, "metadata_train.tsv"))
    metadata_columns = metadata.columns.tolist()

    additional_data = SimpleNamespace(metadata_columns=metadata_columns,
                                      clinical_names=clinical_names,
                                      rois_names=rois_names,
                                      scores=scores, metrics=metrics)

    rois_names = rois_names.tolist()

    stabdir = os.path.join(outdir, runs[0], f"stability_with_{'-'.join(runs[1:])}")
    local_stabdir = os.path.join(
        stabdir,
        (f"n_subjects_{n_subjects}_sampling_{sampling}_sample_latents_"
            f"{sample_latents}"))
    if ensemble_models:
        local_stabdir += "_ensemble"
    if not os.path.exists(local_stabdir):
        raise ValueError("You must compute the stability between runs before "
                         "ploting final results with validation runs.")
    with open(os.path.join(local_stabdir, "best_heuristic_prior"), 'rb') as f:
        best_heuristic_prior = pickle.load(f)

    ref_run = runs[0]
    other_runs = runs[1:]
    compute_across_runs = runs
    if len(validation_runs) != 0:
        ref_run = validation_runs[0]
        other_runs = validation_runs[1:]
        compute_across_runs = validation_runs
    expdir = os.path.join(outdir, ref_run)
    resultsdir = os.path.join(expdir, f"results_with_{'-'.join(other_runs)}")
    if not os.path.isdir(resultsdir):
        os.makedirs(resultsdir)

    df, coefs = compute_stable_associations(dataset, datasetdir, outdir,
        best_heuristic_prior, compute_across_runs, additional_data,
        granularity, model_selection, min_occurence, n_subjects,
        sampling, sample_latents, ensemble_models=ensemble_models)
   
    print_text(f"Number of associations : {df.shape[0]}")
    print_subtitle(f"Plot regression coefficients radar plots...")

    marker_signif = "star"
    marker_non_signif = "circle"
    counts = collections.Counter(df["roi"].values)
    # selected_rois = [item[0] for item in counts.most_common()]
    # n_colors = n_rois * len(df["metric"].unique())
    color_name = "tab20+tab20b+tab20c"
    # if n_colors < 10:
    #     color_name = "Plotly"
    # elif n_colors < 13:
    #     color_name = "Paired"
    # elif n_colors < 21:
    #     color_name = "tab20"
    textfont = dict(
        size=20,
        family="Droid Serif")
    colors = get_color_list(color_name)#, n_colors)
    all_selected_rois = []
    for _metric, _df in df.groupby(["metric"]):
        selected_scores = []
        significativity = []
        coefs_sign = []
        counts = collections.Counter(_df["roi"].values)
        # print(counts.most_common(n_rois+10))
        selected_rois = [item[0] for item in counts.most_common(len(list(counts))) if item[1] >= n_connections]
        for _roi in selected_rois:
            roi_idx = rois_names.index(f"{_roi}_{_metric}")
            #if n_models > 1:
            selected_coefs = coefs[:, roi_idx]
            # else:
            #    selected_coefs = coefs[:, :, roi_idx].mean(axis=0)
            selected_scores.append(np.absolute(selected_coefs))
            # idx_sign = [((df["metric"] == _metric) & (df["roi"] == _roi) & (df["score"] == _score)).any() for _score in scores]
            idx_sign = [_score in _df.loc[_df["roi"] == _roi, "score"].values for _score in scores]
            significativity.append(idx_sign)
            coefs_sign.append((selected_coefs >= 0).astype(int))
        all_selected_rois += [area for area in selected_rois if area not in all_selected_rois]
        selected_scores = np.asarray(selected_scores)
        fig = go.Figure()
        for roi_idx, _roi in enumerate(selected_rois):
            color_idx = all_selected_rois.index(_roi)
            color = plt_to_plotly_rgb(colors[color_idx])
            _scores = selected_scores[roi_idx].tolist()
            fig.add_trace(
                go.Scatterpolar(
                    r=_scores + _scores[:1],
                    theta=[
                        "<b>" + short_clinical_names[dataset][name] + "</b>"
                        for name in clinical_names + clinical_names[:1]],
                    mode="lines+text",
                    marker_color=color,
                    legendgroup="roi",
                    legendgrouptitle = dict(
                        font=dict(
                            size=textfont["size"] + 4,
                            family=textfont["family"]),
                        text="<b>ROIs</b>"),
                    name=_roi))
        for marker, sign_label, sign in [
            (marker_non_signif, "non significative", False),
            (marker_signif, "significative", True)]:
            significative_scores = []
            score_names = []
            markers = []
            color = []
            for roi_idx, roi_coefs in enumerate(selected_scores):
                for coef_idx, coef in enumerate(roi_coefs):
                    if significativity[roi_idx][coef_idx] == sign:
                        significative_scores.append(coef)
                        score_names.append(clinical_names[coef_idx])
                        markers.append(marker)
                        if coefs_sign[roi_idx][coef_idx] == 1:
                            color.append("red")
                        else:
                            color.append("blue")
            fig.add_trace(go.Scatterpolar(
                r=np.array(significative_scores),
                theta=np.array(["<b>" + short_clinical_names[dataset][name]
                                + "</b>" for name in score_names]),
                # fill='toself',
                mode="markers",
                legendgroup="significativity",
                legendgrouptitle = dict(
                    font=dict(
                        size=textfont["size"] + 4,
                        family="Droid Serif"),
                    text="<b>Significativity</b>"),
                marker_symbol=np.array(markers),
                marker_size=5,
                marker_color=color,
                name=sign_label
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True, showticklabels=False, ticks="",
                    range=[0, np.array(selected_scores).max() + 0.003])),
            font=textfont)
        filename = os.path.join(
            resultsdir, f"three_selected_rois_{_metric}_polarplots.png")
        fig.write_image(filename)
        print_result(f"{_metric} regression coefficients for 3 selected "
                        f"ROIs: {filename}")

    filename = os.path.join(resultsdir, "most_connected_rois.png")
    plot_areas(all_selected_rois, np.arange(len(all_selected_rois)), filename, color_name)

    print_subtitle(f"Plot significant scores/ROIs flows...")
    if plot_associations:
        for _metric, _df in df.groupby(["metric"]):
            significant_scores = _df["score"].values
            significant_rois = _df["roi"].values
            significant_coefs = []
            colors = []
            for _roi, _score in zip(significant_rois, significant_scores):
                score_idx = clinical_names.index(_score)
                roi_idx = rois_names.index(f"{_roi}_{_metric}")
                # if n_models > 1:
                significant_coef = coefs[score_idx, roi_idx]
                # else:
                #   significant_coef = coefs[:, score_idx, roi_idx].mean()
                significant_coefs.append(significant_coef)
            significant_coefs = np.asarray(significant_coefs)
            colors = ["rgba(255,0,0,0.4)" if coef > 0 else
                        "rgba(0,0,255,0.4)" for coef in significant_coefs]
            sankey_plot = go.Parcats(
                domain={"x": [0.05, 0.9], "y": [0, 1]},
                dimensions=[{"label": "Score",
                                "values": significant_scores},
                            {"label": "ROI", "values": significant_rois}],
                counts=np.abs(significant_coefs),
                line={"color": colors, "shape": "hspline"},
                labelfont=dict(family="Droid Serif", size=28),
                tickfont=dict(family="Droid Serif", size=20))
            fig = go.Figure(data=[sankey_plot])
            filename = os.path.join(
                resultsdir, f"score2roi_{_metric}_flow.png")
            fig.write_image(filename)
            print_result(f"flow for the {_metric} metric: {filename}")


def daa_plot_metric_score_stable(dataset, datasetdir, outdir, runs, score, metric,
                                 validation_runs=[], granularity="overall",
                                 model_selection="no_selection",
                                 min_occurence=5, plot_associations=False,
                                 n_subjects=301, sampling="likelihood",
                                 sample_latents=False, rescaled=True,
                                 ensemble_models=False):
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
    expdir = os.path.join(outdir, runs[0])

    flags_file = os.path.join(expdir, "flags.rar")
    if not os.path.isfile(flags_file):
        raise ValueError("You need first to train the model.")
    checkpoints_dir = os.path.join(expdir, "checkpoints")
    experiment, flags = MultimodalExperiment.get_experiment(
        flags_file, checkpoints_dir)
    n_models = flags.num_models
    scalers = experiment.scalers

    clinical_names = np.load(
        os.path.join(datasetdir, "clinical_names.npy"), allow_pickle=True)
    clinical_names = clinical_names.tolist()
    rois_names = np.load(
        os.path.join(datasetdir, "rois_names.npy"), allow_pickle=True)
    metadata = pd.read_table(
        os.path.join(datasetdir, "metadata_train.tsv"))
    metadata_columns = metadata.columns.tolist()

    additional_data = SimpleNamespace(metadata_columns=metadata_columns,
                                      clinical_names=clinical_names,
                                      rois_names=rois_names,
                                      scores=[score], metrics=[metric])

    rois_names = rois_names.tolist()

    stabdir = os.path.join(outdir, runs[0], f"stability_with_{'-'.join(runs[1:])}")
    local_stabdir = os.path.join(
        stabdir,
        (f"n_subjects_{n_subjects}_sampling_{sampling}_sample_latents_"
            f"{sample_latents}"))
    if ensemble_models:
        local_stabdir += "_ensemble"
    if not os.path.exists(local_stabdir):
        raise ValueError("You must compute the stability between runs before "
                         "ploting final results with validation runs.")
    with open(os.path.join(local_stabdir, "best_heuristic_prior"), 'rb') as f:
        best_heuristic_prior = pickle.load(f)

    ref_run = runs[0]
    other_runs = runs[1:]
    compute_across_runs = runs
    if len(validation_runs) != 0:
        ref_run = validation_runs[0]
        other_runs = validation_runs[1:]
        compute_across_runs = validation_runs
    expdir = os.path.join(outdir, ref_run)
    resultsdir = os.path.join(expdir, f"results_with_{'-'.join(other_runs)}")
    if not os.path.isdir(resultsdir):
        os.makedirs(resultsdir)

    df, coefs = compute_stable_associations(dataset, datasetdir, outdir,
        best_heuristic_prior, compute_across_runs, additional_data,
        granularity, model_selection, min_occurence, n_subjects,
        sampling, sample_latents, ensemble_models=ensemble_models)
   
    marker_signif = "star"
    marker_non_signif = "circle"

    areas = df["roi"][(df["metric"] == metric) & (df["score"] == score)].to_list()
    area_idx = [rois_names.index(f"{name}_{metric}") for name in areas]
    score_idx = clinical_names.index(score)
    values = coefs[score_idx, area_idx]
    if rescaled:
        scaling_factors = []
        for roi_idx in area_idx:
            scaling_factor = sum([
                scalers[i]["rois"].scale_[roi_idx] /
                scalers[i]["clinical"].scale_[score_idx]
                for i in range(n_models)]) / n_models
            scaling_factors.append(scaling_factor)
        scaling_factors = np.asarray(scaling_factors)
        values *= scaling_factors

    print_subtitle(f"Plot regression coefficients ...")
    # color_name = "Plotly"
    # if len(areas) <= 6:
    #     color_name = "tab10"
    # elif len(areas) <= 9:
    #     color_name = "Plotly"
    # elif len(areas) <= 10:
    #     color_name = "tab10"
    # elif len(areas) <= 12:
    #     color_name = "Paired"
    # else:
        # color_name = "Alphabet"
    color_name = "tab20+tab20b+tab20c"
    print(f"Number of significative rois in {metric} for {score} : {len(areas)}")
    filename_areas = os.path.join(
        resultsdir, f"associated_rois_for_{score}_in_{metric}.png")
    filename_bar = os.path.join(
        resultsdir, f"association_for_{score}_in_{metric}.png")
    plt.rcParams.update({'font.size': 20, "font.family": "serif"})
    plot_areas(areas, np.arange(len(areas)) + 0.01, filename_areas, color_name)
    plot_coefs(areas, values, filename=filename_bar, color_name=color_name)


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
        flags_file, checkpoints_dir)
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