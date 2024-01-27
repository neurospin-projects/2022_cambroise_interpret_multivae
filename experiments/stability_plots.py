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
from multimodal_cohort.constants import short_clinical_names, short_roi_names
from multimodal_cohort.utils import extract_and_order_by
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from scipy.stats import pearsonr, combine_pvalues
from color_utils import (print_title, print_subtitle, print_text, print_result)
from daa_functions import (compute_significativity, compute_all_stability,
                           compute_all_associations, Heuristic, make_digital_avatars,
                           compute_daa_statistics, compute_all_stability_fast)
from workflow import score_models
from stability import (compute_stable_associations, select_stable_associations,
                       compute_associations_probs)


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

    associations, all_values = compute_stable_associations(dataset, datasetdir, outdir,
        best_heuristic_prior, compute_across_runs, additional_data,
        granularity, model_selection, n_subjects,
        sampling, sample_latents, ensemble_models=ensemble_models)
    df, coefs = select_stable_associations(associations, all_values, min_occurence)
   
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


def daa_plot_most_associated_stable(
    dataset, datasetdir, outdir, runs,
    granularity="metric_score", model_selection="no_selection",
    metrics=["thickness", "meancurv", "area"],
    mandatory_scores=[], min_occurence=25,
    n_rois=6, n_subjects=301,
    sampling="likelihood", sample_latents=False,
    ensemble_models=False, save_legend=False,
    any_n_models=False, only_significant=False):
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
    from color_utils import plt_to_plotly_rgb, get_color_list, rois_to_colors

    print_title(f"PLOT DAA most associated rois across scores with stability.")

    clinical_names = np.load(
        os.path.join(datasetdir, "clinical_names.npy"), allow_pickle=True)
    clinical_names = clinical_names.tolist()
    scores = clinical_names.copy()
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

    stabdir = os.path.join(outdir, runs[0], f"stability_with_{len(runs) - 1}_other_runs")
    local_stabdir = os.path.join(
        stabdir,
        (f"n_subjects_{n_subjects}_sampling_{sampling}_sample_latents_"
            f"{sample_latents}"))
    if ensemble_models:
        local_stabdir += "_ensemble_final_simplest"
    if not os.path.exists(local_stabdir):
        raise ValueError("You must compute the stability between runs before "
                         "ploting final results with validation runs.")
    with open(os.path.join(local_stabdir, "best_heuristic_prior"), 'rb') as f:
        best_heuristic_prior = pickle.load(f)

    ref_run = runs[0]
    other_runs = runs[1:]
    expdir = os.path.join(outdir, ref_run)
    resultsdir = os.path.join(expdir, f"results_with_{len(other_runs)}_other_runs")
    if not os.path.isdir(resultsdir):
        os.makedirs(resultsdir)

    if not any_n_models:
        associations, all_values = compute_stable_associations(dataset, datasetdir, outdir,
            best_heuristic_prior, runs, additional_data,
            granularity, model_selection, n_subjects,
            sampling, sample_latents, ensemble_models=ensemble_models)
        df, coefs = select_stable_associations(associations, all_values, min_occurence, additional_data)
    else:
        rois = np.array(
            list(dict.fromkeys([name.rsplit("_", 1)[0] for name in rois_names])))
        number_of_models = list(range(1, len(runs) + 1))
        all_associations_probs = np.empty((len(number_of_models), len(scores),
                                           len(metrics), len(rois)))
        for n_models_idx, n_models in enumerate(number_of_models):
            associations, all_values = compute_stable_associations(
                dataset, datasetdir, outdir,
                best_heuristic_prior, runs[:n_models], additional_data,
                granularity, model_selection, n_subjects,
                sampling, sample_latents, ensemble_models=True, fast=True)
            if n_models == len(number_of_models) - 1:
                coefs = np.mean(all_values, axis=0)
            associations = np.array(associations, dtype=int)
            for score_idx, score in enumerate(scores):
                for metric_idx, metric in enumerate(metrics):
                    associations_probs = compute_associations_probs(
                        associations[:, score_idx, metric_idx])
                    all_associations_probs[n_models_idx, score_idx, metric_idx] = (
                        associations_probs)
        stable_associations = (all_associations_probs >= min_occurence / 100)
        retained_associations = stable_associations.any(axis=0)
        listed_associations = {"metric": [], "roi": [], "score": []}
        for score_idx, score in enumerate(scores):
            for metric_idx, metric in enumerate(metrics):
                selected_rois = retained_associations[score_idx, metric_idx]
                for roi in rois[selected_rois]:
                    listed_associations["score"].append(score)
                    listed_associations["metric"].append(metric)
                    listed_associations["roi"].append(roi)
        df = pd.DataFrame(listed_associations)

    print_text(f"Number of associations : {df.shape[0]}")
    print_subtitle(f"Plot regression coefficients radar plots...")

    marker_signif = "star"
    marker_non_signif = "circle"
    counts = collections.Counter(df["roi"].values)
    # selected_rois = [item[0] for item in counts.most_common()]
    # n_colors = n_rois * len(df["metric"].unique())
    color_name = "tab20+tab20b+tab20c"
    roi_color_mapping = {
        "lh_cingul": ["tab20_0-2", "tab20_18-20"],
        "rh_cingul" : ["tab20_8-10", "tab20_12-14"],
        "lh_oc": ["tab20_2-4"],
        "rh_oc": ["tab20_7-8", "tab20_6-7"],
        "lh_temp": ["tab20_4-6"],
        "lh_perical": ["tab20b_2-3"],
        "rh_perical": ["tab20b_18-19"],
    }
    textfont = dict(
        size=24,
        family="Droid Serif")
    colors = get_color_list(color_name)
    all_selected_rois = []
    for _metric, _df in df.groupby(["metric"]):
        selected_scores = []
        significativity = []
        coefs_sign = []
        _df_to_count = _df.copy()
        if len(mandatory_scores) > 0:
            _df_to_count = _df[_df["score"].isin(mandatory_scores)]
        counts = collections.Counter(_df_to_count["roi"].values)
        # print(counts.most_common(n_rois+10))
        selected_rois = [item[0] for item in counts.most_common(len(list(counts))) if item[1] >= len(mandatory_scores)]#[:n_rois]
        n_rois = len(selected_rois)
        for _roi in selected_rois:
            roi_idx = rois_names.index(f"{_roi}_{_metric}")
            selected_coefs = coefs[:, roi_idx]
            selected_scores.append(np.absolute(selected_coefs))
            idx_sign = [_score in _df.loc[_df["roi"] == _roi, "score"].values for _score in scores]
            significativity.append(idx_sign)
            roi_coefs = (selected_coefs >= 0).astype(int) * 2 - 1
            if only_significant:
                significant_idx = [score_idx for score_idx, score in enumerate(scores)
                                  if score not in _df.loc[_df["roi"] == _roi, "score"].values]
                roi_coefs[significant_idx] = 0
            coefs_sign.append(roi_coefs)
        all_selected_rois += [area for area in selected_rois if area not in all_selected_rois]
        colors = rois_to_colors(all_selected_rois, roi_color_mapping)
        selected_scores = np.asarray(selected_scores)

        eps = 2
        marg = 2
        theta = (360 - n_rois * len(clinical_names) * eps - marg * (len(clinical_names) + 1))  / len(clinical_names)
        width = theta / n_rois
        fig = go.Figure()
        for roi_idx, _roi in enumerate(selected_rois):
            color_idx = all_selected_rois.index(_roi)
            color = plt_to_plotly_rgb(colors[color_idx])
            _scores = selected_scores[roi_idx].tolist()
            thetas = [marg + (n_rois * eps + theta + marg) * score_idx +
                      roi_idx * (width + eps) + width / 2
                      for score_idx in range(len(clinical_names))]
            fig.add_trace(go.Barpolar(
                r=coefs_sign[roi_idx].tolist(),
                theta=thetas,
                width=width,
                marker_color=color,
                marker_line_color="black",
                marker_line_width=2,
                opacity=0.8,
                legendgroup="roi",
                legendgrouptitle = dict(
                    font=dict(
                        size=textfont["size"] + 4,
                        family=textfont["family"]),
                    text="<b>ROIs</b>"),
                # showlegend=False,
                name=_roi,)
            )
        fig.update_layout(
            # template="simple_white",
            polar=dict(
                radialaxis=dict(
                    visible=True, showticklabels=False, ticks="outside",
                    dtick=[-1, 0, 1], range=[-1.55, 1.55], linecolor="grey",
                    tickcolor="red"),
                angularaxis=dict(
                    visible=True, showticklabels=False, ticks="",
                    dtick=(n_rois * eps + theta + marg), showline=False,
                    gridcolor="grey", gridwidth=2#, color="grey"
                )),
            showlegend=False,
            width=700,
            height=700,
            font=textfont,
            )
        label_textfont = textfont.copy()
        label_textfont["color"] = [
            "mediumvioletred" if score in mandatory_scores else"black" for score in (
                clinical_names + clinical_names[:1])]
        fig.add_trace(
            go.Scatterpolar(
                r=[1.5] * len(clinical_names),
                theta = [marg + (n_rois * eps + theta + marg) * score_idx
                            + (n_rois * eps + theta + marg) / 2
                            for score_idx in range(len(clinical_names))],
                text=["<b>" + short_clinical_names[dataset][name] + "</b>"
                    for name in clinical_names + clinical_names[:1]],
                mode="text",
                textfont=label_textfont,
                showlegend=False))
        plusminustexfont = textfont.copy()
        plusminustexfont["color"] = ["blue", "red"]
        plusminustexfont["size"] = 30
        fig.add_trace(
            go.Scatterpolar(
                r=[-1.2, 1.15],
                theta = [330, 356],
                text=["<b>-</b>", "<b>+</b>"],
                mode="text",
                textfont=plusminustexfont,
                showlegend=False))
        if not only_significant:
            for marker, sign_label, sign in [
                # (marker_non_signif, "non significative", False),
                (marker_signif, "significative", True)]:
                significative_scores = []
                score_names = []
                markers = []
                color = []
                thetas = []
                for roi_idx, roi_coefs in enumerate(selected_scores):
                    for coef_idx, coef in enumerate(roi_coefs):
                        if significativity[roi_idx][coef_idx] == sign:
                            significative_scores.append(coefs_sign[roi_idx][coef_idx])
                            score_names.append(clinical_names[coef_idx])
                            markers.append(marker)
                            thetas.append(
                                marg + (n_rois * eps + theta + marg) * coef_idx +
                                roi_idx * (width + eps) + width / 2)
                            # if coefs_sign[roi_idx][coef_idx] == 1:
                            #     color.append("red")
                            # else:
                            #     color.append("blue")
                            color.append("black")
                fig.add_trace(go.Scatterpolar(
                    r=np.array(significative_scores) / 2,
                    theta=thetas,
                    mode="markers",
                    legendgroup="significativity",
                    legendgrouptitle = dict(
                        font=dict(
                            size=textfont["size"] + 4,
                            family="Droid Serif"),
                        text="<b>Significativity</b>"),
                    showlegend=False,
                    marker_symbol=np.array(markers),
                    marker_size=10,
                    marker_color=color,
                    name=sign_label
                ))
        filename = os.path.join(
            resultsdir, f"most_associated_rois_{_metric}_polarplots.png")
        fig.write_image(filename)
        print_result(f"{_metric} association signs for 3 most associated "
                        f"ROIs: {filename}")
    filename = os.path.join(resultsdir, "most_associated_rois.png")
    colors = rois_to_colors(all_selected_rois, roi_color_mapping)
    plot_areas(all_selected_rois, colors, filename, save_legend=save_legend)


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
        flags_file, checkpoints_dir, datasetdir=datasetdir,
        outdir=outdir)
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

    associations, all_values = compute_stable_associations(dataset, datasetdir, outdir,
        best_heuristic_prior, compute_across_runs, additional_data,
        granularity, model_selection, n_subjects,
        sampling, sample_latents, ensemble_models=ensemble_models)
    df, coefs = select_stable_associations(associations, all_values, min_occurence)
   
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


def daa_plot_metric_score_coefs_sign_stable(dataset, datasetdir, outdir, runs,
                                            score, metric, granularity="metric_score",
                                            model_selection="no_selection",
                                            metrics=["thickness", "meancurv", "area"],
                                            min_occurence=5, n_subjects=301,
                                            sampling="likelihood",
                                            sample_latents=False,
                                            ensemble_models=False,
                                            any_n_models=False):
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
    from plotting import plot_areas, plot_coefs, plot_areas_signs
    import matplotlib.pyplot as plt

    print_title(f"PLOT DAA results: {dataset}")
    expdir = os.path.join(outdir, runs[0])

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

    stabdir = os.path.join(outdir, runs[0], f"stability_with_{len(runs) - 1}_other_runs")
    local_stabdir = os.path.join(
        stabdir,
        (f"n_subjects_{n_subjects}_sampling_{sampling}_sample_latents_"
            f"{sample_latents}"))
    if ensemble_models:
        local_stabdir += "_ensemble_final_simplest"
    if not os.path.exists(local_stabdir):
        raise ValueError("You must compute the stability between runs before "
                         "ploting final results with validation runs.")
    with open(os.path.join(local_stabdir, "best_heuristic_prior"), 'rb') as f:
        best_heuristic_prior = pickle.load(f)

    ref_run = runs[0]
    other_runs = runs[1:]
    expdir = os.path.join(outdir, ref_run)
    resultsdir = os.path.join(expdir, f"results_with_{len(other_runs)}_other_runs")
    if not os.path.isdir(resultsdir):
        os.makedirs(resultsdir)

    if not any_n_models:
        associations, all_values = compute_stable_associations(dataset, datasetdir, outdir,
            best_heuristic_prior, runs, additional_data,
            granularity, model_selection, n_subjects,
            sampling, sample_latents, ensemble_models=ensemble_models)
        df, coefs = select_stable_associations(associations, all_values, min_occurence, additional_data)
        areas = df["roi"][(df["metric"] == metric) & (df["score"] == score)].to_list()
    else:
        rois = np.array(
            list(dict.fromkeys([name.rsplit("_", 1)[0] for name in rois_names])))
        number_of_models = list(range(1, len(runs) + 1))
        all_associations_probs = np.empty((len(number_of_models), len(rois)))
        for n_models_idx, n_models in enumerate(number_of_models):
            associations, all_values = compute_stable_associations(
                dataset, datasetdir, outdir,
                best_heuristic_prior, runs[:n_models], additional_data,
                granularity, model_selection, n_subjects,
                sampling, sample_latents, ensemble_models=True, fast=True)
            if n_models == len(number_of_models) - 1:
                coefs = np.mean(all_values, axis=0)
            associations = np.array(associations, dtype=int)
            associations_probs = compute_associations_probs(
                associations[:, 0, 0])
            all_associations_probs[n_models_idx] = associations_probs
        stable_associations = (all_associations_probs >= min_occurence / 100)
        retained_associations = stable_associations.any(axis=0)
        areas = rois[retained_associations]
   
    marker_signif = "star"
    marker_non_signif = "circle"

    area_idx = [rois_names.index(f"{name}_{metric}") for name in areas]
    score_idx = clinical_names.index(score)
    values = coefs[score_idx, area_idx]
    signs = (values >= 0).astype(int) 
    print(f"Number of significative rois in {metric} for {score} : {len(areas)}")
    filename_areas = os.path.join(
        resultsdir, f"associated_rois_for_{score}_in_{metric}_with_sign.png")
    plt.rcParams.update({'font.size': 20, "font.family": "serif"})
    plot_areas_signs(areas, signs + 0.01, filename_areas)


def plot_metric_score_stability_against_n_models(
    dataset, datasetdir, outdir, runs, score, metric,
    granularity="overall",
    model_selection="no_selection",
    use_custom_heuristic=False,
    heuristic={"coefs_mean":{"strategy":"num", "num":12}},
    min_stability=0.25, n_subjects=301, sampling="likelihood",
    sample_latents=False, emph_rois=[]):
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
    from color_utils import get_color_list

    print_title(f"PLOT DAA results: {dataset}")

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
    metrics = ["thickness", "meancurv", "area"]
    rois_names = rois_names.tolist()

    stabdir = os.path.join(outdir, runs[0], f"stability_with_{len(runs) - 1}_other_runs")
    local_stabdir = os.path.join(
        stabdir,
        (f"n_subjects_{n_subjects}_sampling_{sampling}_sample_latents_"
            f"{sample_latents}_ensemble_final_simplest"))
    print(local_stabdir)
    if not os.path.exists(local_stabdir) or use_custom_heuristic:
        heuristic_name, heuristic_params = list(heuristic.items())[0]
        strategy = heuristic_params["strategy"]
        value = heuristic_params[strategy]
        if granularity == "overall":
            best_heuristic_prior = {model_selection: {
            granularity:(None, None, None, None, None,
                                    heuristic_name,
                                    f"strategy_{strategy}_value_{value}")}}
        else:
            if granularity == "metric":
                granularity_label = metric
            elif granularity == "score":
                granularity_label = score
            else:
                granularity_label = f"{metric}_{score}"
            best_heuristic_prior = {model_selection: {
                granularity:{
                    granularity_label:(None, None, None, None, None,
                                        heuristic_name,
                                        f"strategy_{strategy}_value_{value}")}}}
    else:
        print("Reading best ensembling functions")
        with open(os.path.join(local_stabdir, "best_heuristic_prior"), 'rb') as f:
            best_heuristic_prior = pickle.load(f)

    ref_run = runs[0]
    other_runs = runs[1:]
    expdir = os.path.join(outdir, ref_run)
    resultsdir = os.path.join(expdir, f"results_with_{len(other_runs)}_other_runs")
    if not os.path.isdir(resultsdir):
        os.makedirs(resultsdir)
    
    number_of_models = list(range(1, len(runs) + 1))
    all_associations_probs = []
    for n_models in number_of_models:
        associations, all_values = compute_stable_associations(
            dataset, datasetdir, outdir,
            best_heuristic_prior, runs[:n_models], additional_data,
            granularity, model_selection, n_subjects,
            sampling, sample_latents, ensemble_models=True, fast=True)
        associations = np.array(associations, dtype=int)
        associations_probs = compute_associations_probs(
            associations[:, 0, 0])
        all_associations_probs.append((associations_probs))
    
    rois = np.array(
        list(dict.fromkeys([name.rsplit("_", 1)[0] for name in rois_names])))
    all_associations_probs = np.array(all_associations_probs)
    print(all_associations_probs.shape)
    colors = get_color_list(color_name="tab20+tab20b+tab20c")
    color_idx = 0
    plt.rcParams.update({"font.size": 20, "font.family": "serif",
                          "mathtext.fontset": "stix",
                          "font.family" :"STIXGeneral"})
    fig_width = 10
    plt.figure(figsize=(fig_width, 3/4 * fig_width))
    for roi_idx in range(all_associations_probs.shape[1]):
        color = "black"
        ls = "dashed"
        linewidth = 1
        if ((len(emph_rois) == 0 and 
             (all_associations_probs[:, roi_idx] > min_stability).any())
            or rois[roi_idx] in emph_rois):
            # color = "red"
            color = colors[color_idx]
            color_idx += 1
            ls = "solid"
            linewidth = 2
            print(rois[roi_idx])
        plt.plot(number_of_models, all_associations_probs[:, roi_idx],
                 color=color, ls=ls, linewidth=linewidth)
    if len(emph_rois) == 0:
        plt.hlines(y=min_stability, xmin=1, xmax=20, color="red", label="$\pi_{thr}$", ls="dashed")
        # plt.legend()
    plt.xticks(range(2, number_of_models[-1] + 1, 2))
    plt.xlabel("$n_E$", size=32)
    plt.ylabel("$\Pi$", rotation="horizontal", labelpad=10, size=32)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.tight_layout()
    plt.show()

def plot_metric_score_coefs_against_n_models(
    dataset, datasetdir, outdir, runs, score, metric,
    aggregating_function="mean", emph_rois=[],
    n_subjects=301, sampling="likelihood",
    sample_latents=False):
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
    from color_utils import get_color_list

    print_title(f"PLOT DAA results: {dataset}")

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
    metrics = ["thickness", "meancurv", "area"]
    rois_names = rois_names.tolist()

    ref_run = runs[0]
    other_runs = runs[1:]
    expdir = os.path.join(outdir, ref_run)
    resultsdir = os.path.join(expdir, f"results_with_{(len(runs) - 1)}_other_runs")
    if not os.path.isdir(resultsdir):
        os.makedirs(resultsdir)
    run_coefs = []
    run_pvalues = []
    run_model_scores = []
    for run in runs:
        simdirs = [path for path in glob.glob(os.path.join(
            outdir, run, "daa", "*")) if os.path.isdir(path)]
        for dirname in simdirs:
            # print_text(dirname)
            if not os.path.exists(os.path.join(dirname, "coefs.npy")):
                continue
            coefs = np.load(os.path.join(dirname, "coefs.npy"))
            pvalues = np.load(os.path.join(dirname, "pvalues.npy"))

            local_sampling = dirname.split("sampling_")[1].split("_sample")[0]
            local_sample_latents = dirname.split("latents_")[1].split("_seed")[0]
            local_n_subjects = int(dirname.split("subjects_")[1].split("_M")[0])

            if (local_n_subjects != n_subjects or
                (sampling is not None and local_sampling != sampling) or
                (sample_latents is not None and
                local_sample_latents != str(sample_latents))):
                continue
            model_scores = score_models(dataset, datasetdir, outdir, run,
                                        scores=clinical_names)
            run_coefs.append(coefs)
            run_pvalues.append(pvalues)
            run_model_scores.append(model_scores)

    number_of_models = list(range(1, len(runs) + 1))
    rois = np.array(
        list(dict.fromkeys([name.rsplit("_", 1)[0] for name in rois_names])))
    
    final_results = []
    coefs = np.array(run_coefs)
    pvalues = np.array(run_pvalues)
    model_scores = np.array(run_model_scores)

    score_idx = clinical_names.index(score)
    split_idx = np.random.randint(coefs.shape[1])
    all_coefs_per_n_models = []
    for n_models in number_of_models:
        params = {"strategy": "num", "num": 12}
        heuristic_param = {f"coefs_{aggregating_function}": params}
        heuristic = Heuristic(heuristic_param, additional_data, True)
        _, agg_coefs = heuristic(coefs[:n_models,split_idx],
                                    pvalues[:n_models,split_idx],
                                    model_scores[:n_models,split_idx], True)
        all_coefs_per_n_models.append(agg_coefs[score_idx])
    
    colors = get_color_list(color_name="tab20+tab20b+tab20c")
    all_coefs_per_n_models = np.array(all_coefs_per_n_models)
    plt.rcParams.update({"font.size": 20, "font.family": "serif",
                          "mathtext.fontset": "stix",
                          "font.family" :"STIXGeneral"})
    fig_width = 10
    plt.figure(figsize=(fig_width, 3/4 * fig_width))
    metric_indices = []
    for roi_idx in range(all_coefs_per_n_models.shape[1]):
        if metric in rois_names[roi_idx]:
            roi_name = rois_names[roi_idx].rsplit("_", 1)[0]
            metric_indices.append(roi_idx)
            color = "black"
            ls = "dashed"
            linewidth = 1
            zorder = 5
            if roi_name in emph_rois:
                # color = "red"
                color_idx = emph_rois.index(roi_name)
                color = colors[color_idx]
                ls = "solid"
                linewidth = 2
                zorder = 10
            plt.plot(number_of_models, all_coefs_per_n_models[:, roi_idx],
                     color=color, ls=ls, linewidth=linewidth, zorder=zorder)
    plt.xticks(list(range(2, number_of_models[-1] + 1, 2)))
    min_y = np.round(all_coefs_per_n_models[:, metric_indices].min() * 100)
    max_y = np.ceil(all_coefs_per_n_models[:, metric_indices].max() * 100)
    plt.yticks(np.arange(min_y, max_y + 1) * 1e-2)
    plt.xlabel("$n_E$", size=30)
    plt.ylabel(r"$\beta$", rotation="horizontal", labelpad=10, size=30)
    plt.tight_layout()
    #plt.ticklabel_format(axis='y', style='sci')
    plt.show()
   
def generate_latex_associations_table(
    dataset, datasetdir, outdir, runs,
    metrics=["thickness", "meancurv", "area"],
    granularity="overall",
    model_selection="no_selection",
    use_custom_heuristic=False,
    heuristic={"coefs_mean":{"strategy":"num", "num":12}},
    min_stability=0.40, n_subjects=301, sampling="likelihood",
    sample_latents=False):
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
    from plotting import short_roi_name
    import matplotlib.pyplot as plt
    from color_utils import get_color_list

    print_title(f"PLOT DAA results: {dataset}")

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
                                      scores=clinical_names, metrics=metrics)
    rois_names = rois_names.tolist()

    stabdir = os.path.join(outdir, runs[0], f"stability_with_{len(runs) - 1}_other_runs")
    local_stabdir = os.path.join(
        stabdir,
        (f"n_subjects_{n_subjects}_sampling_{sampling}_sample_latents_"
            f"{sample_latents}_ensemble_final_simplest"))
    print(local_stabdir)
    if not os.path.exists(local_stabdir) or use_custom_heuristic:
        heuristic_name, heuristic_params = list(heuristic.items())[0]
        strategy = heuristic_params["strategy"]
        value = heuristic_params[strategy]
        if granularity == "overall":
            best_heuristic_prior = {model_selection: {
            granularity:(None, None, None, None, None,
                                    heuristic_name,
                                    f"strategy_{strategy}_value_{value}")}}
        else:
            if granularity == "metric":
                granularity_label = metric
            elif granularity == "score":
                granularity_label = score
            else:
                granularity_label = f"{metric}_{score}"
            best_heuristic_prior = {model_selection: {
                granularity:{
                    granularity_label:(None, None, None, None, None,
                                        heuristic_name,
                                        f"strategy_{strategy}_value_{value}")}}}
    else:
        print("Reading best ensembling functions")
        with open(os.path.join(local_stabdir, "best_heuristic_prior"), 'rb') as f:
            best_heuristic_prior = pickle.load(f)

    ref_run = runs[0]
    other_runs = runs[1:]
    expdir = os.path.join(outdir, ref_run)
    resultsdir = os.path.join(expdir, f"results_with_{len(other_runs)}_other_runs")
    if not os.path.isdir(resultsdir):
        os.makedirs(resultsdir)
    rois = np.array(
        list(dict.fromkeys([name.rsplit("_", 1)[0] for name in rois_names])))
    number_of_models = list(range(1, len(runs) + 1))
    all_associations_probs = np.empty(
        (len(number_of_models), len(metrics), len(clinical_names), len(rois)))
    for n_models_idx, n_models in enumerate(number_of_models):
        associations, all_values = compute_stable_associations(
            dataset, datasetdir, outdir,
            best_heuristic_prior, runs[:n_models], additional_data,
            granularity, model_selection, n_subjects,
            sampling, sample_latents, ensemble_models=True, fast=True)
        if n_models == len(number_of_models) - 1:
            all_associations_coefs = np.mean(all_values, axis=0)
        associations = np.array(associations, dtype=int)
        for metric_idx, metric in enumerate(metrics):
            for score_idx, score in enumerate(clinical_names):
                associations_probs = compute_associations_probs(
                    associations[:, score_idx, metric_idx])
                all_associations_probs[n_models_idx, metric_idx, score_idx] = (
                    associations_probs)
    stable_associations = (all_associations_probs >= min_stability)
    retained_associations = stable_associations.any(axis=0)
    print(retained_associations.shape)
    scores = ["SRS_Total", "SCARED_P_Total", "ARI_P_Total_Score", "SDQ_Hyperactivity"]
    max_rois_per_metric = retained_associations.sum(axis=-1).max(-1)
    print(max_rois_per_metric)

    table = "\\begin{longtable}{p{2cm}|c c c c c c c c}\n\\toprule\n"
    table += "\multicolumn{1}{r|}{\\textbf{Score}}"
    for score in scores:
        table += " & \\textbf{" + short_clinical_names[dataset][score] + "}"
    table += "\\\\\n"
    for score_idx, _ in enumerate(scores):
        table += "\cmidrule(lr){" + str(score_idx + 2) + "-" + str(score_idx + 2) + "}"
    table += "\n\\textbf{Metric}\\\\\n"
    table += "\\toprule\n"

    for metric_idx, metric in enumerate(metrics):
        table += "\\textbf{" + metric.title() + "}"
        number_of_rois = max_rois_per_metric[metric_idx]
        for roi_idx in range(number_of_rois):
            for score_idx, score in enumerate(scores):
                score_index = clinical_names.index(score)
                selected_rois = retained_associations[metric_idx, score_index]
                selected_coefs = all_associations_coefs[score_index]
                if roi_idx < selected_rois.sum():
                    roi_name = rois[selected_rois][roi_idx]
                    roi_index = rois_names.index(roi_name + "_" + metric)
                    coef = selected_coefs[roi_index]
                    color = "red" if coef >= 0 else "blue"
                    table += " & \color{" + color + "}" + short_roi_name(roi_name)
                else:
                    table += " & "
            table += "\\\\\n"
        if metric_idx < len(metrics) - 1:
            table += "\midrule\n"
    scores = ["CBCL_AB", "CBCL_AP", "CBCL_WD"]
    table += "\\bottomrule\n"
    table += "\\toprule\n"
    table += "\multicolumn{1}{r|}{\\textbf{Score}}"
    for score in scores:
        table += " & \\textbf{" + short_clinical_names[dataset][score] + "}"
    table += "\\\\\n"
    for score_idx, _ in enumerate(scores):
        table += "\cmidrule(lr){" + str(score_idx + 2) + "-" + str(score_idx + 2) + "}"
    table += "\n\\textbf{Metric}\\\\\n"
    table += "\\toprule\n"
    for metric_idx, metric in enumerate(metrics):
        table += "\\textbf{" + metric.title() + "}"
        number_of_rois = max_rois_per_metric[metric_idx]
        for roi_idx in range(number_of_rois):
            for score_idx, score in enumerate(scores):
                score_index = clinical_names.index(score)
                selected_rois = retained_associations[metric_idx, score_index]
                selected_coefs = all_associations_coefs[score_index]
                if roi_idx < selected_rois.sum():
                    roi_name = rois[selected_rois][roi_idx]
                    roi_index = rois_names.index(roi_name + "_" + metric)
                    coef = selected_coefs[roi_index]
                    color = "red" if coef >= 0 else "blue"
                    table += " & \color{" + color + "}" + short_roi_name(roi_name)
                else:
                    table += " & "
            table += "\\\\\n"
        if metric_idx < len(metrics) - 1:
            table += "\midrule\n"
    table += "\\bottomrule\n"
    table +=  "\caption{Retained associations for each score and metric. {\color{blue}Blue} indicates a negative association and {\color{red}red} denote a positive one. L: left, R: right, S: sulcus, G: gyrus, Lat: lateral"
    already_legended = []
    for origin, replace in short_roi_names.items():
        if replace not in already_legended and replace != "":
            table += f", {replace}: {origin}"
            already_legended.append(replace)
    table += ".}\n\label{tab:associations}\n\end{longtable}"
    print(table)

def generate_csv_associations_table(
    dataset, datasetdir, outdir, runs,
    metrics=["thickness", "meancurv", "area"],
    granularity="overall",
    model_selection="no_selection",
    use_custom_heuristic=False,
    heuristic={"coefs_mean":{"strategy":"num", "num":12}},
    min_stability=0.40, n_subjects=301, sampling="likelihood",
    sample_latents=False, excel=False):
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
    from plotting import short_roi_name
    import matplotlib.pyplot as plt
    from color_utils import get_color_list

    print_title(f"PLOT DAA results: {dataset}")

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
                                      scores=clinical_names, metrics=metrics)
    rois_names = rois_names.tolist()

    stabdir = os.path.join(outdir, runs[0], f"stability_with_{len(runs) - 1}_other_runs")
    local_stabdir = os.path.join(
        stabdir,
        (f"n_subjects_{n_subjects}_sampling_{sampling}_sample_latents_"
            f"{sample_latents}_ensemble_final_simplest"))
    print(local_stabdir)
    if not os.path.exists(local_stabdir) or use_custom_heuristic:
        heuristic_name, heuristic_params = list(heuristic.items())[0]
        strategy = heuristic_params["strategy"]
        value = heuristic_params[strategy]
        if granularity == "overall":
            best_heuristic_prior = {model_selection: {
            granularity:(None, None, None, None, None,
                                    heuristic_name,
                                    f"strategy_{strategy}_value_{value}")}}
        else:
            if granularity == "metric":
                granularity_label = metric
            elif granularity == "score":
                granularity_label = score
            else:
                granularity_label = f"{metric}_{score}"
            best_heuristic_prior = {model_selection: {
                granularity:{
                    granularity_label:(None, None, None, None, None,
                                        heuristic_name,
                                        f"strategy_{strategy}_value_{value}")}}}
    else:
        print("Reading best ensembling functions")
        with open(os.path.join(local_stabdir, "best_heuristic_prior"), 'rb') as f:
            best_heuristic_prior = pickle.load(f)

    ref_run = runs[0]
    other_runs = runs[1:]
    expdir = os.path.join(outdir, ref_run)
    resultsdir = os.path.join(expdir, f"results_with_{len(other_runs)}_other_runs")
    if not os.path.isdir(resultsdir):
        os.makedirs(resultsdir)
    rois = np.array(
        list(dict.fromkeys([name.rsplit("_", 1)[0] for name in rois_names])))
    number_of_models = list(range(1, len(runs) + 1))
    all_associations_probs = np.empty(
        (len(number_of_models), len(metrics), len(clinical_names), len(rois)))
    for n_models_idx, n_models in enumerate(number_of_models):
        associations, all_values = compute_stable_associations(
            dataset, datasetdir, outdir,
            best_heuristic_prior, runs[:n_models], additional_data,
            granularity, model_selection, n_subjects,
            sampling, sample_latents, ensemble_models=True, fast=True)
        if n_models == len(number_of_models) - 1:
            all_associations_coefs = np.mean(all_values, axis=0)
        associations = np.array(associations, dtype=int)
        for metric_idx, metric in enumerate(metrics):
            for score_idx, score in enumerate(clinical_names):
                associations_probs = compute_associations_probs(
                    associations[:, score_idx, metric_idx])
                all_associations_probs[n_models_idx, metric_idx, score_idx] = (
                    associations_probs)
    stable_associations = (all_associations_probs >= min_stability)
    retained_associations = stable_associations.any(axis=0)
    print(retained_associations.shape)
    scores = ["SRS_Total", "SCARED_P_Total", "ARI_P_Total_Score", "SDQ_Hyperactivity"]
    max_rois_per_metric = retained_associations.sum(axis=-1).max(-1)
    print(max_rois_per_metric)

    table = "\\begin{longtable}{p{2cm}|c c c c c c c c}\n\\toprule\n"
    table += "\multicolumn{1}{r|}{\\textbf{Score}}"
    for score in scores:
        table += " & \\textbf{" + short_clinical_names[dataset][score] + "}"
    table += "\\\\\n"
    for score_idx, _ in enumerate(scores):
        table += "\cmidrule(lr){" + str(score_idx + 2) + "-" + str(score_idx + 2) + "}"
    table += "\n\\textbf{Metric}\\\\\n"
    table += "\\toprule\n"

    for metric_idx, metric in enumerate(metrics):
        table += "\\textbf{" + metric.title() + "}"
        number_of_rois = max_rois_per_metric[metric_idx]
        for roi_idx in range(number_of_rois):
            for score_idx, score in enumerate(scores):
                score_index = clinical_names.index(score)
                selected_rois = retained_associations[metric_idx, score_index]
                selected_coefs = all_associations_coefs[score_index]
                if roi_idx < selected_rois.sum():
                    roi_name = rois[selected_rois][roi_idx]
                    roi_index = rois_names.index(roi_name + "_" + metric)
                    coef = selected_coefs[roi_index]
                    color = "red" if coef >= 0 else "blue"
                    table += " & \color{" + color + "}" + short_roi_name(roi_name)
                else:
                    table += " & "
            table += "\\\\\n"
        if metric_idx < len(metrics) - 1:
            table += "\midrule\n"
    scores = ["CBCL_AB", "CBCL_AP", "CBCL_WD"]
    table += "\\bottomrule\n"
    table += "\\toprule\n"
    table += "\multicolumn{1}{r|}{\\textbf{Score}}"
    for score in scores:
        table += " & \\textbf{" + short_clinical_names[dataset][score] + "}"
    table += "\\\\\n"
    for score_idx, _ in enumerate(scores):
        table += "\cmidrule(lr){" + str(score_idx + 2) + "-" + str(score_idx + 2) + "}"
    table += "\n\\textbf{Metric}\\\\\n"
    table += "\\toprule\n"
    for metric_idx, metric in enumerate(metrics):
        table += "\\textbf{" + metric.title() + "}"
        number_of_rois = max_rois_per_metric[metric_idx]
        for roi_idx in range(number_of_rois):
            for score_idx, score in enumerate(scores):
                score_index = clinical_names.index(score)
                selected_rois = retained_associations[metric_idx, score_index]
                selected_coefs = all_associations_coefs[score_index]
                if roi_idx < selected_rois.sum():
                    roi_name = rois[selected_rois][roi_idx]
                    roi_index = rois_names.index(roi_name + "_" + metric)
                    coef = selected_coefs[roi_index]
                    color = "red" if coef >= 0 else "blue"
                    table += " & \color{" + color + "}" + short_roi_name(roi_name)
                else:
                    table += " & "
            table += "\\\\\n"
        if metric_idx < len(metrics) - 1:
            table += "\midrule\n"
    table += "\\bottomrule\n"
    table +=  "\caption{Retained associations for each score and metric. {\color{blue}Blue} indicates a negative association and {\color{red}red} denote a positive one. L: left, R: right, S: sulcus, G: gyrus, Lat: lateral"
    already_legended = []
    for origin, replace in short_roi_names.items():
        if replace not in already_legended and replace != "":
            table += f", {replace}: {origin}"
            already_legended.append(replace)
    table += ".}\n\label{tab:associations}\n\end{longtable}"
    print(table)