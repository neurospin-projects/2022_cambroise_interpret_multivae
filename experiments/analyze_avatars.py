import os
import glob
import itertools
import torch
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, lines
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from multimodal_cohort.experiment import MultimodalExperiment
from multimodal_cohort.constants import short_clinical_names
from multimodal_cohort.utils import extract_and_order_by
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from scipy.stats import pearsonr, combine_pvalues

from color_utils import (print_title, print_subtitle, print_text)
from daa_functions import compute_significativity, compute_all_stability
from workflow import score_models


def analyze_avatars(dataset, datasetdir, outdir, run, n_validation=5,
                    n_samples=200, n_subjects=50,
                    M=1000, reg_method="hierarchical",
                    sampling_strategy="likelihood", sample_latents=True,
                    val_step=0, seed=1037, n_subjects_to_plot=5):
    # Files paths
    rois_file = os.path.join(datasetdir, "rois_data.npy")
    rois_subjects_file = os.path.join(datasetdir, "rois_subjects.npy")
    rois_names_file = os.path.join(datasetdir, "rois_names.npy")

    clinical_file = os.path.join(datasetdir, "clinical_data.npy")
    clinical_subjects_file = os.path.join(datasetdir, "clinical_subjects.npy")
    clinical_names_file = os.path.join(datasetdir, "clinical_names.npy")

    flags_file = os.path.join(outdir, run, "flags.rar")
    if not os.path.isfile(flags_file):
        raise ValueError("You need first to train the model.")    
    checkpoints_dir = os.path.join(outdir, run, "checkpoints")
    experiment, flags = MultimodalExperiment.get_experiment(
        flags_file, checkpoints_dir)

    params = SimpleNamespace(
        n_validation=n_validation, n_subjects=n_subjects, M=M,
        n_samples=n_samples, reg_method=reg_method,
        sampling=sampling_strategy, sample_latents=sample_latents)
    name = "_".join(["_".join([key, str(val)])
                     for key, val in params.__dict__.items()])
    resdir = os.path.join(outdir, run, "daa", name)

    da_file = os.path.join(resdir, "rois_digital_avatars.npy")
    sampled_scores_file = os.path.join(resdir, "sampled_scores.npy")
    metadata_file = os.path.join(resdir, "metadatas.npy")

    # Files loading
    rois_data = np.load(rois_file, mmap_mode="r")
    rois_subjects = np.load(rois_subjects_file, allow_pickle=True)
    rois_names = np.load(rois_names_file, allow_pickle=True)

    clinical_data = np.load(clinical_file, mmap_mode="r")
    clinical_subjects = np.load(clinical_subjects_file, allow_pickle=True)
    clinical_names = np.load(clinical_names_file, allow_pickle=True)

    da = np.load(da_file, mmap_mode="r")
    scores = np.load(sampled_scores_file)
    metadata = np.load(metadata_file, allow_pickle=True)

    da = da[val_step]
    scores = scores[val_step]
    metadata = metadata[val_step]
    print(da.shape)
    print(scores.shape)
    print(metadata.shape)

    np.random.seed(seed)
    subj_indices = np.random.randint(n_subjects, size=n_subjects_to_plot)

    # scaled_clinical_values = experiment.scalers["clinical"].inverse_transform(clinical_data)
    plt.rcParams.update({'font.size': 20, "font.family": "serif"})
    for score_idx in range(len(clinical_names)):
        plt.figure()
        for idx, subj_idx in enumerate(subj_indices):
            sampled_scores = scores[subj_idx]
            true_sampled_scores = experiment.scalers["clinical"].inverse_transform(sampled_scores)[:, score_idx]
            sns.kdeplot(true_sampled_scores, color=list(colors.TABLEAU_COLORS)[idx])
            
            clinical_subj_idx = clinical_subjects.tolist().index(metadata[subj_idx, 0])
            plt.axvline(clinical_data[clinical_subj_idx, score_idx], color=list(colors.TABLEAU_COLORS)[idx])
            plt.title(short_clinical_names[dataset][clinical_names[score_idx]])
            plt.tight_layout()

    
    selected_scores = ["SRS_Total", "CBCL_AP", "SDQ_Hyperactivity", "ARI_P_Total_Score"]
    selected_scores = [clinical_names.tolist().index(score) for score in selected_scores]
    selected_rois = np.random.randint(len(rois_names), size=3)
    fig, axes = plt.subplots(len(selected_scores), len(selected_rois), sharey=True, figsize=(5 * len(selected_rois), 3 * len(selected_scores)))
    for idx, score_idx in enumerate(selected_scores):
        for roi_num, roi_idx in enumerate(selected_rois):
            # if score_idx == selected_scores[0]:
            axes[idx, roi_num].scatter(scores[subj_indices, :, score_idx].flatten(),
                                       da[subj_indices, score_idx, :, roi_idx].flatten(),
                                       c=np.repeat(np.arange(n_subjects_to_plot)[:, np.newaxis], n_samples, axis=1).flatten())
            if idx == 0:
                axes[idx, roi_num].set_title(rois_names[roi_idx])
                # axes[roi_idx].set_title(roi_name)
            if roi_num == 0:
                axes[idx, roi_num].set_ylabel(short_clinical_names[dataset][clinical_names[score_idx]])
    plt.tight_layout()
    plt.show()


def assess_robustness(dataset, datasetdir, outdir, run, n_validation=5,
                     n_samples=200, n_subjects=50,
                     M=1000, reg_method="hierarchical",
                     sampling_strategy="likelihood", sample_latents=True,
                     seed=1037, n_models_to_plot=5):
    
    clinical_file = os.path.join(datasetdir, "clinical_names.npy")
    rois_file = os.path.join(datasetdir, "rois_names.npy")

    params = SimpleNamespace(
        n_validation=n_validation, n_subjects=n_subjects, M=M,
        n_samples=n_samples, reg_method=reg_method,
        sampling=sampling_strategy, sample_latents=sample_latents,
        seed=seed)

    name = "_".join(["_".join([key, str(val)])
                     for key, val in params.__dict__.items()])
    resdir = os.path.join(outdir, run, "daa", name)

    clinical_names = np.load(clinical_file, allow_pickle=True)
    rois_names = np.load(rois_file, allow_pickle=True)
    flags = torch.load(os.path.join(outdir, run, "flags.rar"))
    pvalues = np.load(os.path.join(resdir, "pvalues.npy"))

    n_rois = len(rois_names)
    n_scores = len(clinical_names)
    significativity_thr = (0.05 / n_rois / n_scores)

    if flags.num_models == 1:
        pvalues = pvalues[np.newaxis]
    for model_idx in range(flags.num_models)[:n_models_to_plot]:
        trust_levels = np.arange(0, 1.01, 0.05)
        assoc_counts = {"score": [], "metric": [], "trust_level": [], "num_assoc":[]}
        for trust_level in trust_levels:
            local_trust_level = params.n_validation * trust_level
            idx_sign = (
                (pvalues[model_idx] < significativity_thr).sum(
                    axis=0) >= local_trust_level)

            data = {"metric": [], "roi": [], "score": []}
            for idx, score in enumerate(clinical_names):
                rois_idx = np.where(idx_sign[idx])
                for name in rois_names[rois_idx]:
                    name, metric = name.rsplit("_", 1)
                    data["score"].append(score)
                    data["metric"].append(metric)
                    data["roi"].append(name)
            df = pd.DataFrame.from_dict(data)
            counts = df.groupby(["score", "metric"]).count()
            for (score, metric), count in counts["roi"].items():
                assoc_counts["score"].append(score)
                assoc_counts["metric"].append(metric)
                assoc_counts["trust_level"].append(trust_level)
                assoc_counts["num_assoc"].append(count)
        assoc_counts = pd.DataFrame(assoc_counts).sort_values("trust_level")
        
        fig, axes = plt.subplots(2, 4)
        for score_idx, score in enumerate(clinical_names):
            ax = axes[score_idx // 4, score_idx % 4]
            for metric, counts in assoc_counts[
                assoc_counts["score"] == score].groupby("metric"):
                ax.plot(trust_levels[:len(counts["num_assoc"].values)],
                        counts["num_assoc"].values,
                        label=metric)
            ax.set_title(score)
            if score_idx == len(clinical_names) - 1:
                ax.legend()
        fig.tight_layout()
    
    for vote_prop in np.linspace(0.5, 1, min(n_models_to_plot, flags.num_models)):#range(flags.num_models)[:n_models_to_plot]:
        trust_levels = np.arange(0, 1.01, 0.05)
        assoc_counts = {"score": [], "metric": [], "trust_level": [], "num_assoc":[]}
        for trust_level in trust_levels:
            local_trust_level = params.n_validation * trust_level
            idx_sign = ((
                (pvalues < significativity_thr).sum(
                    axis=1) >= local_trust_level).sum(0) >= 
                        vote_prop * flags.num_models)

            data = {"metric": [], "roi": [], "score": []}
            for idx, score in enumerate(clinical_names):
                rois_idx = np.where(idx_sign[idx])
                for name in rois_names[rois_idx]:
                    name, metric = name.rsplit("_", 1)
                    data["score"].append(score)
                    data["metric"].append(metric)
                    data["roi"].append(name)
            df = pd.DataFrame.from_dict(data)
            counts = df.groupby(["score", "metric"]).count()
            for (score, metric), count in counts["roi"].items():
                assoc_counts["score"].append(score)
                assoc_counts["metric"].append(metric)
                assoc_counts["trust_level"].append(trust_level)
                assoc_counts["num_assoc"].append(count)
        assoc_counts = pd.DataFrame(assoc_counts).sort_values("trust_level")
        
        fig, axes = plt.subplots(2, 4)
        for score_idx, score in enumerate(clinical_names):
            ax = axes[score_idx // 4, score_idx % 4]
            for metric, counts in assoc_counts[
                assoc_counts["score"] == score].groupby("metric"):
                ax.plot(trust_levels[:len(counts["num_assoc"].values)],
                        counts["num_assoc"].values,
                        label=metric)
            ax.set_title(score)
            if score_idx == len(clinical_names) - 1:
                ax.legend()
        fig.tight_layout()
    plt.show()

def univariate_tests(dataset, datasetdir, continuous_covs=[],
                     categorical_covs=[], seed=1037):

    from plotting import plot_areas, plot_coefs
    if type(continuous_covs) is not list:
        continuous_covs = [continuous_covs]
    if type(categorical_covs) is not list:
        categorical_covs = [categorical_covs]

    rois_file = os.path.join(datasetdir, "rois_data.npy")
    rois_subjects_file = os.path.join(datasetdir, "rois_subjects.npy")
    rois_names_file = os.path.join(datasetdir, "rois_names.npy")

    clinical_file = os.path.join(datasetdir, "clinical_data.npy")
    clinical_subjects_file = os.path.join(datasetdir, "clinical_subjects.npy")
    clinical_names_file = os.path.join(datasetdir, "clinical_names.npy")

    metadata_file = os.path.join(datasetdir, "metadata.tsv")

    rois_data = np.load(rois_file, mmap_mode="r")
    rois_subjects = np.load(rois_subjects_file, allow_pickle=True)
    rois_names = np.load(rois_names_file, allow_pickle=True)

    clinical_data = np.load(clinical_file, mmap_mode="r")
    clinical_subjects = np.load(clinical_subjects_file, allow_pickle=True)
    clinical_names = np.load(clinical_names_file, allow_pickle=True)

    metadata = pd.read_table(metadata_file)

    subjects = set(clinical_subjects.tolist()).intersection(rois_subjects)

    subjects = sorted(list(subjects))
    rois_idx = [rois_subjects.tolist().index(subject) for subject in subjects]
    clinical_idx = [clinical_subjects.tolist().index(subject) for subject in subjects]


    rois_data = StandardScaler().fit_transform(rois_data[rois_idx])
    clinical_data = StandardScaler().fit_transform(clinical_data[clinical_idx])

    metadata = extract_and_order_by(metadata, "participant_id", subjects)

    n_rois = len(rois_names)
    n_scores = len(clinical_names)
    significativity_thr = (0.05 / n_rois / n_scores)

    associations = np.zeros((n_scores, n_rois))
    pvalues = np.zeros((n_scores, n_rois))
    base_df = metadata.copy()
    for score_idx, score in enumerate(clinical_names):
        for col in metadata.columns[1:]:
            cov_values = metadata[col].to_numpy()
            if col in ["site", "sex"]:
                cov_values = OrdinalEncoder().fit_transform(cov_values[:, np.newaxis])[:, 0]
            print(f"Correlation between {col} and {score} : "
                  f"{pearsonr(cov_values, clinical_data[:, score_idx])}")
        for roi_idx, roi in enumerate(rois_names):
            df = base_df.copy()
            df["roi"] = rois_data[:, roi_idx]
            df["score"] = clinical_data[:, score_idx]
            formula = "roi ~ score"
            if len(continuous_covs) > 0:
                formula += " + " + " + ".join(continuous_covs)
            if len(categorical_covs) > 0:
                formula += " + " + " + ".join([f"C({cov})" for cov in categorical_covs])
            model = sm.OLS.from_formula(
                formula,
                data=df)
            results = model.fit()
            associations[score_idx, roi_idx] = results.params["score"]
            pvalues[score_idx, roi_idx] = results.pvalues["score"]
    
    idx_sign = (pvalues < significativity_thr)
    print(idx_sign.sum())
    print(idx_sign.sum(axis=1))
    for score_idx, score in enumerate(clinical_names):
        print(score, idx_sign[score_idx].sum())
        if idx_sign[score_idx].sum() > 0:
            rois_idx = np.where(idx_sign[score_idx])
            areas = ["_".join(name.split("_")[:-1]) for name in rois_names[rois_idx]]
            values = associations[score_idx, rois_idx].squeeze(0)
            print(len(areas))
            print(values.shape)

            color_name = "Plotly"
            if len(areas) <= 6:
                color_name = "tab10"
            elif len(areas) <= 9:
                color_name = "Plotly"
            elif len(areas) <= 10:
                color_name = "tab10"
            elif len(areas) <= 12:
                color_name = "Paired"
            else:
                color_name = "Alphabet"
            print("Number of significative rois in thickness for {} : ".format(score), len(areas))
            plt.rcParams.update({'font.size': 20, "font.family": "serif"})
            plot_areas(areas, np.arange(len(areas)) + 0.01, color_name=color_name)
            plot_coefs(areas, values, color_name=color_name)
    plt.show()


def combine_all_pvalues(pvalues, method="fisher"):
    combined_pvalues = np.ones(pvalues.shape[-2:])
    for score_idx in range(pvalues.shape[-2]):
        for roi_metric_idx in range(pvalues.shape[-1]):
            res = combine_pvalues(
                pvalues[:, :, score_idx, roi_metric_idx].flatten())
            combined_pvalues[score_idx, roi_metric_idx] = res[1]
    return combined_pvalues


def compute_associations(dataset, datasetdir, outdir, runs,
                         heuristics, heuristics_params,
                         metrics=["thickness", "meancurv", "area"],
                         scores=None, model_indices=None, 
                         validation_indices=None, n_subjects=150):
    assert len(runs) == 2

    global_results = []

    # Computing heuristics with various parameters for each metric / score
    for run_idx, run in enumerate(runs):
        run_results = {}
        expdir = os.path.join(outdir, run)
        daadir = os.path.join(expdir, "daa")
        # print_text(f"experimental directory: {expdir}")
        # print_text(f"DAA directory: {daadir}")
        simdirs = [path for path in glob.glob(os.path.join(daadir, "*"))
                if os.path.isdir(path)]
        # print_text(f"Simulation directories: {','.join(simdirs)}")

        flags_file = os.path.join(expdir, "flags.rar")
        if not os.path.isfile(flags_file):
            raise ValueError("You need first to train the model.")
        checkpoints_dir = os.path.join(expdir, "checkpoints")
        # experiment, flags = MultimodalExperiment.get_experiment(
        #     flags_file, checkpoints_dir)

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
                                      rois_names=rois_names)

        for dirname in simdirs:
            # print_text(dirname)
            if not os.path.exists(os.path.join(dirname, "coefs.npy")):
                continue
            coefs = np.load(os.path.join(dirname, "coefs.npy"))
            pvalues = np.load(os.path.join(dirname, "pvalues.npy"))

            if model_indices is None:
                run_model_indices = range(len(pvalues))
            elif np.array(model_indices).ndim == 2:
                run_model_indices = model_indices[:, run_idx]
            else:
                run_model_indices = model_indices

            n_validation = pvalues.shape[1]
            if validation_indices is not None:
                n_validation = len(validation_indices)


            sampling = dirname.split("sampling_")[1].split("_sample")[0]
            sample_latent = dirname.split("latents_")[1].split("_seed")[0]
            local_n_subjects = int(dirname.split("subjects_")[1].split("_M")[0])

            if local_n_subjects != n_subjects:
                continue
            
            dir_results = {}
            # Selection of model / validation indices of interest
            # print(validation_indices)
            if validation_indices is not None:
                pvalues = pvalues[:, validation_indices]
                coefs = coefs[:, validation_indices]
            else:
                pvalues = pvalues[run_model_indices]
                coefs = coefs[run_model_indices]

            # Aggregation
            average_pvalues = pvalues.mean((0, 1))
            product_pvalues = pvalues.prod((0, 1))
            min_pvalues = pvalues.min((0, 1))
            std_pvalues = pvalues.std((0, 1))
            average_coefs = coefs.mean((0, 1))
            max_coefs = np.absolute(coefs).max((0, 1))
            std_coefs = coefs.std((0, 1))

            other_agg_pvalues = {}
            combine_pvalues_heuristics = [heuristic for heuristic in heuristics
                                          if "pvalues_combine" in heuristic]
            for heuristic in combine_pvalues_heuristics:
                method = heuristic.split("combine_")[-1]
                other_agg_pvalues[method] = combine_all_pvalues(pvalues, method)
            
            if "pvalues_coefs" in heuristics:
                other_agg_pvalues["coefs"] = non_nullity_coef(coefs)
            
            weighted_mean_heuristics = [heuristic for heuristic in heuristics
                                        if "weighted_mean" in heuristic]
            if len(weighted_mean_heuristics) != 0:
                model_scores = score_models(dataset, datasetdir, outdir, run,
                                            scores=scores)
                for heuristic in weighted_mean_heuristics:
                    method = heuristic.split("weighted_mean_")[-1]
                    if method.startswith("rank"):
                        sorted_idx = np.argsort(
                            model_scores[run_model_indices]).tolist()
                        weights = []
                        for idx in range(len(run_model_indices)):
                            weights.append(sorted_idx.index(idx) + 1)
                        weights = np.array(weights)
                    elif method.startswith("score"):
                        weights = model_scores[run_model_indices]
                    if heuristic.endswith("softmax"):
                        weights = np.exp(weights)
                    elif heuristic.endswith("log"):
                        weights = np.log(weights)
                    weights = weights / weights.sum()            
                    other_agg_pvalues[method] = np.average(
                        coefs.mean(1), axis=0, weights=weights)
            if "pvalues_vote" in heuristics:
                dir_results["pvalues_vote"] = {}
                for trust_level, vote_prop in itertools.product(
                    heuristics_params["pvalues_vote"]["vote_prop"],
                    heuristics_params["pvalues_vote"]["trust_level"]):
                    _, df = compute_significativity(
                        pvalues, trust_level, vote_prop, n_validation, additional_data,
                        correct_threshold=True, verbose=False)
                    dir_results["pvalues_vote"][
                        f"vote_prop_{vote_prop}_trust_level_{trust_level}"] = df

            rois = np.array(
                list(set([name.rsplit("_", 1)[0] for name in rois_names])))
            
            for metric in metrics:
                for score in scores:
                    score_idx = clinical_names.index(score)
                    metric_indices = np.array(
                        [roi_idx for roi_idx, name in enumerate(rois_names)
                        if metric in name])
                    for heuristic in heuristics:
                        higher_is_better = False
                        if "strategy" in heuristics_params[heuristic]:
                            if heuristic.startswith("coefs"):
                                if heuristic.endswith("mean"):
                                    agg_coefs = average_coefs
                                elif heuristic.endswith("max"):
                                    agg_coefs = max_coefs
                                elif "weighted_mean" in heuristic:
                                    method = heuristic.split("weighted_mean_")[-1]
                                    agg_coefs = other_agg_pvalues[method]
                                values = np.absolute(agg_coefs[score_idx, metric_indices])
                                higher_is_better = True
                                significance_indices = np.argsort(values)[::-1]
                                stds = std_coefs[score_idx, metric_indices]
                            elif heuristic.startswith("pvalues"):
                                if heuristic.endswith("mean"):
                                    agg_pvalues = average_pvalues
                                elif heuristic.endswith("min"):
                                    agg_pvalues = min_pvalues
                                elif heuristic.endswith("prod"):
                                    agg_pvalues = product_pvalues
                                elif "combine" in heuristic:
                                    method = heuristic.split("combine_")[-1]
                                    agg_pvalues = other_agg_pvalues[method]
                                elif heuristic.endswith("coefs"):
                                    agg_pvalues = other_agg_pvalues["coefs"]
                                values = agg_pvalues[score_idx, metric_indices]
                                significance_indices = np.argsort(values)
                                stds = std_pvalues[score_idx, metric_indices]
                            for strategy in heuristics_params[heuristic]["strategy"]:
                                if "-" not in strategy:
                                    for strat_param in heuristics_params[heuristic][strategy]:
                                        if strategy == "num":
                                            rois_indices = significance_indices[:strat_param]
                                        elif strategy == "thr":
                                            ordered_values = values[significance_indices]
                                            rois_indices = significance_indices[
                                                ordered_values >= strat_param if
                                                higher_is_better else
                                                ordered_values <= strat_param]
                                        elif strategy == "var":
                                            ordered_values = values[significance_indices]
                                            ordered_stds = stds[significance_indices]
                                            rois_indices = significance_indices[
                                                ordered_values - strat_param * ordered_stds > 0]
                                        area_idx = metric_indices[rois_indices]
                                        areas = np.array(rois_names)[area_idx]
                                        areas = [area.rsplit("_", 1)[0] for area in areas]
                                        
                                        if heuristic not in dir_results.keys():
                                            dir_results[heuristic] = {}
                                        strat_param_name = f"strategy_{strategy}_value_{strat_param}"
                                        if strat_param_name not in dir_results[heuristic].keys():
                                            dir_results[heuristic][strat_param_name] = {
                                                "metric": [], "roi": [], "score": []}
                                        for area in areas:
                                            dir_results[heuristic][strat_param_name]["score"].append(score)
                                            dir_results[heuristic][strat_param_name]["metric"].append(metric)
                                            dir_results[heuristic][strat_param_name]["roi"].append(area)
                                else:
                                    second_param = strategy.split("-")[1]
                                    for num, other_param in itertools.product(heuristics_params[heuristic]["num"],
                                                                      heuristics_params[heuristic][second_param]):
                                        ordered_values = values[significance_indices]
                                        ordered_stds = stds[significance_indices]
                                        if second_param == "var":
                                            rois_indices = significance_indices[:num][
                                                ordered_values[:num] - other_param * ordered_stds[:num] > 0]
                                        elif second_param == "thr":
                                            rois_indices = significance_indices[:num][
                                                ordered_values[:num] >= other_param if
                                                higher_is_better else
                                                ordered_values[:num] <= other_param]
                                    
                                        area_idx = metric_indices[rois_indices]
                                        areas = np.array(rois_names)[area_idx]
                                        areas = [area.rsplit("_", 1)[0] for area in areas]
                                        
                                        if heuristic not in dir_results.keys():
                                            dir_results[heuristic] = {}
                                        strat_param_name = f"strategy_{strategy}_values_{num}_{other_param}"
                                        if strat_param_name not in dir_results[heuristic].keys():
                                            dir_results[heuristic][strat_param_name] = {
                                                "metric": [], "roi": [], "score": []}
                                        for area in areas:
                                            dir_results[heuristic][strat_param_name]["score"].append(score)
                                            dir_results[heuristic][strat_param_name]["metric"].append(metric)
                                            dir_results[heuristic][strat_param_name]["roi"].append(area)
            run_results[f"{sampling}_{sample_latent}"] = dir_results
        global_results.append(run_results)
    return global_results


def evaluate_stability(dataset, datasetdir, outdir, runs=[],
                       metrics=["thickness", "meancurv", "area"],
                       scores=None, select_good_models=False,
                       n_subjects=150):
    assert len(runs) == 2
    heuristics = ["pvalues_vote", "pvalues_min", "pvalues_mean",
                  "coefs_mean", "coefs_max", 
                  "coefs_weighted_mean_score", "coefs_weighted_mean_rank",
                  "coefs_weighted_mean_score_softmax", "coefs_weighted_mean_rank_softmax",
                  "coefs_weighted_mean_score_log"]
                #"pvalues_combine_fisher", "pvalues_coefs", "pvalues_prod", 
                #   "pvalues_combine_pearson", "pvalues_combine_tippett",
                #   "pvalues_combine_stouffer"]#, "composite"]
    heuristics_params = {
        "pvalues_vote": {"vote_prop": [0.95, 1], "trust_level": [0.95, 1]},
        # "pvalues_prod": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 30)), "thr": [5e-3, 1e-5, 1e-20, 1e-50, 1e-100], "var": [1, 0.5, 0.25]},
        "pvalues_min": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 30)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "pvalues_coefs": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 50)), "thr": [1e-250, 1e-200, 1e-150, 1e-100, 1e-50, 1e-20], "var": [1, 0.5, 0.25]},
        # "pvalues_combine_fisher": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "pvalues_combine_pearson": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "pvalues_combine_tippett": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "pvalues_combine_stouffer": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "pvalues_combine_mudholkar_george": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        "pvalues_mean": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 30)), "thr": [1e-3, 1e-5, 1e-8], "var": [1, 0.5, 0.25]},
        "coefs_mean": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 30)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        "coefs_max": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 30)), "thr": [1e-3, 1e-5, 1e-8], "var": [1, 0.5, 0.25]},
        #"composite": {"strategy": ["thr", "num", "var"], "num": [10], "thr": [1e-10]}
        "coefs_weighted_mean_score": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 30)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        "coefs_weighted_mean_rank": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 30)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        "coefs_weighted_mean_score_softmax": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 30)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        "coefs_weighted_mean_rank_softmax": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 30)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        "coefs_weighted_mean_score_log": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 30)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
    }

    import matplotlib.pyplot as plt

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
    if select_good_models:
        n_worsts_to_remove = 5
        model_indices = []
        for run in runs:
            model_indices.append(list(range(50)))
            model_scores = score_models(dataset, datasetdir, outdir, run, scores=scores)
            worst_models = np.argsort(model_scores)[:n_worsts_to_remove]
            for model_idx in worst_models:
                model_indices[-1].remove(model_idx)
        model_indices = np.array(model_indices).T
        # n_params -= n_worsts_to_remove

    # Computing heuristics with various parameters for each metric / score
    global_results = compute_associations(dataset, datasetdir, outdir, runs,
                                          heuristics, heuristics_params,
                                          metrics, scores, model_indices,
                                          n_subjects=n_subjects)


    # Computing stability
    ideal_Ns = np.array(list(range(1, 25)))#np.sqrt(len(rois))

    best_values = {"stability" : np.empty(len(ideal_Ns)),
                   "penalized_stability": np.empty(len(ideal_Ns)),
                   "heuristic": np.empty(len(ideal_Ns), dtype=object),
                   "strat_param" : np.empty(len(ideal_Ns), dtype=object),
                   "daa_params": np.empty(len(ideal_Ns), dtype=object)}

    best_values_per_metric_score = {
        "stability" : np.zeros((len(metrics), len(scores), len(ideal_Ns))),
        "penalized_stability": np.zeros((len(metrics), len(scores), len(ideal_Ns))),
        "heuristic": np.empty((len(metrics), len(scores), len(ideal_Ns)), dtype=object),
        "strat_param" : np.empty((len(metrics), len(scores), len(ideal_Ns)), dtype=object),
        "daa_params": np.empty((len(metrics), len(scores), len(ideal_Ns)), dtype=object)
    }

    best_values_per_metric = {
        "stability" : np.zeros((len(metrics), len(ideal_Ns))),
        "penalized_stability": np.zeros((len(metrics), len(ideal_Ns))),
        "heuristic": np.empty((len(metrics), len(ideal_Ns)), dtype=object),
        "strat_param" : np.empty((len(metrics), len(ideal_Ns)), dtype=object),
        "daa_params": np.empty((len(metrics), len(ideal_Ns)), dtype=object)
    }

    best_values_per_score = {
        "stability" : np.zeros((len(scores), len(ideal_Ns))),
        "penalized_stability": np.zeros((len(scores), len(ideal_Ns))),
        "heuristic": np.empty((len(scores), len(ideal_Ns)), dtype=object),
        "strat_param" : np.empty((len(scores), len(ideal_Ns)), dtype=object),
        "daa_params": np.empty((len(scores), len(ideal_Ns)), dtype=object)
    }

    variables = list(best_values.keys())

    # Compute penalized stability for each ideal_N value
    for N_idx, ideal_N in enumerate(ideal_Ns):
        stability_per_score_metric = {
            "daa_params": [], "heuristic": [], "strat_param": [], "metric": [],
            "score": [], "stability": [], "penalized_stability": []}
        for daa_params in set(list(global_results[0].keys())).intersection(global_results[1].keys()):
            for heuristic in heuristics:
                if "strategy" in heuristics_params[heuristic]:
                    for strategy in heuristics_params[heuristic]["strategy"]:
                        if "-" not in strategy:
                            for strat_param in heuristics_params[heuristic][strategy]:
                                strat_param_name = f"strategy_{strategy}_value_{strat_param}"

                                local_stability_per_metric_score = (
                                    compute_all_stability(global_results,
                                                          daa_params,
                                                          heuristic,
                                                          strat_param_name,
                                                          ideal_N, metrics,
                                                          scores))
                                for key, value in stability_per_score_metric.items():
                                    value += local_stability_per_metric_score[key]

                        else:
                            second_param = strategy.split("-")[1]
                            for num, other_param in itertools.product(heuristics_params[heuristic]["num"],
                                                            heuristics_params[heuristic][second_param]):
                                strat_param_name = f"strategy_{strategy}_values_{num}_{other_param}"

                                local_stability_per_metric_score = (
                                    compute_all_stability(global_results,
                                                          daa_params,
                                                          heuristic,
                                                          strat_param_name,
                                                          ideal_N, metrics,
                                                          scores))
                                for key, value in stability_per_score_metric.items():
                                    value += local_stability_per_metric_score[key]
                else:
                    for trust_level, vote_prop in itertools.product(
                        heuristics_params["pvalues_vote"]["vote_prop"],
                        heuristics_params["pvalues_vote"]["trust_level"]):

                        strat_param_name = f"vote_prop_{vote_prop}_trust_level_{trust_level}"

                        local_stability_per_metric_score = (
                            compute_all_stability(global_results, daa_params,
                                                  heuristic, strat_param_name,
                                                  ideal_N, metrics, scores))
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
                local_stability = stability_per_score_metric[idx]
                sorted_local_stability = local_stability.sort_values(
                    "penalized_stability", ascending=False)
                for variable in variables:
                    best_values_per_metric_score[variable][
                        metric_idx, score_idx, N_idx] = (
                        sorted_local_stability[variable].to_list()[0])

        final_stability = stability_per_score_metric.groupby(
            ["daa_params", "heuristic", "strat_param"],
            as_index=False).mean()
        sorted_stability = final_stability.sort_values(
            "penalized_stability", ascending=False)
        for variable in variables:
            best_values[variable][N_idx] = (
                sorted_stability[variable].to_list()[0])
        sorted_stability = stability_per_score_metric.groupby(
                ["daa_params", "heuristic", "strat_param"],
                as_index=False).mean()

        for metric_idx, metric in enumerate(metrics):
            idx = (stability_per_score_metric["metric"] == metric)
            local_stability = stability_per_score_metric[idx].groupby(
                ["daa_params", "heuristic", "strat_param", "metric"],
                as_index=False).mean()
            sorted_local_stability = local_stability.sort_values(
                "penalized_stability", ascending=False)
            for variable in variables:
                best_values_per_metric[variable][metric_idx, N_idx] = (
                    sorted_local_stability[variable].to_list()[0])

        for score_idx, score in enumerate(scores):
            idx = (stability_per_score_metric["score"] == score)
            local_stability = stability_per_score_metric[idx].groupby(
                ["daa_params", "heuristic", "strat_param", "score"],
                as_index=False).mean()
            sorted_local_stability = local_stability.sort_values(
                "penalized_stability", ascending=False)
            for variable in variables:
                best_values_per_score[variable][score_idx, N_idx] = (
                    sorted_local_stability[variable].to_list()[0])
    
    # Plot stability for each case
    plot_stability = True
    plot_heuristic_hist = False
    if plot_stability:
        plt.plot(ideal_Ns, best_values["stability"], label="raw")
        plt.plot(ideal_Ns, best_values["penalized_stability"], label="penalized")
        plt.legend(title="Stability")
        plt.title("Best stability when varying N*")
        for metric_idx, metric in enumerate(metrics):
            fig, ax = plt.subplots(figsize=(12, 9))
            ax.set_title(f"Stability for {metric}")
            handles = []
            for score_idx, score in enumerate(scores):
                handle = ax.plot(ideal_Ns, best_values_per_metric_score["stability"][metric_idx, score_idx],
                                label=score, ls="-", c=list(colors.TABLEAU_COLORS)[score_idx])
                ax.plot(ideal_Ns, best_values_per_metric_score["penalized_stability"][metric_idx, score_idx],
                        ls="--", c=list(colors.TABLEAU_COLORS)[score_idx])
                handles += handle
            
            # create manual symbols for legend
            line_stab = lines.Line2D([0], [0], label='raw', color='k', ls="-")
            line_pen_stab = lines.Line2D([0], [0], label='penalized', color='k', ls="--")

            handle = ax.plot(ideal_Ns, best_values_per_metric["stability"][metric_idx],
                            label="Average", ls="-", c="k", lw=3)
            ax.plot(ideal_Ns, best_values_per_metric["penalized_stability"][metric_idx],
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
                handle = ax.plot(ideal_Ns, best_values_per_metric_score["stability"][metric_idx, score_idx],
                                 label=metric, ls="-", c=list(colors.TABLEAU_COLORS)[metric_idx])
                ax.plot(ideal_Ns, best_values_per_metric_score["penalized_stability"][metric_idx, score_idx],
                        ls="--", c=list(colors.TABLEAU_COLORS)[metric_idx])
                handles += handle

            # create manual symbols for legend
            line_stab = lines.Line2D([0], [0], label='raw', color='k', ls="-")
            line_pen_stab = lines.Line2D([0], [0], label='penalized', color='k', ls="--")

            handle = ax.plot(ideal_Ns, best_values_per_score["stability"][score_idx],
                            label="Average", ls="-", c="k", lw=3)
            ax.plot(ideal_Ns, best_values_per_score["penalized_stability"][score_idx],
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
    
    for metric_idx, metric in enumerate(metrics):
        local_values = {}
        for variable in variables:
            local_values[variable] = best_values_per_metric[variable][metric_idx]

        best_pen_N_idx = np.argwhere(local_values["penalized_stability"] == 
                                     np.amax(local_values["penalized_stability"])
                                    ).flatten()
        best_N_idx = np.argwhere(local_values["stability"] ==
                                 np.amax(local_values["stability"])).flatten()
        best_pen_stab = local_values["penalized_stability"][best_pen_N_idx]
        best_stab = local_values["stability"][best_N_idx]
        best_pen_N = ideal_Ns[best_pen_N_idx]
        best_N = ideal_Ns[best_N_idx]
        best_pen_param = local_values["strat_param"][best_pen_N_idx]
        best_params = local_values["strat_param"][best_N_idx]
        best_pen_daa = local_values["daa_params"][best_pen_N_idx]
        best_daa = local_values["daa_params"][best_N_idx]
        best_pen_heuristic = local_values["heuristic"][best_pen_N_idx]
        best_heuristic = local_values["heuristic"][best_N_idx]
        print(f"Best average penalized stability for {metric} : {best_pen_stab} for N_pen in {best_pen_N} and {best_pen_heuristic} with {best_pen_param} and daa params {best_pen_daa}.")
        print(f"Best average stability for {metric} : {best_stab} for N_pen in {best_N} and {best_heuristic} with {best_params} and daa params {best_daa}.")
    
    for score_idx, score in enumerate(scores):
        local_values = {}
        for variable in variables:
            local_values[variable] = best_values_per_score[variable][score_idx]

        best_pen_N_idx = np.argwhere(local_values["penalized_stability"] == 
                                     np.amax(local_values["penalized_stability"])
                                    ).flatten()
        best_N_idx = np.argwhere(local_values["stability"] ==
                                 np.amax(local_values["stability"])).flatten()
        best_pen_stab = local_values["penalized_stability"][best_pen_N_idx]
        best_stab = local_values["stability"][best_N_idx]
        best_pen_N = ideal_Ns[best_pen_N_idx]
        best_N = ideal_Ns[best_N_idx]
        best_pen_param = local_values["strat_param"][best_pen_N_idx]
        best_params = local_values["strat_param"][best_N_idx]
        best_pen_daa = local_values["daa_params"][best_pen_N_idx]
        best_daa = local_values["daa_params"][best_N_idx]
        best_pen_heuristic = local_values["heuristic"][best_pen_N_idx]
        best_heuristic = local_values["heuristic"][best_N_idx]
        print(f"Best average penalized stability for {score} : {best_pen_stab} for N_pen in {best_pen_N} and {best_pen_heuristic} with {best_pen_param} and daa params {best_pen_daa}.")
        print(f"Best average stability for {score} : {best_stab} for N_pen in {best_N} and {best_heuristic} with {best_params} and daa params {best_daa}.")


    best_pen_N_idx = np.argwhere(best_values["penalized_stability"] == 
                                    np.amax(best_values["penalized_stability"])
                                ).flatten()
    best_N_idx = np.argwhere(best_values["stability"] ==
                                np.amax(best_values["stability"])).flatten()
    best_pen_stab = best_values["penalized_stability"][best_pen_N_idx]
    best_stab = best_values["stability"][best_N_idx]
    best_pen_N = ideal_Ns[best_pen_N_idx]
    best_N = ideal_Ns[best_N_idx]
    best_pen_param = best_values["strat_param"][best_pen_N_idx]
    best_params = best_values["strat_param"][best_N_idx]
    best_pen_daa = best_values["daa_params"][best_pen_N_idx]
    best_daa = best_values["daa_params"][best_N_idx]
    best_pen_heuristic = best_values["heuristic"][best_pen_N_idx]
    best_heuristic = best_values["heuristic"][best_N_idx]
    print(f"Best average penalized stability overall : {best_pen_stab} for N_pen in {best_pen_N} and {best_pen_heuristic} with {best_pen_param} and daa params {best_pen_daa}.")
    print(f"Best average stability overall : {best_stab} for N_pen in {best_N} and {best_heuristic} with {best_params} and daa params {best_daa}.")
    plt.show()


def evaluate_stability_scaling(dataset, datasetdir, outdir, runs=[],
                               metrics=["thickness", "meancurv", "area"],
                               scores=None, vary_models=True, n_subjects=150,
                               scaling_params=None, select_good_models=0):
    assert len(runs) == 2
    heuristics = ["pvalues_vote", "pvalues_min", "pvalues_mean",
                  "coefs_max", "coefs_mean"]#,
                #   "coefs_weighted_mean_score", "coefs_weighted_mean_rank"]
                #, "pvalues_combine_fisher", "pvalues_coefs", "pvalues_prod",
                #   "pvalues_combine_pearson", "pvalues_combine_tippett",
                #   "pvalues_combine_stouffer"]#, "composite"]
    heuristics_params = {
        "pvalues_vote": {"vote_prop": [0.95, 1], "trust_level": [0.95, 1]},
        # "pvalues_prod": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 30)), "thr": [5e-3, 1e-5, 1e-20, 1e-50, 1e-100], "var": [1, 0.5, 0.25]},
        "pvalues_min": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 30)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "pvalues_coefs": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 30)), "thr": [1e-250, 1e-200, 1e-150, 1e-100, 1e-50, 1e-20], "var": [1, 0.5, 0.25]},
        # "pvalues_combine_fisher": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "pvalues_combine_pearson": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "pvalues_combine_tippett": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "pvalues_combine_stouffer": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        # "pvalues_combine_mudholkar_george": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 30)), "thr": [5e-2, 1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        "pvalues_mean": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 30)), "thr": [1e-3, 1e-5, 1e-8], "var": [1, 0.5, 0.25]},
        "coefs_mean": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 30)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        "coefs_max": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 30)), "thr": [1e-3, 1e-5, 1e-8], "var": [1, 0.5, 0.25]},
        "coefs_weighted_mean_score": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 30)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        "coefs_weighted_mean_rank": {"strategy": ["thr", "num", "var", "num-var", "num-thr"], "num": list(range(1, 30)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        
        #"composite": {"strategy": ["thr", "num", "var"], "num": [10], "thr": [1e-10]}
    }

    # heuristics = ["coefs_mean"]
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

    additional_data = SimpleNamespace(metadata_columns=metadata_columns,
                                    clinical_names=clinical_names,
                                    rois_names=rois_names)

    # Computing heuristics with various parameters for each metric / score
    if scaling_params is None:
        scaling_params = list(range(50))
        if not vary_models:
            scaling_params = list(range(20))

    n_params = len(scaling_params)

    if select_good_models > 0:
        assert select_good_models < n_params
        scaling_params = []
        for run in runs:
            scaling_params.append([])
            model_scores = score_models(dataset, datasetdir, outdir, run, scores=scores)
            sorted_models = np.argsort(model_scores)
            next_model_indices = list(range(50))
            for idx_to_remove in range(select_good_models):
                new_worst_model = sorted_models[idx_to_remove]
                next_model_indices.remove(new_worst_model)
                scaling_params[-1].append(next_model_indices.copy())
        # scaling_params = np.array(scaling_params).T
        n_params = select_good_models

    scaled_global_results = []
    for param_idx in range(1, n_params + 1):
        validation_indices = None
        if select_good_models > 0:
            run0_indices = scaling_params[0][param_idx - 1]
            run1_indices = scaling_params[1][param_idx - 1]
            model_indices = np.array([run0_indices, run1_indices]).T
        else:
            model_indices = scaling_params[:param_idx]
            if not vary_models:
                validation_indices = scaling_params[:param_idx]
                model_indices = None
        global_results = compute_associations(dataset, datasetdir, outdir, runs,
                                              heuristics, heuristics_params,
                                              metrics, scores, model_indices=model_indices,
                                              validation_indices=validation_indices,
                                              n_subjects=n_subjects)
        scaled_global_results.append(global_results)


    # Computing stability
    ideal_Ns = np.array(list(range(1, 25))) # np.sqrt(len(rois))

    best_values = {"stability" : np.empty((n_params, len(ideal_Ns))),
                   "penalized_stability": np.empty((n_params, len(ideal_Ns))),
                   "heuristic": np.empty((n_params, len(ideal_Ns)), dtype=object),
                   "strat_param" : np.empty((n_params, len(ideal_Ns)), dtype=object),
                   "daa_params": np.empty((n_params, len(ideal_Ns)), dtype=object)}

    best_values_per_metric_score = {
        "stability" : np.zeros((len(metrics), len(scores), n_params, len(ideal_Ns))),
        "penalized_stability": np.zeros((len(metrics), len(scores), n_params, len(ideal_Ns))),
        "heuristic": np.empty((len(metrics), len(scores), n_params, len(ideal_Ns)), dtype=object),
        "strat_param" : np.empty((len(metrics), len(scores), n_params, len(ideal_Ns)), dtype=object),
        "daa_params": np.empty((len(metrics), len(scores), n_params, len(ideal_Ns)), dtype=object)
    }

    best_values_per_metric = {
        "stability" : np.zeros((len(metrics), n_params, len(ideal_Ns))),
        "penalized_stability": np.zeros((len(metrics), n_params, len(ideal_Ns))),
        "heuristic": np.empty((len(metrics), n_params, len(ideal_Ns)), dtype=object),
        "strat_param" : np.empty((len(metrics), n_params, len(ideal_Ns)), dtype=object),
        "daa_params": np.empty((len(metrics), n_params, len(ideal_Ns)), dtype=object)
    }

    best_values_per_score = {
        "stability" : np.zeros((len(scores), n_params, len(ideal_Ns))),
        "penalized_stability": np.zeros((len(scores), n_params, len(ideal_Ns))),
        "heuristic": np.empty((len(scores), n_params, len(ideal_Ns)), dtype=object),
        "strat_param" : np.empty((len(scores), n_params, len(ideal_Ns)), dtype=object),
        "daa_params": np.empty((len(scores), n_params, len(ideal_Ns)), dtype=object)
    }

    variables = list(best_values.keys())

    # Compute penalized stability for each ideal_N value
    for param_idx in tqdm(range(n_params)):
        global_results = scaled_global_results[param_idx]
        for N_idx, ideal_N in enumerate(ideal_Ns):
            stability_per_score_metric = {
                "daa_params": [], "heuristic": [], "strat_param": [],
                "metric": [], "score": [], "stability": [],
                "penalized_stability": []}
            for daa_params in set(list(global_results[0].keys())).intersection(
                global_results[1].keys()):
                for heuristic in heuristics:
                    if "strategy" in heuristics_params[heuristic]:
                        for strategy in heuristics_params[heuristic]["strategy"]:
                            if "-" not in strategy:
                                for strat_param in heuristics_params[heuristic][strategy]:
                                    strat_param_name = f"strategy_{strategy}_value_{strat_param}"

                                    local_stability_per_metric_score = (
                                        compute_all_stability(global_results,
                                                            daa_params,
                                                            heuristic,
                                                            strat_param_name,
                                                            ideal_N, metrics,
                                                            scores))
                                    for key, value in stability_per_score_metric.items():
                                        value += local_stability_per_metric_score[key]

                            else:
                                second_param = strategy.split("-")[1]
                                for num, other_param in itertools.product(heuristics_params[heuristic]["num"],
                                                                heuristics_params[heuristic][second_param]):
                                    strat_param_name = f"strategy_{strategy}_values_{num}_{other_param}"

                                    local_stability_per_metric_score = (
                                        compute_all_stability(global_results,
                                                            daa_params,
                                                            heuristic,
                                                            strat_param_name,
                                                            ideal_N, metrics,
                                                            scores))
                                    for key, value in stability_per_score_metric.items():
                                        value += local_stability_per_metric_score[key]
                    else:
                        for trust_level, vote_prop in itertools.product(
                            heuristics_params["pvalues_vote"]["vote_prop"],
                            heuristics_params["pvalues_vote"]["trust_level"]):

                            strat_param_name = f"vote_prop_{vote_prop}_trust_level_{trust_level}"

                            local_stability_per_metric_score = (
                                compute_all_stability(global_results, daa_params,
                                                    heuristic, strat_param_name,
                                                    ideal_N, metrics, scores))
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
                    local_stability = stability_per_score_metric[idx]
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


def non_nullity_coef(coefs):
    combined_pvalues = np.ones(coefs.shape[-2:], dtype="double")
    for score_idx in range(coefs.shape[-2]):
        for roi_metric_idx in range(coefs.shape[-1]):
            df_coefs = pd.DataFrame(
                coefs[:, :, score_idx, roi_metric_idx].flatten(), columns=['beta'])
            est = sm.OLS.from_formula("beta ~ 1", data=df_coefs)
            idx_of_beta = "Intercept"
            results = est.fit()
            combined_pvalues[score_idx, roi_metric_idx] = (
                results.pvalues[idx_of_beta])
    return combined_pvalues


def study_heuristics(dataset, datasetdir, outdir, runs=[],
                     metrics=["thickness", "meancurv", "area"],
                     scores=None):
    assert len(runs) == 2
    heuristics = [#"pvalues_vote", "pvalues_min", "pvalues_mean",
                  #"coefs_mean", "coefs_max",
                  "pvalues_combine_fisher",
                  "pvalues_combine_pearson", "pvalues_combine_tippett",
                  "pvalues_combine_stouffer"]#, "composite"]
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

    clinical_names = np.load(
            os.path.join(datasetdir, "clinical_names.npy"), allow_pickle=True)
    clinical_names = clinical_names.tolist()
    if scores is None:
        scores = clinical_names
    rois_names = np.load(
        os.path.join(datasetdir, "rois_names.npy"), allow_pickle=True)
    rois = np.array(
                list(set([name.rsplit("_", 1)[0] for name in rois_names])))

    daa_params = "gaussian_False"


    # Computing heuristics with various parameters for each metric / score
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
            if f"{sampling}_{sample_latent}" != daa_params or n_subjects != 150:
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
            combined_pvalues = non_nullity_coef(coefs)
            for score_idx, score in enumerate(scores):
                for metric_idx, metric in enumerate(rois_names):
                    value = combined_pvalues[score_idx, metric_idx]
                    if value == 0:
                        print(score, metric)
            x = combined_pvalues.flatten()
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
            plt.figure()
            plt.hist(x[x != 0], bins=np.power(10, bins))#logbins)
            plt.xscale('log')
            plt.title(f"Combine coefs pvalues histogram for run {run} with {daa_params}")
            plt.xlabel("log10 pvalues")                
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
    plt.show()