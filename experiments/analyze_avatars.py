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
from multimodal_cohort.experiment import MultimodalExperiment
from multimodal_cohort.constants import short_clinical_names
from multimodal_cohort.utils import extract_and_order_by
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from scipy.stats import pearsonr

from color_utils import (print_title, print_subtitle, print_text)
from daa_functions import compute_significativity


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


def evaluate_stability(dataset, datasetdir, outdir, runs=[],
                       metrics=["thickness", "meancurv", "area"],
                       scores=None):
    assert len(runs) == 2
    heuristics = ["pvalues_vote", "pvalues_min", "pvalues_mean",
                  "coefs_mean", "coefs_max"]#, "composite"]
    heuristics_params = {
        "pvalues_vote": {"vote_prop": [0.95, 1], "trust_level": [0.95, 1]},
        "pvalues_prod": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 50)), "thr": [1e-20, 1e-50, 1e-100], "var": [1, 0.5, 0.25]},
        "pvalues_min": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 50)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        "pvalues_mean": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 50)), "thr": [1e-3, 1e-5, 1e-8], "var": [1, 0.5, 0.25]},
        "coefs_mean": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 50)), "thr": [1e-5, 1e-8, 1e-10], "var": [1, 0.5, 0.25]},
        "coefs_max": {"strategy": ["thr", "num", "var", "num-var"], "num": list(range(1, 50)), "thr": [1e-3, 1e-5, 1e-8], "var": [1, 0.5, 0.25]},
        #"composite": {"strategy": ["thr", "num", "var"], "num": [10], "thr": [1e-10]}
    }

    from plotting import plot_areas, plot_coefs
    import matplotlib.pyplot as plt

    global_results = []

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

            n_validation = int(
                dirname.split("n_validation_")[1].split("_n_s")[0])
            sampling = dirname.split("sampling_")[1].split("_sample")[0]
            sample_latent = dirname.split("latents_")[1].split("_seed")[0]
            
            
            dir_results = {}
            average_pvalues = pvalues.mean((0, 1))
            product_pvalues = pvalues.prod((0, 1))
            min_pvalues = pvalues.min((0, 1))
            std_pvalues = pvalues.std((0, 1))
            average_coefs = coefs.mean((0, 1))
            max_coefs = np.absolute(coefs).max((0, 1))
            std_coefs = coefs.std((0, 1))

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
                                values = np.absolute(agg_coefs[score_idx, metric_indices])
                                higher_is_better = True
                                significance_indices = np.argsort(values)[::-1]
                                stds = std_coefs[score_idx, metric_indices]
                            else:
                                if heuristic.endswith("mean"):
                                    agg_pvalues = average_pvalues
                                elif heuristic.endswith("min"):
                                    agg_pvalues = min_pvalues
                                elif heuristic.endswith("prod"):
                                    agg_pvalues = product_pvalues
                                values = agg_pvalues[score_idx, metric_indices]
                                significance_indices = np.argsort(values)
                                stds = std_pvalues[score_idx, metric_indices]
                            for strategy in heuristics_params[heuristic]["strategy"]:
                                if strategy != "num-var":
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
                                    for num, var in itertools.product(heuristics_params[heuristic]["num"],
                                                                      heuristics_params[heuristic]["var"]):
                                        ordered_values = values[significance_indices]
                                        ordered_stds = stds[significance_indices]
                                        rois_indices = significance_indices[:num][
                                            ordered_values[:num] - var * ordered_stds[:num] > 0]
                                    
                                        area_idx = metric_indices[rois_indices]
                                        areas = np.array(rois_names)[area_idx]
                                        areas = [area.rsplit("_", 1)[0] for area in areas]
                                        
                                        if heuristic not in dir_results.keys():
                                            dir_results[heuristic] = {}
                                        strat_param_name = f"strategy_{strategy}_values_{num}_{var}"
                                        if strat_param_name not in dir_results[heuristic].keys():
                                            dir_results[heuristic][strat_param_name] = {
                                                "metric": [], "roi": [], "score": []}
                                        for area in areas:
                                            dir_results[heuristic][strat_param_name]["score"].append(score)
                                            dir_results[heuristic][strat_param_name]["metric"].append(metric)
                                            dir_results[heuristic][strat_param_name]["roi"].append(area)
            run_results[f"{sampling}_{sample_latent}"] = dir_results
        global_results.append(run_results)

    # Computing stability
    ideal_Ns = np.array(list(range(1, 25)))#np.sqrt(len(rois))
    best_stability = []
    best_penalized_stability = []

    best_stability_per_metric_score = np.zeros((len(metrics), len(scores), len(ideal_Ns)))
    best_penalized_stability_per_metric_score = np.zeros((len(metrics), len(scores), len(ideal_Ns)))

    best_stability_per_metric = np.zeros((len(metrics), len(ideal_Ns)))
    best_penalized_stability_per_metric = np.zeros((len(metrics), len(ideal_Ns)))

    best_stability_per_score = np.zeros((len(scores), len(ideal_Ns)))
    best_penalized_stability_per_score = np.zeros((len(scores), len(ideal_Ns)))

    best_heuristic_params = []
    best_heuristic_params_per_metric_score = np.empty((len(metrics), len(scores), len(ideal_Ns)), dtype=object)
    best_heuristic_params_per_metric = np.empty((len(metrics), len(ideal_Ns)), dtype=object)
    best_heuristic_params_per_score = np.empty((len(scores), len(ideal_Ns)), dtype=object)

    for N_idx, ideal_N in enumerate(ideal_Ns):
        final_stability = {
            "daa_params": [], "heuristic": [], "strat_param": [],"stability": [],
            "penalized_stability": []}
        stability_per_score_metric = {
            "daa_params": [], "heuristic": [], "strat_param": [], "metric": [],
            "score": [], "stability": [], "penalized_stability": [], "N0": [], "N1": [], "N01": []}
        for daa_params in set(list(global_results[0].keys())).intersection(global_results[1].keys()):
            dir_results0 = global_results[0][f"{sampling}_{sample_latent}"]
            dir_results1 = global_results[1][f"{sampling}_{sample_latent}"]
            for heuristic in heuristics:
                if "strategy" in heuristics_params[heuristic]:
                    for strategy in heuristics_params[heuristic]["strategy"]:
                        if strategy != "num-var":
                            for strat_param in heuristics_params[heuristic][strategy]:
                                strat_param_name = f"strategy_{strategy}_value_{strat_param}"

                                res0 = dir_results0[heuristic][strat_param_name]
                                res1 = dir_results1[heuristic][strat_param_name]

                                all_assoc0 = list(zip(res0["score"], res0["metric"], res0["roi"]))
                                all_assoc1 = list(zip(res1["score"], res1["metric"], res1["roi"]))
                                # print(daa_params, heuristic, strat_param_name, "run0", len(list(all_assoc0)))
                                # print(daa_params, heuristic, strat_param_name, "run1", len(list(all_assoc1)))
                                # print(list(all_assoc0))
                                all_stability = np.zeros((len(metric), len(scores)))
                                all_penalized_stability = np.zeros((len(metric), len(scores)))
                                for metric_idx, metric in enumerate(metrics):
                                    for score_idx, score in enumerate(scores):
                                        local_assoc0 = [assoc for assoc in all_assoc0 if metric in assoc and score in assoc]
                                        local_assoc1 = [assoc for assoc in all_assoc1 if metric in assoc and score in assoc]
                                        N0 = len(local_assoc0)
                                        N1 = len(local_assoc1)
                                        N01 = len([assoc for assoc in local_assoc0 if assoc in local_assoc1])

                                        eps = 1e-8
                                        stability = (N01 / (N0 + eps) + N01 / (N1 + eps) ) / (N0 / (N01 + eps) + N1 / (N01 + eps) + eps)
                                        penality = 2  / (N01 / ideal_N + ideal_N / (N01 + eps))
                                        penalized_stability = stability * penality
                                        all_stability[metric_idx, score_idx] = stability
                                        all_penalized_stability[metric_idx, score_idx] = penalized_stability
                                        stability_per_score_metric["daa_params"].append(daa_params)
                                        stability_per_score_metric["heuristic"].append(heuristic)
                                        stability_per_score_metric["strat_param"].append(strat_param_name)
                                        stability_per_score_metric["metric"].append(metric)
                                        stability_per_score_metric["score"].append(score)
                                        stability_per_score_metric["stability"].append(stability)
                                        stability_per_score_metric["penalized_stability"].append(penalized_stability)
                                        stability_per_score_metric["N0"].append(N0)
                                        stability_per_score_metric["N1"].append(N1)
                                        stability_per_score_metric["N01"].append(N01)
                                final_stability["daa_params"].append(daa_params)
                                final_stability["heuristic"].append(heuristic)
                                final_stability["strat_param"].append(strat_param_name)
                                final_stability["stability"].append(all_stability.mean())
                                final_stability["penalized_stability"].append(all_penalized_stability.mean())
                        else:
                            for num, var in itertools.product(heuristics_params[heuristic]["num"],
                                                            heuristics_params[heuristic]["var"]):
                                strat_param_name = f"strategy_{strategy}_values_{num}_{var}"

                                res0 = dir_results0[heuristic][strat_param_name]
                                res1 = dir_results1[heuristic][strat_param_name]

                                all_assoc0 = list(zip(res0["score"], res0["metric"], res0["roi"]))
                                all_assoc1 = list(zip(res1["score"], res1["metric"], res1["roi"]))
                                # print(daa_params, heuristic, strat_param_name, "run0", len(list(all_assoc0)))
                                # print(daa_params, heuristic, strat_param_name, "run1", len(list(all_assoc1)))
                                # print(list(all_assoc0))
                                all_stability = np.zeros((len(metric), len(scores)))
                                all_penalized_stability = np.zeros((len(metric), len(scores)))
                                for metric_idx, metric in enumerate(metrics):
                                    for score_idx, score in enumerate(scores):
                                        local_assoc0 = [assoc for assoc in all_assoc0 if metric in assoc and score in assoc]
                                        local_assoc1 = [assoc for assoc in all_assoc1 if metric in assoc and score in assoc]
                                        N0 = len(local_assoc0)
                                        N1 = len(local_assoc1)
                                        N01 = len([assoc for assoc in local_assoc0 if assoc in local_assoc1])

                                        eps = 1e-8
                                        stability = (N01 / (N0 + eps) + N01 / (N1 + eps) ) / (N0 / (N01 + eps) + N1 / (N01 + eps) + eps)
                                        penality = 2  / (N01 / ideal_N + ideal_N / (N01 + eps))
                                        penalized_stability = stability * penality
                                        all_stability[metric_idx, score_idx] = stability
                                        all_penalized_stability[metric_idx, score_idx] = penalized_stability
                                        stability_per_score_metric["daa_params"].append(daa_params)
                                        stability_per_score_metric["heuristic"].append(heuristic)
                                        stability_per_score_metric["strat_param"].append(strat_param_name)
                                        stability_per_score_metric["metric"].append(metric)
                                        stability_per_score_metric["score"].append(score)
                                        stability_per_score_metric["stability"].append(stability)
                                        stability_per_score_metric["penalized_stability"].append(penalized_stability)
                                        stability_per_score_metric["N0"].append(N0)
                                        stability_per_score_metric["N1"].append(N1)
                                        stability_per_score_metric["N01"].append(N01)
                                final_stability["daa_params"].append(daa_params)
                                final_stability["heuristic"].append(heuristic)
                                final_stability["strat_param"].append(strat_param_name)
                                final_stability["stability"].append(all_stability.mean())
                                final_stability["penalized_stability"].append(all_penalized_stability.mean())
                else:
                    for trust_level, vote_prop in itertools.product(
                        heuristics_params["pvalues_vote"]["vote_prop"],
                        heuristics_params["pvalues_vote"]["trust_level"]):

                        strat_param_name = f"vote_prop_{vote_prop}_trust_level_{trust_level}"
                        res0 = dir_results0[heuristic][strat_param_name]
                        res1 = dir_results1[heuristic][strat_param_name]

                        all_assoc0 = list(zip(res0["score"], res0["metric"], res0["roi"]))
                        all_assoc1 = list(zip(res1["score"], res1["metric"], res1["roi"]))
                        all_stability = np.zeros((len(metric), len(scores)))
                        all_penalized_stability = np.zeros((len(metric), len(scores)))
                        for metric_idx, metric in enumerate(metrics):
                            for score_idx, score in enumerate(scores):
                                local_assoc0 = [assoc for assoc in all_assoc0 if metric in assoc and score in assoc]
                                local_assoc1 = [assoc for assoc in all_assoc1 if metric in assoc and score in assoc]
                                N0 = len(local_assoc0)
                                N1 = len(local_assoc1)
                                N01 = len([assoc for assoc in local_assoc0 if assoc in local_assoc1])
                                eps = 1e-8
                                stability = (N01 / (N0 + eps) + N01 / (N1 + eps)) / (N0 / (N01 + eps) + N1 / (N01 + eps) + eps)
                                penality = 2  / (N01 / ideal_N + ideal_N / (N01 + eps))
                                penalized_stability = stability * penality
                                all_stability[metric_idx, score_idx] = stability
                                all_penalized_stability[metric_idx, score_idx] = penalized_stability
                                stability_per_score_metric["daa_params"].append(daa_params)
                                stability_per_score_metric["heuristic"].append(heuristic)
                                stability_per_score_metric["strat_param"].append(strat_param_name)
                                stability_per_score_metric["metric"].append(metric)
                                stability_per_score_metric["score"].append(score)
                                stability_per_score_metric["stability"].append(stability)
                                stability_per_score_metric["penalized_stability"].append(penalized_stability)
                                stability_per_score_metric["N0"].append(N0)
                                stability_per_score_metric["N1"].append(N1)
                                stability_per_score_metric["N01"].append(N01)
                        final_stability["daa_params"].append(daa_params)
                        final_stability["heuristic"].append(heuristic)
                        final_stability["strat_param"].append(strat_param_name)
                        final_stability["stability"].append(all_stability.mean())
                        final_stability["penalized_stability"].append(all_penalized_stability.mean())

        stability_per_score_metric = pd.DataFrame.from_dict(stability_per_score_metric)
        final_stability = pd.DataFrame.from_dict(final_stability)
        # print(stability_per_score_metric.sort_values("penalized_stability", ascending=False))
        # print(final_stability.sort_values("penalized_stability", ascending=False))

        for metric_idx, metric in enumerate(metrics):
            for score_idx, score in enumerate(scores):
                idx = ((stability_per_score_metric["metric"] == metric) &
                    (stability_per_score_metric["score"] == score))
                local_stability = stability_per_score_metric[idx]
                sorted_local_stability = local_stability.sort_values("penalized_stability", ascending=False)
                best_stability_per_metric_score[metric_idx, score_idx, N_idx] = (
                    sorted_local_stability["stability"].to_list()[0])
                best_penalized_stability_per_metric_score[metric_idx, score_idx, N_idx] = (
                    sorted_local_stability["penalized_stability"].to_list()[0])
                best_heuristic_params_per_metric_score[metric_idx, score_idx, N_idx] = (
                    sorted_local_stability["strat_param"].to_list()[0])
                # print(local_stability.sort_values("penalized_stability", ascending=False))
        sorted_stability = final_stability.sort_values("penalized_stability", ascending=False)
        best_stability.append(sorted_stability["stability"].to_list()[0])
        best_penalized_stability.append(sorted_stability["penalized_stability"].to_list()[0])
        best_heuristic_params.append(sorted_stability["strat_param"].to_list()[0])

        for metric_idx, metric in enumerate(metrics):
            idx = (stability_per_score_metric["metric"] == metric)
            local_stability = stability_per_score_metric[idx].groupby(
                ["daa_params", "heuristic", "strat_param", "metric"],
                as_index=False).mean()
            print(local_stability.columns)
            sorted_local_stability = local_stability.sort_values("penalized_stability", ascending=False)
            best_stability_per_metric[metric_idx, N_idx] = (
                sorted_local_stability["stability"].to_list()[0])
            best_penalized_stability_per_metric[metric_idx, N_idx] = (
                sorted_local_stability["penalized_stability"].to_list()[0])
            best_heuristic_params_per_metric[metric_idx, N_idx] = (
                sorted_local_stability["strat_param"].to_list()[0])
                # print(local_stability.sort_values("penalized_stability", ascending=False))
        for score_idx, score in enumerate(scores):
            idx = (stability_per_score_metric["score"] == score)
            local_stability = stability_per_score_metric[idx].groupby(
                ["daa_params", "heuristic", "strat_param", "score"],
                as_index=False).mean()
            sorted_local_stability = local_stability.sort_values("penalized_stability", ascending=False)
            best_stability_per_score[score_idx, N_idx] = (
                sorted_local_stability["stability"].to_list()[0])
            best_penalized_stability_per_score[score_idx, N_idx] = (
                sorted_local_stability["penalized_stability"].to_list()[0])
            best_heuristic_params_per_score[score_idx, N_idx] = (
                sorted_local_stability["strat_param"].to_list()[0])
               
    plot_stability = True
    plot_heuristic_hist = False
    if plot_stability:
        plt.plot(ideal_Ns, best_stability)
        plt.plot(ideal_Ns, best_penalized_stability)
        for metric_idx, metric in enumerate(metrics):
            fig, ax = plt.subplots(figsize=(12, 9))
            ax.set_title(f"Stability for {metric}")
            handles = []
            for score_idx, score in enumerate(scores):
                handle = ax.plot(ideal_Ns, best_stability_per_metric_score[metric_idx, score_idx],
                                label=score, ls="-", c=list(colors.TABLEAU_COLORS)[score_idx])
                ax.plot(ideal_Ns, best_penalized_stability_per_metric_score[metric_idx, score_idx],
                        ls="--", c=list(colors.TABLEAU_COLORS)[score_idx])
                handles += handle
            # handles, labels = plt.gca().get_legend_handles_labels()
            
            # create manual symbols for legend
            line_stab = lines.Line2D([0], [0], label='raw', color='k', ls="-")
            line_pen_stab = lines.Line2D([0], [0], label='penalized', color='k', ls="--")
            # add manual symbols to auto legend
            # handles.extend([patch, line, point]#)
            handle = ax.plot(ideal_Ns, best_stability_per_metric[metric_idx],
                            label="Average", ls="-", c="k", lw=3)
            ax.plot(ideal_Ns, best_penalized_stability_per_metric[metric_idx],
                        ls="--", c="k", lw=3)
            handles += handle

            first_legend = ax.legend(handles=handles, loc='lower right', title="Score")
            ax.add_artist(first_legend)
            ax.legend(handles=[line_stab, line_pen_stab], title="Stability", loc="upper right")
            
            # plt.legend(title="Score", loc="lower right")#handles=handles)
            # plt.legend([line_stab, line_pen_stab], ["raw", "penalized"], title="Stability", loc="upper right")

        for score_idx, score in enumerate(scores):
            fig, ax = plt.subplots(figsize=(12, 9))
            ax.set_title(f"Stability for {score}")
            handles = []
            for metric_idx, metric in enumerate(metrics):
                handle = ax.plot(ideal_Ns, best_stability_per_metric_score[metric_idx, score_idx],
                                 label=metric, ls="-", c=list(colors.TABLEAU_COLORS)[metric_idx])
                ax.plot(ideal_Ns, best_penalized_stability_per_metric_score[metric_idx, score_idx],
                        ls="--", c=list(colors.TABLEAU_COLORS)[metric_idx])
                handles += handle
            # handles, labels = plt.gca().get_legend_handles_labels()

            # create manual symbols for legend
            line_stab = lines.Line2D([0], [0], label='raw', color='k', ls="-")
            line_pen_stab = lines.Line2D([0], [0], label='penalized', color='k', ls="--")
            # add manual symbols to auto legend
            # handles.extend([patch, line, point]#)
            handle = ax.plot(ideal_Ns, best_stability_per_score[score_idx],
                            label="Average", ls="-", c="k", lw=5)
            ax.plot(ideal_Ns, best_penalized_stability_per_score[score_idx],
                        ls="--", c="k", lw=5)
            handles += handle
            
            first_legend = ax.legend(handles=handles, title="Metric", loc="lower right")
            ax.add_artist(first_legend)
            ax.legend(handles=[line_stab, line_pen_stab], title="Stability", loc="upper right")
    
    if plot_heuristic_hist:
        plt.figure(figsize=(24, 16))
        plt.hist(best_heuristic_params)
        plt.xticks(rotation = 315)
        plt.subplots_adjust(bottom=0.3)
        plt.title("Histogram of best heuristics and params on average")
        for metric_idx, metric in enumerate(metrics):
            plt.figure(figsize=(24, 16))
            plt.title(f"Histogram of best heuristic and param for {metric} on average")
            plt.hist(best_heuristic_params_per_metric[metric_idx])
            plt.xticks(rotation = 315)
            plt.subplots_adjust(bottom=0.3)
        for score_idx, score in enumerate(scores):
            plt.figure(figsize=(24, 16))
            plt.title(f"Histogram of best heuristic and param for {score} on average")
            plt.hist(best_heuristic_params_per_score[score_idx])
            plt.xticks(rotation = 315)
            plt.subplots_adjust(bottom=0.3)
        plt.figure(figsize=(24, 16))
        plt.title(f"Histogram of best heuristic and param for {score} accross metrics and scores")
        plt.hist(best_heuristic_params_per_metric_score.reshape((-1, best_heuristic_params_per_metric_score.shape[-1])))
        plt.xticks(rotation = 315)
        plt.subplots_adjust(bottom=0.3)

    for metric_idx, metric in enumerate(metrics):
        for score_idx, score in enumerate(scores):
            local_pen_stab = best_penalized_stability_per_metric_score[metric_idx, score_idx]
            local_stab = best_stability_per_metric_score[metric_idx, score_idx]
            local_params = best_heuristic_params_per_metric_score[metric_idx, score_idx]
            best_pen_N_idx = np.argwhere(local_pen_stab == np.amax(local_pen_stab)).flatten()
            best_N_idx = np.argwhere(local_stab == np.amax(local_stab)).flatten()
            best_pen_stab = local_pen_stab[best_pen_N_idx]
            best_stab = local_stab[best_N_idx]
            best_pen_N = ideal_Ns[best_pen_N_idx]
            best_N = ideal_Ns[best_N_idx]
            best_pen_param = local_params[best_pen_N_idx]
            best_params = local_params[best_N_idx]
            print(f"Best penalized stability for {metric} and {score} : {best_pen_stab} for N_pen in {best_pen_N} and coef mean with {best_pen_param}.")
            print(f"Best stability for {metric} and {score}: {best_stab} for N_pen in {best_N} and coef mean with {best_params}.")
    
    for metric_idx, metric in enumerate(metrics):
        local_pen_stab = best_penalized_stability_per_metric[metric_idx]
        local_stab = best_stability_per_metric[metric_idx]
        local_params = best_heuristic_params_per_metric[metric_idx]
        best_pen_N_idx = np.argwhere(local_pen_stab == np.amax(local_pen_stab)).flatten()
        best_N_idx = np.argwhere(local_stab == np.amax(local_stab)).flatten()
        best_pen_stab = local_pen_stab[best_pen_N_idx]
        best_stab = local_stab[best_N_idx]
        best_pen_N = ideal_Ns[best_pen_N_idx]
        best_N = ideal_Ns[best_N_idx]
        best_pen_param = local_params[best_pen_N_idx]
        best_params = local_params[best_N_idx]
        print(f"Best average penalized stability for {metric} : {best_pen_stab} for N_pen in {best_pen_N} and coef mean with {best_pen_param}.")
        print(f"Best average stability for {metric} : {best_stab} for N_pen in {best_N} and coef mean with {best_params}.")
    
    for score_idx, score in enumerate(scores):
        local_pen_stab = best_penalized_stability_per_score[score_idx]
        local_stab = best_stability_per_score[score_idx]
        local_params = best_heuristic_params_per_score[score_idx]
        best_pen_N_idx = np.argwhere(local_pen_stab == np.amax(local_pen_stab)).flatten()
        best_N_idx = np.argwhere(local_stab == np.amax(local_stab)).flatten()
        best_pen_stab = local_pen_stab[best_pen_N_idx]
        best_stab = local_stab[best_N_idx]
        best_pen_N = ideal_Ns[best_pen_N_idx]
        best_N = ideal_Ns[best_N_idx]
        best_pen_param = local_params[best_pen_N_idx]
        best_params = local_params[best_N_idx]
        print(f"Best average penalized stability for {score} : {best_pen_stab} for N_pen in {best_pen_N} and coef mean with {best_pen_param}.")
        print(f"Best average stability for {score} : {best_stab} for N_pen in {best_N} and coef mean with {best_params}.")

    best_stability = np.array(best_stability)
    best_penalized_stability = np.array(best_penalized_stability)
    best_heuristic_params = np.array(best_heuristic_params)
    best_pen_N_idx = np.argwhere(best_penalized_stability == np.amax(best_penalized_stability)).flatten()
    best_N_idx = np.argwhere(best_stability == np.amax(best_stability)).flatten()
    best_pen_stab = best_penalized_stability[best_pen_N_idx]
    best_stab = best_stability[best_N_idx]
    best_pen_N = ideal_Ns[best_pen_N_idx]
    best_N = ideal_Ns[best_N_idx]
    best_pen_param = best_heuristic_params[best_pen_N_idx]
    best_params = best_heuristic_params[best_N_idx]
    print(f"Best average penalized stability overall : {best_pen_stab} for N_pen in {best_pen_N} and coef mean with {best_pen_param}.")
    print(f"Best average stability overall : {best_stab} for N_pen in {best_N} and coef mean with {best_params}.")
    plt.show()