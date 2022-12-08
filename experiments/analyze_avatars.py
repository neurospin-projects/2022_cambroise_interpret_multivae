import os
import torch
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import seaborn as sns
from multimodal_cohort.experiment import MultimodalExperiment
from multimodal_cohort.constants import short_clinical_names
from multimodal_cohort.utils import extract_and_order_by
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from scipy.stats import pearsonr


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
                     seed=1037):
    
    clinical_file = os.path.join(datasetdir, "clinical_names.npy")
    rois_file = os.path.join(datasetdir, "rois_names.npy")

    params = SimpleNamespace(
        n_validation=n_validation, n_subjects=n_subjects, M=M,
        n_samples=n_samples, reg_method=reg_method,
        sampling=sampling_strategy, sample_latents=sample_latents)

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
    for model_idx in range(flags.num_models):
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
        fig.tight_layout()
    
    for n_votes in range(flags.num_models):
        trust_levels = np.arange(0, 1.01, 0.05)
        assoc_counts = {"score": [], "metric": [], "trust_level": [], "num_assoc":[]}
        for trust_level in trust_levels:
            local_trust_level = params.n_validation * trust_level
            idx_sign = (
                (pvalues < significativity_thr).sum(
                    axis=1) >= local_trust_level).sum(0) > n_votes

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