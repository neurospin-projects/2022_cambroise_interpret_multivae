# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define the different workflows used during the analysis.
"""

# Imports
import os
import glob
import json
import collections
import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from types import SimpleNamespace
import torch
from torch.distributions import Normal
from torch.autograd import Variable
from torch.utils.data import DataLoader
from run_epochs import run_epochs, run_epochs_model
from multimodal_cohort.flags import parser
from utils.filehandling import create_dir_structure
from multimodal_cohort.experiment import MultimodalExperiment
from multimodal_cohort.dataset import DataManager, MissingModalitySampler
from stat_utils import data2cmat, vec2cmat, fit_rsa, make_regression
from color_utils import (
    print_title, print_subtitle, print_text, print_result, print_flags)
from daa_functions import (make_digital_avatars, compute_daa_statistics,
                           compute_significativity)


def train_exp(dataset, datasetdir, outdir, input_dims, input_channels=3,
              input_ico_order=5, use_surface=False, num_models=1, latent_dim=20,
              style_dim=[3, 20], data_seed="defaults",
              num_hidden_layer_encoder=1, num_hidden_layer_decoder=0,
              allow_missing_blocks=True, factorized_representation=True,
              likelihood="normal", learning_rate=0.002, batch_size=256,
              num_epochs=1500, eval_freq=25, eval_freq_fid=100, beta=1.,
              data_multiplications=1, dropout_rate=0., initial_out_logvar=-3.,
              learn_output_scale=True, out_scale_per_subject=False,
              method="joint_elbo", grad_scaling=False,
              learn_output_covmatrix=[]):
    """ Train the model.
    Parameters
    ----------
    dataset: str
        the dataset name: euaims or hbn.
    datasetdir: str
        the path to the dataset associated data.
    outdir: str
        the destination folder.
    input_dims: list of int
        input dimension for each modality.
    num_models: int, default 1
        number of models to train
    latent_dim: int, default 20
        dimension of common factor latent space.
    style_dim: list of int, default [3, 20]
        dimension of specific latent spaces.
    num_hidden_layers: int, default 1
        number of hidden laters in the model.
    allow_missing_blocks: bool, default False
        optionally, allows for missing modalities.
    beta: float, default 1
        default weight of sum of weighted divergence terms.
    likelihood: str or list, default 'normal'
        output distribution.
    learning_rate: float, default 0.002
        starting learning rate.
    batch_size: int, default 256
        batch size for training.
    num_epochs: int, default 1500
        the number of epochs for training.
    eval_freq: int, default 25
        frequency of evaluation of latent representation of generative
        performance (in number of epochs).
    eval_freq_fid: int, default 100
        frequency of evaluation of latent representation of generative
        performance (in number of epochs).
    data_multiplications: int, default 1
        number of pairs per sample.
    dropout_rate: float, default 0
        the dropout rate in the training.
    initial_out_logvar: float, default -3
        initial output logvar.
    learn_output_scale: bool, default True
        optionally, allows for different scales per feature.
    learn_output_covmatrix:  list, default []
        list the modalities for which to learn cov matrix in likelihood model.
    """
    print_title(f"TRAIN: {dataset}")

    flags = SimpleNamespace(
        dataset=dataset, datasetdir=datasetdir, num_models=num_models,
        allow_missing_blocks=allow_missing_blocks, batch_size=batch_size,
        beta=beta, beta_1=0.9, beta_2=0.999, beta_content=1.0,
        beta_style=1.0, calc_nll=False, calc_prd=False,
        class_dim=latent_dim, data_multiplications=data_multiplications,
        dim=64, dir_data="../data", dir_experiment=outdir, dir_fid=None,
        div_weight=None, div_weight_uniform_content=None,
        end_epoch=num_epochs, eval_freq=eval_freq, eval_freq_fid=eval_freq_fid,
        factorized_representation=factorized_representation, img_size_m1=28, img_size_m2=32,
        inception_state_dict="../inception_state_dict.pth",
        initial_learning_rate=learning_rate, use_surface=use_surface,
        initial_out_logvar=initial_out_logvar, input_dim=input_dims,
        input_channels=input_channels, input_ico_order=input_ico_order,
        joint_elbo=False, kl_annealing=0, include_prior_expert=False,
        learn_output_scale=learn_output_scale, learn_output_sample_scale=out_scale_per_subject,
        len_sequence=8, likelihood=likelihood, load_saved=False, method=method,
        model_save="model", modality_jsd=False, modality_moe=False,
        modality_poe=False, num_channels_m1=1, num_channels_m2=3,
        num_classes=2, num_hidden_layer_encoder=num_hidden_layer_encoder,
        num_hidden_layer_decoder=num_hidden_layer_decoder,
        dropout_rate=dropout_rate,
        num_samples_fid=10000, num_training_samples_lr=500,
        poe_unimodal_elbos=True, save_figure=False, start_epoch=0, style_dim=style_dim,
        subsampled_reconstruction=True, data_seed=data_seed, grad_scaling=grad_scaling,
        learn_output_covmatrix=learn_output_covmatrix)
    print(flags)
    use_cuda = torch.cuda.is_available()
    flags.device = torch.device("cuda" if use_cuda else "cpu")
    if flags.method == "poe":
        flags.modality_poe = True
        flags.poe_unimodal_elbos = True
    elif flags.method == "moe":
        flags.modality_moe = True
    elif flags.method == "jsd":
        flags.modality_jsd = True
    elif flags.method == "joint_elbo":
        flags.joint_elbo = True
    else:
        print("Method not implemented...exit!")
        return

    flags.num_mods = len(flags.input_dim)
    if flags.div_weight_uniform_content is None:
        flags.div_weight_uniform_content = 1 / (flags.num_mods + 1)
    flags.alpha_modalities = [flags.div_weight_uniform_content]
    if flags.div_weight is None:
        flags.div_weight = 1 / (flags.num_mods + 1)
    flags.alpha_modalities.extend([
        flags.div_weight for _ in range(flags.num_mods)])
    create_dir_structure(flags)

    if not type(flags.likelihood) is list:
        flags.likelihood = [flags.likelihood] * len(flags.input_dim)

    if not flags.factorized_representation:
        flags.style_dim = [0] * len(flags.style_dim)

    mst = MultimodalExperiment(flags)
    mst.set_optimizers()
    run_epochs(mst)

    if os.path.exists(os.path.join(flags.dir_experiment, "runs.tsv")):
        runs = pd.read_table(os.path.join(flags.dir_experiment, "runs.tsv"))
        new_run = pd.DataFrame(dict(
            name=[flags.str_experiment],
            dataset=[flags.dataset],
            out_scale_per_subject=[flags.learn_output_sample_scale],
            n_hidden_layer_encoder=[flags.num_hidden_layer_encoder],
            n_hidden_layer_decoder=[flags.num_hidden_layer_decoder],
            allow_missing_blocks=[flags.allow_missing_blocks]))
        runs = pd.concat((runs, new_run))
    else:
        runs = dict(name=[],
                    dataset=[],
                    out_scale_per_subject=[],
                    n_hidden_layer_encoder=[],
                    n_hidden_layer_decoder=[],
                    allow_missing_blocks=[])
        for run in os.listdir(flags.dir_experiment):
            if run.startswith("hbn") or run.startswith("euaims"):
                flags = torch.load(os.path.join(flags.dir_experiment, run, "flags.rar"))
                runs["name"].append(flags.str_experiment)
                runs["dataset"].append(flags.dataset)
                runs["out_scale_per_subject"].append(flags.learn_output_sample_scale)
                runs["n_hidden_layer_encoder"].append(flags.num_hidden_layer_encoder)
                runs["n_hidden_layer_decoder"].append(flags.num_hidden_layer_decoder)
                runs["allow_missing_blocks"].append(flags.allow_missing_blocks)
        runs = pd.DataFrame(runs)
    runs.to_csv(os.path.join(flags.dir_experiment, "runs.tsv"), index=False, sep="\t")


def retrain_exp(dataset, datasetdir, outdir, run):

    expdir = os.path.join(outdir, run)
    flags_file = os.path.join(expdir, "flags.rar")
    if not os.path.isfile(flags_file):
        raise ValueError("You need first to train the model.")
    checkpoints_dir = os.path.join(expdir, "checkpoints")
    experiment, flags = MultimodalExperiment.get_experiment(
        flags_file, checkpoints_dir)
    n_models = experiment.flags.num_models
    for model_idx in range(n_models):
        dir_network_last_epoch = os.path.join(checkpoints_dir,
                                        str(flags.end_epoch - 1).zfill(4))
        if n_models > 1:
            dir_network_last_epoch = os.path.join(checkpoints_dir,
                                            f"model_{model_idx}",
                                            str(flags.end_epoch - 1).zfill(4))
        if not os.path.exists(dir_network_last_epoch):
            print_text(f"Retraining model {model_idx}.")
            run_epochs_model(experiment, model_idx)




def daa_exp(dataset, datasetdir, outdir, run, sampling="likelihood",
            n_validation=5, n_samples=200, n_subjects=50,
            M=1000, trust_level=0.75, seed=1037, reg_method="hierarchical",
            sample_latents=True, vote_prop=1):
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
    print_flags(flags)

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
    resdir = os.path.join(daadir, name)
    if not os.path.isdir(resdir):
        os.mkdir(resdir)

    da_file = os.path.join(resdir, "rois_digital_avatars.npy")
    pvals_file = os.path.join(resdir, "pvalues.npy")

    if not os.path.exists(da_file):
        make_digital_avatars(outdir, run, params, additional_data)
    if not os.path.exists(pvals_file):
        compute_daa_statistics(outdir, run, params, additional_data)

    pvalues = np.load(pvals_file)

    print_subtitle("Compute statistics significativity...")
    idx_sign, significant_assoc = compute_significativity(
        pvalues, trust_level, vote_prop, n_validation, additional_data,
        correct_threshold=True)

    significant_file = os.path.join(resdir, "significant_rois.tsv")
    significant_assoc.to_csv(significant_file, sep="\t", index=False)
    print_result(f"significant ROIs: {significant_file}")


def anova_exp(dataset, datasetdir, outdir, run, n_validation=5,
              n_samples=200, n_subjects=50, sampling_strategy="likelihood",
              M=1000, trust_level=0.75, seed=1037, reg_method="hierarchical",
              sample_latents=True, vote_prop=1):
    """ Perform the anova analysis one the subjects coefficient to see if
    there is any site effect.
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
    n_validation: int, default 50
        the number of times we repeat the experiments.
    n_samples: int, default 200
        the size of each traverse.
    n_subjects: int, default 50
        the number of perturbed samples for each clinical score (keep only
        subjects with no missing data).
    k: int, default 1000
        estimate the distribution per clinical scores from k Normal
        distributions.
    trust_level: float, default 0.75
        after thresholding the Bonferoni-corrected p-values at 0.05, apply
        a voting threshold at `trust_level * n_validation`.
    seed: int, default 1037
        optionally specify a seed to control expriment reproducibility, set
        to None for randomization.
    """
    if reg_method != "hierarchical":
        raise("Anova only makes sense when using a hierachical regression")
    print_title(f"ANOVA: {dataset}")
    expdir = os.path.join(outdir, run)
    daadir = os.path.join(expdir, "daa")
    if not os.path.isdir(daadir):
        os.mkdir(daadir)
    print_text(f"experimental directory: {expdir}")
    print_text(f"DAA directory: {daadir}")

    clinical_names = np.load(
        os.path.join(datasetdir, "clinical_names.npy"), allow_pickle=True)
    rois_names = np.load(
        os.path.join(datasetdir, "rois_names.npy"), allow_pickle=True)
    flags = torch.load(os.path.join(expdir, "flags.rar"))
    n_models = flags.num_models
    n_scores = len(clinical_names)
    n_rois = len(rois_names)
    
    params = SimpleNamespace(
        n_validation=n_validation, n_subjects=n_subjects, M=M,
        n_samples=n_samples, reg_method=reg_method,
        sampling=sampling_strategy, sample_latents=sample_latents,
        seed=seed)
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    name = "_".join(["_".join([key, str(val)])
                     for key, val in params.__dict__.items()])
    resdir = os.path.join(daadir, name)
    if not os.path.isdir(resdir):
        os.mkdir(resdir)
    all_coefs_file = os.path.join(resdir, "all_coefs.npy")
    pvals_file = os.path.join(resdir, "pvalues.npy")
    all_coefs = np.load(all_coefs_file, allow_pickle=True)
    pvalues = np.load(pvals_file)
    print_text(f"all coefficients: {all_coefs.shape}")
    
    print_subtitle("Compute statistics significativity...")
    significativity_thr = (0.05 / n_rois / n_scores)
    trust_level = params.n_validation * trust_level
    print_text(f"voting trust level: {trust_level} / {params.n_validation}")
    val_axis = 0 if n_models == 1 else 1
    idx_sign = ((pvalues < significativity_thr).sum(axis=val_axis) >= trust_level)
    if n_models > 1:
        idx_sign = idx_sign.sum(0) >= vote_prop * n_models
    modified_rois_names = [
        name.replace("&", "_").replace("-", "_") for name in rois_names]
    anova_pvalues = np.zeros((n_models, n_validation, n_scores, n_rois))
    if n_models == 1:
        all_coefs = all_coefs[np.newaxis]
    for model_idx in range(n_models):
        for val_idx in range(n_validation):
            for score_idx, score in enumerate(clinical_names):
                coefs = pd.DataFrame(
                    all_coefs[model_idx][val_idx][score_idx],
                    columns=["participant_id", "site"] + modified_rois_names)
                coefs[modified_rois_names] = coefs[modified_rois_names].astype(float)

                for roi_idx, name in enumerate(modified_rois_names):
                    anova_ols = sm.OLS.from_formula(
                        "{} ~ C(site)".format(name),
                        data=coefs).fit()
                    anova_res = anova_lm(anova_ols, type=2)
                    anova_pvalues[model_idx, val_idx, score_idx, roi_idx] = (
                        anova_res["PR(>F)"]["C(site)"])

    print_result(f"results ANOVA: {anova_pvalues.shape}")
    print(anova_pvalues.min())
    print(anova_pvalues.max())
    print(anova_pvalues.mean((0, 1)).min())
    print(anova_pvalues.mean((0, 1)).max())
    print(anova_pvalues[:, :, idx_sign].min())
    print(anova_pvalues[:, :, idx_sign].max())
    print(anova_pvalues[:, :, idx_sign].mean((0, 1)).min())
    print(anova_pvalues[:, :, idx_sign].mean((0, 1)).max())

def rsa_exp(dataset, datasetdir, outdir, run, n_validation=1, n_subjects=301,
            sample_latents=False):
    """ Perform Representational Similarity Analysis (RSA) on estimated
    latent representations.
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
    n_validation: int, default 50
        the number of times we repeat the experiments.
    n_subjects: int, default 50
        the number of samples for each clinical score used to compute the
        (dis)similarity matrices.
    seed: int, default 1037
        optionally specify a seed to control expriment reproducibility, set
        to None for randomization.
    """
    print_title(f"RSA ANALYSIS: {dataset}")
    expdir = os.path.join(outdir, run)
    rsadir = os.path.join(expdir, "rsa")
    if not os.path.isdir(rsadir):
        os.mkdir(rsadir)
    print_text(f"experimental directory: {expdir}")
    print_text(f"RSA directory: {rsadir}")

    print_subtitle("Loading data...")
    flags_file = os.path.join(expdir, "flags.rar")
    if not os.path.isfile(flags_file):
        raise ValueError("You need first to train the model.")
    checkpoints_dir = os.path.join(expdir, "checkpoints")
    experiment, flags = MultimodalExperiment.get_experiment(
        flags_file, checkpoints_dir)

    n_models = flags.num_models
    clinical_names = np.load(
        os.path.join(datasetdir, "clinical_names.npy"), allow_pickle=True)
    modalities = ["clinical", "rois"]
    print_text(f"modalities: {modalities}")
    cov_names = ["age", "sex", "site"]
    if dataset == "euaims":
        cov_names.append("fsiq")
    categorical_covs = ["sex", "site"]
    latent_names = ["joint", "clinical_rois", "clinical_style", "rois_style"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kendalltaus = np.zeros((n_models, len(latent_names), n_validation, len(clinical_names) + len(cov_names), 2))
    latent_disimilarities, scores_disimilarities = [], []
    for model_idx in range(n_models):
        trainset = experiment.dataset_train
        testset = experiment.dataset_test
        model = experiment.models
        if n_models > 1:
            trainset = trainset[model_idx]
            testset = testset[model_idx]
            model = model[model_idx]
        print_text(f"train data: {len(trainset)}")
        print_text(f"test data: {len(testset)}")

        print_subtitle("Compute blocks correlations using Kendall tau statstic...")
        cov_names = ["age", "sex", "site"]
        if dataset == "euaims":
            cov_names.append("fsiq")
        categorical_covs = ["sex", "site"]
        latent_names = ["joint", "clinical_rois", "clinical_style", "rois_style"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        latent_disimilarities.append([])
        scores_disimilarities.append([])
        for val_idx in tqdm(range(n_validation)):
            testloader = DataLoader(
                testset, batch_size=n_subjects, shuffle=True, num_workers=0)
            data = {}
            dataiter = iter(testloader)
            while True:
                data, _, metadata = next(dataiter)
                if all([mod in data.keys() for mod in modalities]):
                    break
            for idx, mod in enumerate(modalities):
                data[mod] = Variable(data[mod]).to(device).float()
            for column in metadata.keys():
                if type(metadata[column]) is list:
                    metadata[column] = np.array(metadata[column])
                else:
                    metadata[column] = metadata[column].cpu().detach().numpy()
            test_size = len(data[mod])
            for latent_idx, latent_name in enumerate(latent_names):
                latents = model(data, sample_latents=sample_latents)["latents"]
                if latent_name == "joint":
                    latents = latents["joint"]
                elif "style" in latent_name:
                    latents = latents["modalities"][latent_name]
                else:
                    latents = latents["subsets"][latent_name]
                if sample_latents:
                    latents = model.reparameterize(latents[0], latents[1])
                else:
                    latents = latents[0]
                if latents is not None:
                    latents = latents.cpu().detach().numpy()
                    print_text(f"latents: {latents.shape}")
                    n_scores = data["clinical"].shape[1]
                    n_subjects = len(latents)
                    cmat = data2cmat(latents)
                    latent_disimilarities[model_idx].append(cmat)
                    print_text(f"(dis)similarity matrix: {cmat.shape}")
                    scores_cmats = []
                    for score_idx in range(n_scores):
                        score_cmat = vec2cmat(data["clinical"][:, score_idx].detach().cpu())
                        scores_cmats.append(score_cmat)
                        tau, pval = fit_rsa(cmat, score_cmat)
                        kendalltaus[model_idx, latent_idx, val_idx, score_idx, 0] = tau
                        kendalltaus[model_idx, latent_idx, val_idx, score_idx, 1] = pval
                    for cov_idx, name in enumerate(cov_names):
                        score_cmat = vec2cmat(
                            metadata[name], categorical=name in categorical_covs)
                        scores_cmats.append(score_cmat)
                        tau, pval = fit_rsa(cmat, score_cmat)
                        kendalltaus[model_idx, latent_idx, val_idx, n_scores + cov_idx, 0] = tau
                        kendalltaus[model_idx, latent_idx, val_idx, n_scores + cov_idx, 1] = pval
                scores_cmats = np.asarray(scores_cmats)
                scores_disimilarities[model_idx].append(scores_cmats)
                print_text(f"scores (dis)similarity matricies: {scores_cmats.shape}")
    latent_disimilarities = np.asarray(latent_disimilarities)
    print_text(f"latent disimilarities: {latent_disimilarities.shape}")
    scores_disimilarities = np.asarray(scores_disimilarities)
    print_text(f"scores disimilarities: {scores_disimilarities.shape}")
    stats_file = os.path.join(rsadir, "kendalltau_stats.npy")
    np.save(stats_file, kendalltaus)
    print_result(f"kendall tau statistics: {stats_file}")
    latdis_file = os.path.join(rsadir, "latent_dissimilarity.npy")
    np.save(latdis_file, latent_disimilarities)
    print_result(f"latent disimilarities: {latdis_file}")
    scdis_file = os.path.join(rsadir, "scores_dissimilarity.npy")
    np.save(scdis_file, scores_disimilarities)
    print_result(f"scores_dissimilarity: {scdis_file}")

    print_subtitle("Summarize Kendall tau statstics...")
    for latent_idx, latent_name in enumerate(latent_names):
        data = {"score": [], "pval": [], "pval_std": [], "r": [], "r_std": []}
        for score_idx in range(n_scores):
            data["score"].append(clinical_names[score_idx])
            data["pval"].append(np.mean(kendalltaus[:, latent_idx, :, score_idx, 1]))
            data["pval_std"].append(np.std(kendalltaus[:, latent_idx, :, score_idx, 1]))
            data["r"].append(np.mean(kendalltaus[:, latent_idx, :, score_idx, 0]))
            data["r_std"].append(np.std(kendalltaus[:, latent_idx, :, score_idx, 0]))
        for cov_idx, cov_name in enumerate(cov_names):
            data["score"].append(cov_name)
            data["pval"].append(np.mean(kendalltaus[:, latent_idx, :, n_scores + cov_idx, 1]))
            data["pval_std"].append(np.std(kendalltaus[:, latent_idx, :, n_scores + cov_idx, 1]))
            data["r"].append(np.mean(kendalltaus[:, latent_idx, :, n_scores + cov_idx, 0]))
            data["r_std"].append(np.std(kendalltaus[:, latent_idx, :, n_scores + cov_idx, 0]))
        df = pd.DataFrame.from_dict(data)
        summary_file = os.path.join(rsadir, f"kendalltau_{latent_name}.tsv")
        df.to_csv(summary_file, sep="\t", index=False)
        print_result(f"kendall tau summary: {summary_file}")
        print(df.groupby(["score"]).apply(lambda e: e[:]))


def score_models(dataset, datasetdir, outdir, run, scores=None, latent_name="joint"):
    """ Perform Representational Similarity Analysis (RSA) on estimated
    latent representations.
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
    n_validation: int, default 50
        the number of times we repeat the experiments.
    n_subjects: int, default 50
        the number of samples for each clinical score used to compute the
        (dis)similarity matrices.
    seed: int, default 1037
        optionally specify a seed to control expriment reproducibility, set
        to None for randomization.
    """
    import matplotlib.pyplot as plt
    expdir = os.path.join(outdir, run)
    rsadir = os.path.join(expdir, "rsa")

    if not os.path.exists(rsadir):
        raise ValueError("You must first run rsa n your models to score them.")
    
    clinical_names = np.load(
        os.path.join(datasetdir, "clinical_names.npy"), allow_pickle=True)
    clinical_names = clinical_names.tolist()
    if scores is None:
        scores = clinical_names
    
    kendalltaus = np.load(os.path.join(rsadir, "kendalltau_stats.npy"))
    score_indices = [clinical_names.index(score) for score in scores]
    latent_names = ["joint", "clinical_rois", "clinical_style", "rois_style"]
    score_per_model = kendalltaus[:, :, :, score_indices, 0]
    score_per_model = score_per_model.mean(axis=(-2, -1))
    score_per_model = score_per_model[:, latent_names.index(latent_name)]
    # ordered_score = np.argsort(score_per_model)
    # for n_worst in range(0, 20):
    #     print(f"{n_worst}th worst score : {score_per_model[ordered_score[n_worst]]}")
    # plt.hist(score_per_model, bins=15)
    # plt.show()
    return score_per_model


def hist_plot_exp(datasets, datasetdirs, scores, outdir):
    """ Display specified score histogram across different cohorts.
    Parameters
    ----------
    datasets: str
        the dataset names.
    datasetdir: list of str
        the path to the datasets associated data.
    scores: list of str
        the scores in each cohort to be plotted.
    outdir: str
        the destination folder.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    print_title("PLOT HISTOGRAM")
    if not isinstance(datasetdirs, list):
        datasetdirs = datasetdirs.split(",")
    assert len(datasets) == len(datasetdirs), "Invalid input list sizes."
    assert len(datasets) == len(scores), "Invalid input list sizes."
    print_text(f"datasets: {datasets}")
    print_text(f"dataset directories: {datasetdirs}")
    print_text(f"scores: {scores}")
    data = {"score": [], "cohort": []}
    for name, path, score in zip(datasets, datasetdirs, scores):
        clinical_data = np.load(os.path.join(path, "clinical_data.npy"),
                                 allow_pickle=True).T
        clinical_names = np.load(os.path.join(path, "clinical_names.npy"),
                                 allow_pickle=True)
        print_text(f"clinical data {name}: {clinical_data.shape}")
        score_idx = np.argwhere(clinical_names == score)[0, 0]
        data["score"].extend(clinical_data[score_idx].tolist())
        data["cohort"].extend([name] * clinical_data.shape[1])
    plt.figure(figsize=(10, 3/4 * 10))
    label = list(data.keys())
    sns_plot = sns.kdeplot(
       data=data, x="score", hue="cohort", fill=True, common_norm=False,
       linewidth=0, multiple="stack")
    ax = plt.gca()
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    fig = sns_plot.get_figure()
    hist_file = os.path.join(outdir, "hist.png")
    fig.savefig(hist_file)
    print_result(f"histogram: {hist_file}")


def rsa_plot_exp(dataset, datasetdir, outdir, run):
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
    from plotting import plot_mosaic

    print_title(f"PLOT RSA results: {dataset}")
    expdir = os.path.join(outdir, run)
    rsadir = os.path.join(expdir, "rsa")
    print_text(f"experimental directory: {expdir}")
    print_text(f"RSA directory: {rsadir}")
    latent_cmat = np.load(os.path.join(rsadir, "latent_dissimilarity.npy"))
    scores_cmat = np.load(os.path.join(rsadir, "scores_dissimilarity.npy"))
    print_text(f"latent dissimilarity: {latent_cmat.shape}")
    print_text(f"scores dissimilarity: {scores_cmat.shape}")
    cmat_file = os.path.join(rsadir, "dissimilarity.png")
    cmat1 = latent_cmat[:1]
    cmat1 /= cmat1.max()
    cmat2 = scores_cmat[0]
    cmat2 /= cmat2.max()
    images = np.concatenate((cmat1, cmat2), axis=0)
    plot_mosaic(images, cmat_file, n_cols=4, image_size=images.shape[-2:])


def daa_plot_most_connected(dataset, datasetdir, outdir, run, trust_level=0.7,
                            n_rois=5, plot_associations=False, vote_prop=1,
                            rescaled=True):
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

    print_title(f"PLOT DAA most associated rois: {dataset}")
    expdir = os.path.join(outdir, run)
    daadir = os.path.join(expdir, "daa")
    print_text(f"experimental directory: {expdir}")
    print_text(f"DAA directory: {daadir}")
    simdirs = [path for path in glob.glob(os.path.join(daadir, "*"))
               if os.path.isdir(path)]
    print_text(f"Simulation directories: {','.join(simdirs)}")

    flags_file = os.path.join(expdir, "flags.rar")
    if not os.path.isfile(flags_file):
        raise ValueError("You need first to train the model.")
    checkpoints_dir = os.path.join(expdir, "checkpoints")
    experiment, flags = MultimodalExperiment.get_experiment(
        flags_file, checkpoints_dir)

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
                                      rois_names=rois_names)

    rois_names = rois_names.tolist()
    n_models = flags.num_models
    scalers = experiment.scalers

    marker_signif = "star"
    marker_non_signif = "circle"
    for dirname in simdirs:
        print_subtitle(f"Drawing radar plots for {dirname}")
        if not os.path.exists(os.path.join(dirname, "coefs.npy")):
            continue
        coefs = np.load(os.path.join(dirname, "coefs.npy"))
        pvalues = np.load(os.path.join(dirname, "pvalues.npy"))

        n_validation = int(
            dirname.split("n_validation_")[1].split("_n_s")[0])

        idx_sign, df = compute_significativity(
            pvalues, trust_level, vote_prop, n_validation, additional_data,
            correct_threshold=True)

        print_subtitle(f"Plot regression coefficients radar plots...")
        counts = collections.Counter(df["roi"].values)
        # selected_rois = [item[0] for item in counts.most_common()]
        n_colors = n_rois * len(df["metric"].unique())
        color_name = "Plotly"
        if n_colors > 9:
            color_name = "Paired"
        if n_colors > 12:
            color_name = "tab20"
        textfont = dict(
            size=20,
            family="Droid Serif")
        colors = get_color_list(color_name, n_colors)
        all_selected_rois = []
        for _metric, _df in df.groupby(["metric"]):
            selected_scores = []
            significativity = []
            counts = collections.Counter(_df["roi"].values)
            selected_rois = [item[0] for item in counts.most_common(n_rois)]
            for _roi in selected_rois:
                roi_idx = rois_names.index(f"{_roi}_{_metric}")
                #if n_models > 1:
                selected_coefs = coefs[:, :, :, roi_idx].mean(axis=(0, 1))
                # else:
                #    selected_coefs = coefs[:, :, roi_idx].mean(axis=0)
                selected_scores.append(selected_coefs)
                significativity.append(idx_sign[:, roi_idx].tolist())
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
            for marker, name, sign in [
                (marker_non_signif, "non significative", False),
                (marker_signif, "significative", True)]:
                significative_scores = []
                score_names = []
                markers = []
                for roi_idx, roi_coefs in enumerate(selected_scores):
                    for coef_idx, coef in enumerate(roi_coefs):
                        if significativity[roi_idx][coef_idx] == sign:
                            significative_scores.append(coef)
                            score_names.append(clinical_names[coef_idx])
                            markers.append(marker)
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
                    marker_color="black",
                    name=name
                ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True, showticklabels=False, ticks="",
                        range=[0, np.array(selected_scores).max() + 0.003])),
                font=textfont)
            filename = os.path.join(
                dirname, f"three_selected_rois_{_metric}_polarplots.png")
            fig.write_image(filename)
            print_result(f"{_metric} regression coefficients for 3 selected "
                         f"ROIs: {filename}")
    
        filename = os.path.join(dirname, "most_connected_rois.png")
        plot_areas(all_selected_rois, np.arange(len(all_selected_rois)), filename, color_name)

        # print_subtitle(f"Plot significant ROIs per score...")
        # data, titles = [], []
        # for (_metric, _score), _df in df.groupby(["metric", "score"]):
        #     roi_indices = {"left": [], "right": []}
        #     for name in _df["roi"].values:
        #         roi_name, hemi = name.rsplit("_", 1)
        #         hemi = "left" if hemi == "lh" else "right"
        #         roi_indices[hemi].append(destrieux_labels.index(roi_name))
        #     parcellations = []
        #     for hemi, indices in roi_indices.items():
        #         _par = destrieux_atlas[f"map_{hemi}"]
        #         parcellations.append(np.isin(_par, indices).astype(int))
        #     data.append(parcellations)
        #     titles.append(f"{_metric} - {_score}")
        # filename = os.path.join(dirname, "most_significant_rois.png")
        # plot_surf_mosaic(data, titles, fsaverage, filename, label=True)

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
                    significant_coef = coefs[:, :, score_idx, roi_idx].mean()
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
                    dirname, f"score2roi_{_metric}_flow.png")
                fig.write_image(filename)
                print_result(f"flow for the {_metric} metric: {filename}")


def daa_plot_most_significant(dataset, datasetdir, outdir, run, n_rois=5,
                              use_coefficients=False):
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

    print_title(f"PLOT DAA most associated rois: {dataset}")
    expdir = os.path.join(outdir, run)
    daadir = os.path.join(expdir, "daa")
    print_text(f"experimental directory: {expdir}")
    print_text(f"DAA directory: {daadir}")
    simdirs = [path for path in glob.glob(os.path.join(daadir, "*"))
               if os.path.isdir(path)]
    print_text(f"Simulation directories: {','.join(simdirs)}")

    flags_file = os.path.join(expdir, "flags.rar")
    if not os.path.isfile(flags_file):
        raise ValueError("You need first to train the model.")
    checkpoints_dir = os.path.join(expdir, "checkpoints")
    # experiment, flags = MultimodalExperiment.get_experiment(
    #     flags_file, checkpoints_dir)

    clinical_names = np.load(
        os.path.join(datasetdir, "clinical_names.npy"), allow_pickle=True)
    clinical_names = clinical_names.tolist()
    rois_names = np.load(
        os.path.join(datasetdir, "rois_names.npy"), allow_pickle=True)
    metadata = pd.read_table(
        os.path.join(datasetdir, "metadata_train.tsv"))
    metadata_columns = metadata.columns.tolist()

    # additional_data = SimpleNamespace(metadata_columns=metadata_columns,
    #                                   clinical_names=clinical_names,
    #                                   rois_names=rois_names)

    rois_names = rois_names.tolist()

    marker_signif = "star"
    marker_non_signif = "circle"
    for dirname in simdirs:
        print_subtitle(f"Drawing radar plots for {dirname}")
        if not os.path.exists(os.path.join(dirname, "coefs.npy")):
            continue
        coefs = np.load(os.path.join(dirname, "coefs.npy"))
        pvalues = np.load(os.path.join(dirname, "pvalues.npy"))

        # n_validation = int(
        #     dirname.split("n_validation_")[1].split("_n_s")[0])

        # idx_sign, df = compute_significativity(
        #     pvalues, trust_level, vote_prop, n_validation, additional_data,
        #     correct_threshold=True)

        print_subtitle(f"Plot regression coefficients radar plots...")
        # counts = collections.Counter(df["roi"].values)
        # selected_rois = [item[0] for item in counts.most_common()]
        n_colors = n_rois * 3
        color_name = "Plotly"
        if n_colors > 9:
            color_name = "Paired"
        if n_colors > 12:
            color_name = "tab20"
        textfont = dict(
            size=20,
            family="Droid Serif")
        colors = get_color_list(color_name, n_colors)
        all_selected_rois = []
        average_pvalues = pvalues.mean((0, 1))
        std_pvalues = pvalues.std((0, 1))
        average_coefs = coefs.mean((0, 1))
        std_coefs = coefs.std((0, 1))

        rois = np.array(list(set([name.rsplit("_", 1)[0] for name in rois_names])))

        for _metric in ["thickness", "meancurv", "area"]:
            selected_coefs = []
            metric_indices = np.array(
                [rois_names.index(f"{_roi}_{_metric}") for _roi in rois])
            if use_coefficients:
                most_significant_indices = np.argsort(
                    np.absolute(average_coefs[:, metric_indices]), axis=None)[::-1]
            else:
                most_significant_indices = np.argsort(
                    average_pvalues[:, metric_indices], axis=None)

            rois_indices = np.array([idx % len(rois) for idx in 
                                     most_significant_indices[:n_rois * 5]])
            _, unique_idx = np.unique(rois_indices, return_index=True)
            rois_indices = rois_indices[np.sort(unique_idx)[:n_rois]]
            selected_rois = rois[roi_indices]
            all_selected_rois += [area for area in selected_rois
                                  if area not in all_selected_rois]
            selected_coefs = np.absolute(average_coefs[:, metric_indices[roi_indices]])
            selected_coefs_std = std_coefs[:, metric_indices[roi_indices]]

            fig = go.Figure()
            for roi_idx, _roi in enumerate(selected_rois):
                color_idx = all_selected_rois.index(_roi)
                color = plt_to_plotly_rgb(colors[color_idx])
                _scores = selected_coefs[:, roi_idx].tolist()
                _scores_pstd = (selected_coefs + selected_coefs_std)[:, roi_idx].tolist()
                _scores_mstd = (selected_coefs - selected_coefs_std)[:, roi_idx].tolist()
                fig.add_trace(
                    go.Scatterpolar(
                        r=_scores + _scores[:1],
                        theta=[
                            "<b>" + short_clinical_names[dataset][name] + "</b>"
                            for name in clinical_names + clinical_names[:1]],
                        mode="lines+text",
                        marker_color=color,
                        line_width=2,
                        legendgroup="roi",
                        legendgrouptitle = dict(
                            font=dict(
                                size=textfont["size"] + 4,
                                family=textfont["family"]),
                            text="<b>ROIs</b>"),
                        name=_roi))
                fig.add_trace(
                    go.Scatterpolar(
                        r=_scores_pstd + _scores_pstd[:1],
                        theta=[
                            "<b>" + short_clinical_names[dataset][name] + "</b>"
                            for name in clinical_names + clinical_names[:1]],
                        mode="lines",
                        marker_color=color,
                        line_width=1,
                        line_dash="dot",
                        showlegend=False
                    ))
                        # legendgroup="roi",
                        # legendgrouptitle = dict(
                        #     font=dict(
                        #         size=textfont["size"] + 4,
                        #         family=textfont["family"]),
                        #     text="<b>ROIs</b>"),
                        # name=_roi))
                fig.add_trace(
                    go.Scatterpolar(
                        r=_scores_mstd + _scores_mstd[:1],
                        theta=[
                            "<b>" + short_clinical_names[dataset][name] + "</b>"
                            for name in clinical_names + clinical_names[:1]],
                        mode="lines",
                        marker_color=color,
                        line_width=1,
                        line_dash="dot",
                        showlegend=False
                    ))
                        # legendgroup="roi",
                        # legendgrouptitle = dict(
                        #     font=dict(
                        #         size=textfont["size"] + 4,
                        #         family=textfont["family"]),
                        #     text="<b>ROIs</b>"),
                        # name=_roi))
            # for marker, name, sign in [
            #     (marker_non_signif, "non significative", False)]:
            #     (marker_signif, "significative", True)]:
            #     significative_scores = []
            #     score_names = []
            #     markers = []
            #     for roi_idx, roi_coefs in enumerate(selected_coefs):
            #         for coef_idx, coef in enumerate(roi_coefs):
            #             if significativity[roi_idx][coef_idx] == sign:
            #     for roi_idx, _roi in enumerate(selected_rois):
            #         _scores = selected_coefs[:, roi_idx].tolist()
            #             significative_scores.append(coef)
            #         score_names.append(clinical_names)
            #         markers.append(marker)
            #         fig.add_trace(go.Scatterpolar(
            #             r=np.array(_scores),
            #             theta=np.array(["<b>" + short_clinical_names[dataset][name]
            #                             + "</b>" for name in clinical_names]),
            #             fill='toself',
            #             mode="markers",
            #             legendgroup="significativity",
            #             legendgrouptitle = dict(
            #                 font=dict(
            #                     size=textfont["size"] + 4,
            #                     family="Droid Serif"),
            #                 text="<b>Significativity</b>"),
            #             showlegend=False,
            #             marker_symbol=marker,
            #             marker_size=5,
            #             marker_color="black",
            #             name=name
            #         ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True, showticklabels=False, ticks="",
                        range=[0, (selected_coefs + selected_coefs_std).max() + 0.0005])),#(selected_coefs).max() + 0.003])),# (selected_coefs+ selected_coefs_std).max() + 0.003])),
                font=textfont)
            filename = os.path.join(
                dirname, f"most_sigificant_rois_{_metric}_polarplots.png")
            fig.write_image(filename)
            print_result(f"{_metric} regression coefficients for most significant "
                         f"ROIs: {filename}")
    
        filename = os.path.join(dirname, "most_significant_rois.png")
        plot_areas(all_selected_rois, np.arange(len(all_selected_rois)), filename, color_name)


def daa_plot_score_metric(dataset, datasetdir, outdir, run, score, metric,
                          trust_level=0.7, vote_prop=1, rescaled=True):
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
    expdir = os.path.join(outdir, run)
    daadir = os.path.join(expdir, "daa")
    print_text(f"experimental directory: {expdir}")
    print_text(f"DAA directory: {daadir}")
    simdirs = [path for path in glob.glob(os.path.join(daadir, "*"))
               if os.path.isdir(path)]
    print_text(f"Simulation directories: {','.join(simdirs)}")

    flags_file = os.path.join(expdir, "flags.rar")
    if not os.path.isfile(flags_file):
        raise ValueError("You need first to train the model.")
    checkpoints_dir = os.path.join(expdir, "checkpoints")
    experiment, flags = MultimodalExperiment.get_experiment(
        flags_file, checkpoints_dir)

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
                                      rois_names=rois_names)
    rois_names = rois_names.tolist()
    significativity_thr = 0.05 / len(clinical_names) / len(rois_names)
    n_models = flags.num_models
    scalers = experiment.scalers

    for dirname in simdirs:
        print_text(dirname)
        if not os.path.exists(os.path.join(dirname, "coefs.npy")):
            continue
        coefs = np.load(os.path.join(dirname, "coefs.npy"))
        pvalues = np.load(os.path.join(dirname, "pvalues.npy"))

        n_validation = int(
            dirname.split("n_validation_")[1].split("_n_s")[0])
        _, df = compute_significativity(
            pvalues, trust_level, vote_prop, n_validation, additional_data,
            correct_threshold=True)

        areas = df["roi"][(df["metric"] == metric) & (df["score"] == score)].to_list()
        area_idx = [rois_names.index(f"{name}_{metric}") for name in areas]
        score_idx = clinical_names.index(score)
        values = coefs[:, :, score_idx, area_idx].mean(axis=(0, 1))
        if rescaled:
            scaling_factors = []
            for roi_idx in area_idx:
                if n_models > 1:
                    scaling_factor = sum([
                        scalers[i]["rois"].scale_[roi_idx] /
                        scalers[i]["clinical"].scale_[score_idx]
                        for i in range(n_models)]) / n_models
                else:
                    scaling_factor = (scalers["rois"].scale_[roi_idx] /
                        scalers["clinical"].scale_[score_idx])
                scaling_factors.append(scaling_factor)
            scaling_factors = np.asarray(scaling_factors)
            values *= scaling_factors

        print_subtitle(f"Plot regression coefficients ...")
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
        filename_areas = os.path.join(
            dirname, f"associated_rois_for_{score}_in_{metric}.png")
        filename_bar = os.path.join(
            dirname, f"association_for_{score}_in_{metric}.png")
        plt.rcParams.update({'font.size': 20, "font.family": "serif"})
        plot_areas(areas, np.arange(len(areas)) + 0.01, filename_areas, color_name)
        plot_coefs(areas, values, filename=filename_bar, color_name=color_name)


def daa_plot_score_metric_strongest(dataset, datasetdir, outdir, run, score,
                                    metric, rescaled=True, n_rois=10,
                                    use_coefficients=False):
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
    expdir = os.path.join(outdir, run)
    daadir = os.path.join(expdir, "daa")
    print_text(f"experimental directory: {expdir}")
    print_text(f"DAA directory: {daadir}")
    simdirs = [path for path in glob.glob(os.path.join(daadir, "*"))
               if os.path.isdir(path)]
    print_text(f"Simulation directories: {','.join(simdirs)}")

    flags_file = os.path.join(expdir, "flags.rar")
    if not os.path.isfile(flags_file):
        raise ValueError("You need first to train the model.")
    checkpoints_dir = os.path.join(expdir, "checkpoints")
    experiment, flags = MultimodalExperiment.get_experiment(
        flags_file, checkpoints_dir)

    clinical_names = np.load(
        os.path.join(datasetdir, "clinical_names.npy"), allow_pickle=True)
    clinical_names = clinical_names.tolist()
    rois_names = np.load(
        os.path.join(datasetdir, "rois_names.npy"), allow_pickle=True)
    metadata = pd.read_table(
        os.path.join(datasetdir, "metadata_train.tsv"))
    metadata_columns = metadata.columns.tolist()

    # additional_data = SimpleNamespace(metadata_columns=metadata_columns,
    #                                   clinical_names=clinical_names,
    #                                   rois_names=rois_names)
    rois_names = rois_names.tolist()
    significativity_thr = 0.05 / len(clinical_names) / len(rois_names)
    n_models = flags.num_models
    scalers = experiment.scalers

    for dirname in simdirs:
        print_text(dirname)
        if not os.path.exists(os.path.join(dirname, "coefs.npy")):
            continue
        coefs = np.load(os.path.join(dirname, "coefs.npy"))
        pvalues = np.load(os.path.join(dirname, "pvalues.npy"))

        # n_validation = int(
        #     dirname.split("n_validation_")[1].split("_n_s")[0])
        # _, df = compute_significativity(
        #     pvalues, trust_level, vote_prop, n_validation, additional_data,
        #     correct_threshold=True)

        average_pvalues = pvalues.mean((0, 1))
        std_pvalues = pvalues.std((0, 1))
        average_coefs = coefs.mean((0, 1))
        std_coefs = coefs.std((0, 1))

        rois = np.array(
            list(set([name.rsplit("_", 1)[0] for name in rois_names])))

        score_idx = clinical_names.index(score)
        metric_indices = np.array(
            [roi_idx for roi_idx, name in enumerate(rois_names)
             if metric in name])
        if use_coefficients:
            most_significant_indices = np.argsort(
                np.absolute(average_coefs[score_idx, metric_indices]))[::-1]
        else:
            most_significant_indices = np.argsort(
                average_pvalues[score_idx, metric_indices])

        rois_indices = most_significant_indices[:n_rois]
        area_idx = metric_indices[rois_indices]
        areas = np.array(rois_names)[area_idx]
        areas = [area.rsplit("_", 1)[0] for area in areas]
        values = average_coefs[score_idx, area_idx]
        stds = std_coefs[score_idx, area_idx]

        if rescaled:
            scaling_factors = []
            for roi_idx in area_idx:
                if n_models > 1:
                    scaling_factor = sum([
                        scalers[i]["rois"].scale_[roi_idx] /
                        scalers[i]["clinical"].scale_[score_idx]
                        for i in range(n_models)]) / n_models
                else:
                    scaling_factor = (scalers["rois"].scale_[roi_idx] /
                        scalers["clinical"].scale_[score_idx])
                scaling_factors.append(scaling_factor)
            scaling_factors = np.asarray(scaling_factors)
            values *= scaling_factors
            stds *= scaling_factors

        print_subtitle(f"Plot regression coefficients ...")
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
        filename_areas = os.path.join(
            dirname, f"strongest_associated_rois_for_{score}_in_{metric}.png")
        filename_bar = os.path.join(
            dirname, f"strongest_association_for_{score}_in_{metric}.png")
        plt.rcParams.update({'font.size': 20, "font.family": "serif"})
        plot_areas(areas, np.arange(len(areas)) + 0.01, filename_areas, color_name)
        plot_coefs(areas, values, stds, filename_bar, color_name)


def avatar_plot_exp(dataset, datasetdir, outdir, run):
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
    from surfify.utils import text2grid
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from nilearn import datasets
    from nilearn.surface import load_surf_mesh
    import plotly.graph_objects as go

    print_title(f"PLOT AVATARS: {dataset}")
    subject_idx = 0
    score_idx = 0
    expdir = os.path.join(outdir, run)
    daadir = os.path.join(expdir, "daa")
    print_text(f"experimental directory: {expdir}")
    print_text(f"DAA directory: {daadir}")
    simdirs = [path for path in glob.glob(os.path.join(daadir, "*"))
               if os.path.isdir(path)]
    print_text(f"Simulation directories: {','.join(simdirs)}")

    destrieux_atlas = datasets.fetch_atlas_surf_destrieux(data_dir=expdir)
    destrieux_labels = [label.decode().replace("_and_", "&")
                        for label in destrieux_atlas["labels"]]
    fsaverage = datasets.fetch_surf_fsaverage(data_dir=expdir)
    spheres = {
        "left": load_surf_mesh(fsaverage["sphere_left"])[0],
        "right": load_surf_mesh(fsaverage["sphere_right"])[0]
    }
    clinical_names = np.load(
        os.path.join(datasetdir, "clinical_names.npy"), allow_pickle=True)
    clinical_names = clinical_names.tolist()
    rois_names = np.load(
        os.path.join(datasetdir, "rois_names.npy"), allow_pickle=True)
    rois_names = rois_names.tolist()
    for dirname in simdirs:
        avatardir = os.path.join(dirname, "avatar")
        if not os.path.isdir(avatardir):
            os.mkdir(avatardir)
        print_text(f"avatar directory: {avatardir}")
        errors = np.load(os.path.join(dirname, "rec_errors.npy"))
        errors = errors.mean(axis=0)
        traverses = np.load(os.path.join(dirname, "trav_scores.npy"))
        traverses = traverses.mean(axis=0).transpose(0, 2, 1)
        n_subjects, n_scores, n_steps = traverses.shape
        print(f"reconstruction errors: {errors.shape}")
        print(f"traverses: {traverses.shape}")
        roi_indices = {"roi": [], "idx": [], "atlas_idx": [], "hemi": [],
                       "metric": [], "hemi": []}
        for roi_idx, name in enumerate(rois_names):
            roi_name, hemi, metric_name = name.rsplit("_", 2)
            hemi = "left" if hemi == "lh" else "right"
            roi_indices["hemi"].append(hemi)
            roi_indices["roi"].append(roi_name)
            roi_indices["metric"].append(metric_name)
            roi_indices["idx"].append(roi_idx)
            roi_indices["atlas_idx"].append(destrieux_labels.index(roi_name))
        df = pd.DataFrame.from_dict(roi_indices)
        print(df.groupby(["metric"]).apply(lambda e: e[:]))

        for _metric, _df1 in df.groupby(["metric"]):
            fig, axs = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 10]})
            plt.axis("off")
            animate_data = []
            vmin = errors[subject_idx, score_idx, :, _df1["idx"].values].min()
            vmax = errors[subject_idx, score_idx, :, _df1["idx"].values].max()
            pertubations = traverses[subject_idx, score_idx]
            peturbations_order = np.argsort(pertubations)
            peturbations_vmin = pertubations.min()
            peturbations_vmax = pertubations.max()
            tot_range = peturbations_vmax - peturbations_vmin
            for _idx in tqdm(range(n_steps)):
                step_idx = peturbations_order[_idx]
                images = []
                for _hemi, _df2 in _df1.groupby(["hemi"]):
                    _par_ref = destrieux_atlas[f"map_{hemi}"]
                    _par = destrieux_atlas[f"map_{hemi}"].astype(float) * 0.
                    for roi_idx, atlas_idx in zip(_df2["idx"], _df2["atlas_idx"]):
                        _par[_par_ref == atlas_idx] = errors[
                            subject_idx, score_idx, step_idx,  roi_idx]
                    proj_texture = text2grid(spheres[_hemi], _par)
                    images.append(proj_texture)
                view = np.concatenate(images, axis=1)
                perturbation = pertubations[step_idx]
                progress = perturbation - peturbations_vmin
                animate_data.append([view, progress])

            rectangles = axs[0].bar([0], [tot_range], width=0.1)
            axs[0].set_ylabel("Score")
            axs[0].spines["top"].set_visible(False)
            axs[0].spines["right"].set_visible(False)
            axs[0].spines["bottom"].set_visible(False)
            axs[0].set_xticks([])
            labels = [item.get_text() for item in axs[0].get_yticklabels()]
            labels[0] = round(peturbations_vmin, 2)
            labels[-2] = round(peturbations_vmax, 2)
            axs[0].set_yticklabels(labels)
            im = axs[1].imshow(animate_data[0][0], vmin=vmin, vmax=vmax,
                               aspect="equal",  cmap="jet")
            cbar_ax = fig.add_axes([0.27, 0.2, 0.6, 0.04])
            fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
            patches = list(rectangles) + [im]

            def init():
                rectangles[0].set_height(0)
                return patches

            def animate(idx):
                rectangles[0].set_height(animate_data[idx][1])
                im.set_array(animate_data[idx][0])
                return patches

            ani = animation.FuncAnimation(fig, animate, init_func=init,
                                          frames=len(animate_data),
                                          interval=50, blit=True)
            writer = animation.FFMpegWriter(fps=15, bitrate=1800)
            filename = os.path.join(
                avatardir, 
                f"sub-{subject_idx}_score-{score_idx}_metric-{_metric}.mp4")
            ani.save(filename, writer=writer)
            print_result(f"movie: {filename}")
