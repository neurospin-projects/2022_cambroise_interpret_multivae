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
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm
from types import SimpleNamespace
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from run_epochs import run_epochs
from multimodal_cohort.flags import parser
from utils.filehandling import create_dir_structure
from multimodal_cohort.experiment import MultimodalExperiment
from multimodal_cohort.dataset import DataManager, MissingModalitySampler
from stat_utils import data2cmat, vec2cmat, fit_rsa
from color_utils import (
    print_title, print_subtitle, print_command, print_text, print_result)


def train_exp(dataset, datasetdir, outdir, input_dims, latent_dim=20,
              num_hidden_layers=1, allow_missing_blocks=False, beta=5.,
              likelihood="normal", initial_learning_rate=0.002, batch_size=256,
              n_epochs=2500, eval_freq=25, eval_freq_fid=100,
              data_multiplications=1, dropout_rate=0., initial_out_logvar=-3.,
              learn_output_scale=False):
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
    latent_dim: int, default 20
        dimension of common factor latent space.
    num_hidden_layers: int, default 1
        number of hidden laters in the model.
    allow_missing_blocks: bool, default False
        optionally, allows for missing modalities.
    beta: float, default 5
        default weight of sum of weighted divergence terms.
    likelihood: str, default 'normal'
        output distribution.
    initial_learning_rate: float, default 0.002
        starting learning rate.
    batch_size: int, default 256
        batch size for training.
    n_epochs: int, default 2500
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
    learn_output_scale: bool, default False
        optionally, allows for different scales per feature.
    """
    print_title(f"TRAIN: {dataset}")
    flags = SimpleNamespace(
        dataset=dataset, datasetdir=datasetdir, dropout_rate=dropout_rate,
        allow_missing_blocks=allow_missing_blocks, batch_size=batch_size,
        beta=beta, beta_1=0.9, beta_2=0.999, beta_content=1.0,
        beta_style=1.0, calc_nll=False, calc_prd=False,
        class_dim=latent_dim, data_multiplications=data_multiplications,
        dim=64, dir_data="../data", dir_experiment=outdir, dir_fid=None,
        div_weight=None, div_weight_uniform_content=None,
        end_epoch=n_epochs, eval_freq=eval_freq, eval_freq_fid=eval_freq_fid,
        factorized_representation=False, img_size_m1=28, img_size_m2=32,
        inception_state_dict="../inception_state_dict.pth",
        initial_learning_rate=initial_learning_rate,
        initial_out_logvar=initial_out_logvar, input_dim=input_dims,
        joint_elbo=False, kl_annealing=0, include_prior_expert=False,
        learn_output_scale=learn_output_scale, len_sequence=8,
        likelihood=likelihood, load_saved=False, method='joint_elbo',
        mm_vae_save="mm_vae", modality_jsd=False, modality_moe=False,
        modality_poe=False, num_channels_m1=1, num_channels_m2=3,
        num_classes=2, num_hidden_layers=num_hidden_layers,
        num_samples_fid=10000, num_training_samples_lr=500,
        poe_unimodal_elbos=True, save_figure=False, start_epoch=0, style_dim=0,
        subsampled_reconstruction=True)
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

    alphabet_path = os.path.join(os.getcwd(), "alphabet.json")
    with open(alphabet_path) as alphabet_file:
        alphabet = str("".join(json.load(alphabet_file)))
    mst = MultimodalExperiment(flags, alphabet)
    mst.set_optimizer()
    run_epochs(mst)


def daa_exp(dataset, datasetdir, outdir, run, linear_gradient=False,
            n_validation=50, n_discretization_steps=200, n_samples=50,
            k=1000, trust_level=0.75, seed=1037):
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
    linear_gradient: bool, default False
        optionally build the traverses min/max bound for each clinical score
        from K samples in the latent space, otherwise define these bounds from
        the clinical table itself.
    n_validation: int, default 50
        the number of times we repeat the experiments.
    n_discretization_steps: int, default 200
        the size of each traverse.
    n_samples: int, default 50
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
    alphabet_file = os.path.join(os.getcwd(), "alphabet.json")
    checkpoints_files = glob.glob(
        os.path.join(expdir, "checkpoints", "*", "mm_vae"))
    if len(checkpoints_files) == 0:
        raise ValueError("You need first to train the model.")
    checkpoints_files = sorted(
        checkpoints_files, key=lambda path: int(path.split(os.sep)[-2]))
    checkpoint_file = checkpoints_files[-1]
    print_text(f"restoring weights: {checkpoint_file}")
    experiment, flags = MultimodalExperiment.get_experiment(
        flags_file, alphabet_file, checkpoint_file)
    model = experiment.mm_vae
    print(model)
    clinical_names = np.load(
        os.path.join(datasetdir, "clinical_names.npy"), allow_pickle=True)
    rois_names = np.load(
        os.path.join(datasetdir, "rois_names.npy"), allow_pickle=True)
    modalities = ["clinical", "rois"]
    print_text(f"modalities: {modalities}")
    trainset = experiment.dataset_train
    print_text(f"train data: {len(trainset)}")
    if flags.allow_missing_blocks:
        trainsampler = MissingModalitySampler(
            trainset, batch_size=len(trainset))
        trainloader = DataLoader(
            trainset, batch_sampler=trainsampler, num_workers=0)
    else:
        trainloader = DataLoader(
            trainset, shuffle=True, batch_size=len(trainset), num_workers=0)
    testset = experiment.dataset_test
    print_text(f"test data: {len(testset)}")
    if flags.allow_missing_blocks:
        testsampler = MissingModalitySampler(testset, batch_size=len(testset))
        testloader = DataLoader(
            testset, batch_sampler=testsampler, num_workers=0)
    else:
        testloader = DataLoader(
            testset, shuffle=True, batch_size=len(testset), num_workers=0)

    print_subtitle("Evaluate model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    _data = {}
    with torch.set_grad_enabled(False):
        for phase, loader in zip(("train", "test"), (trainloader, testloader)):
            dataiter = iter(loader)
            while True:
                data, labels, _ = dataiter.next()
                if all([mod in data.keys() for mod in modalities]):
                    break
            for idx, mod in enumerate(modalities):
                data[mod] = Variable(data[mod]).to(device).float()
            _data[f"z{phase}"] = model.inference(data)
            _data[f"data{phase}"] = data
    latents = SimpleNamespace(**_data)
    print_text(f"z train: {latents.ztrain['mus'].shape}")
    print_text(f"z test: {latents.ztest['mus'].shape}")
    subsets = list(latents.ztest["subsets"])
    print_text(f"subsets: {subsets}")

    print_subtitle("Create digital avatars models using clinical traverses...")
    if linear_gradient:
        print_text("Build the traverses min/max bound for each clinical "
                   "score from the clinical table itself.")
    else:
        print_text("Build the traverses min/max bound for each clinical "
                   "score by using samples to generate and average k Normal "
                   "distributions in the latent space.")
    params = SimpleNamespace(
        n_validation=n_validation, n_samples=n_samples, k=k,
        n_discretization_steps=n_discretization_steps, trust_level=trust_level)
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    name = "_".join(["_".join([key, str(val)])
                     for key, val in params.__dict__.items()])
    resdir = os.path.join(daadir, name)
    if not os.path.isdir(resdir):
        os.mkdir(resdir)
    errors_file = os.path.join(resdir, "rec_erros.npy")
    scores_file = os.path.join(resdir, "trav_scores.npy")
    trainset = latents.datatrain
    clinical_values = trainset["clinical"].cpu().detach().numpy()
    n_scores = clinical_values.shape[1]
    print_text(f"number of scores: {n_scores}")
    rois_values = trainset["rois"].cpu().detach().numpy()
    n_rois = rois_values.shape[1]
    if not os.path.isfile(errors_file):
        min_per_score, max_per_score = np.quantile(
            clinical_values, [0.05, 0.95], 0)
        print_text(f"min range per score: {min_per_score}")
        print_text(f"max range per score: {max_per_score}")
        traverses = torch.FloatTensor(np.linspace(
            min_per_score, max_per_score, params.n_discretization_steps))
        print_text(f"number of ROIs: {n_rois}")
        rec_errors, trav_scores = [], []
        for val_idx in tqdm(range(params.n_validation)):
            if flags.allow_missing_blocks:
                testsampler = MissingModalitySampler(
                    testset, batch_size=params.n_samples,
                    stratify=["age", "sex", "site"], discretize=["age"])
                testloader = DataLoader(
                    testset, batch_sampler=testsampler, num_workers=0)
            else:
                testloader = DataLoader(
                    testset, batch_size=params.n_samples, shuffle=True,
                    num_workers=0)
            data = {}
            dataiter = iter(testloader)
            while True:
                data, _, _ = dataiter.next()
                if all([mod in data.keys() for mod in modalities]):
                    break
            for idx, mod in enumerate(modalities):
                data[mod] = Variable(data[mod]).to(device).float()
            test_size = len(data[mod])
            rois_errors = np.zeros(
                (test_size, n_scores, params.n_discretization_steps, n_rois))
            scores_values = np.zeros(
                (test_size, n_scores, params.n_discretization_steps))
            if not linear_gradient:
                clinical_loc_hats = []
                clinical_scale_hats = []
                for k in range(params.k):
                    reconstructions = model(data, sample_latents=True)["rec"]
                    clinical_loc_hats.append(
                        reconstructions["clinical"].loc.unsqueeze(0))
                    clinical_scale_hats.append(
                        reconstructions["clinical"].scale.unsqueeze(0))
                dist = torch.distributions.Normal(
                    torch.cat(clinical_loc_hats).mean(0),
                    torch.cat(clinical_scale_hats).mean(0))
                scores_values = dist.sample(
                    torch.Size([params.n_discretization_steps]))
            for step in range(params.n_discretization_steps):
                for idx, qname in enumerate(clinical_names):
                    cdata = data["clinical"].clone()
                    if linear_gradient:
                        cdata[:, idx] = traverses[step, idx]
                        scores_values[:, idx, step] = traverses[step, idx]
                    else:
                        cdata[:, idx] = scores_values[step, :, idx]
                    modified_data = {
                        "clinical": cdata,
                        "rois": data["rois"]}
                    reconstructions = model(
                        modified_data, sample_latents=False)["rec"]
                    rois_hat = reconstructions["rois"].loc.detach()
                    rois_errors[:, idx, step] = (
                        rois_hat - data["rois"]).cpu().detach().numpy()
            if linear_gradient:
                scores_values = np.swapaxes(scores_values, 1, 2)
            else:
                scores_values = np.swapaxes(
                    scores_values.detach().numpy(), 0, 1)
            rec_errors.append(rois_errors)
            trav_scores.append(scores_values)
        rec_errors = np.asarray(rec_errors)
        trav_scores = np.asarray(trav_scores)
        np.save(errors_file, rec_errors)
        np.save(scores_file, trav_scores)
    else:
        print_text(f"restoring rec errors: {errors_file}")
        rec_errors = np.load(errors_file)
        print_text(f"restoring regressors: {scores_file}")
        trav_scores = np.load(scores_file)
    print_text(f"reconstruction errors: {rec_errors.shape}")
    print_text(f"traverse scores: {trav_scores.shape}")
    
    print_subtitle("Compute statistics (regression): traverse wrt "
                   "reconstruction errors...")
    coefs_file = os.path.join(resdir, "coefs.npy")
    pvals_file = os.path.join(resdir, "pvalues.npy")
    if not os.path.isfile(pvals_file):
        coefs = np.zeros((params.n_validation, n_scores, n_rois))
        pvalues = np.zeros((params.n_validation, n_scores, n_rois))
        for val_idx in tqdm(range(params.n_validation)):
            rois_errors = rec_errors[val_idx]
            scores_values = trav_scores[val_idx]
            for score_idx in range(n_scores):
                for roi_idx in range(n_rois):
                    X = sm.add_constant(
                        scores_values[:, :, score_idx].flatten())
                    est = sm.OLS(
                        rois_errors[:, score_idx, :, roi_idx].flatten(), X)
                    res = est.fit()
                    coefs[val_idx, score_idx, roi_idx] = res.params[1]
                    pvalues[val_idx, score_idx, roi_idx] = res.pvalues[1]
        np.save(pvals_file, pvalues)
        np.save(coefs_file, coefs)
    else:
        print_text(f"restoring p-values: {pvals_file}")
        pvalues = np.load(pvals_file)
        print_text(f"restoring regressors: {coefs_file}")
        coefs = np.load(coefs_file)
    print_text(f"p_values: {pvalues.shape}")
    print_text(f"regression coefficients: {coefs.shape}")

    print_subtitle("Compute statistics significativity...")
    significativity_thr = (0.05 / n_rois / n_scores)
    trust_level = params.n_validation * params.trust_level
    print_text(f"voting trust level: {trust_level} / {params.n_validation}")
    idx_sign = ((pvalues < significativity_thr).sum(axis=0) >= trust_level)
    data = {"metric": [], "roi": [], "score": []}
    for idx, score in enumerate(clinical_names):
        rois_idx = np.where(idx_sign[idx])
        for name in rois_names[rois_idx]:
            name, metric = name.rsplit("_", 1)
            data["score"].append(score)
            data["metric"].append(metric)
            data["roi"].append(name)
    df = pd.DataFrame.from_dict(data)
    significant_file = os.path.join(resdir, "significant_rois.tsv")
    df.to_csv(significant_file, sep="\t", index=False)
    print_result(f"significant ROIs: {significant_file}")
    print(df.groupby(["metric", "score"]).count())


def rsa_exp(dataset, datasetdir, outdir, run, n_validation=50, n_samples=50):
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
    n_samples: int, default 50
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
    alphabet_file = os.path.join(os.getcwd(), "alphabet.json")
    checkpoints_files = glob.glob(
        os.path.join(expdir, "checkpoints", "*", "mm_vae"))
    if len(checkpoints_files) == 0:
        raise ValueError("You need first to train the model.")
    checkpoints_files = sorted(
        checkpoints_files, key=lambda path: int(path.split(os.sep)[-2]))
    checkpoint_file = checkpoints_files[-1]
    print_text(f"restoring weights: {checkpoint_file}")
    experiment, flags = MultimodalExperiment.get_experiment(
        flags_file, alphabet_file, checkpoint_file)
    model = experiment.mm_vae
    print(model)
    clinical_names = np.load(
        os.path.join(datasetdir, "clinical_names.npy"), allow_pickle=True)
    rois_names = np.load(
        os.path.join(datasetdir, "rois_names.npy"), allow_pickle=True)
    modalities = ["clinical", "rois"]
    print_text(f"modalities: {modalities}")
    trainset = experiment.dataset_train
    print_text(f"train data: {len(trainset)}")
    testset = experiment.dataset_test
    print_text(f"test data: {len(testset)}")

    print_subtitle("Compute blocks correlations using Kendall tau statstic...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    kendalltaus = np.zeros((n_validation, len(clinical_names), 2))
    latent_disimilarities, scores_disimilarities = [], []
    for val_idx in tqdm(range(n_validation)):
        if flags.allow_missing_blocks:
            testsampler = MissingModalitySampler(
                testset, batch_size=n_samples, stratify=["age", "sex", "site"],
                discretize=["age"])
            testloader = DataLoader(
                testset, batch_sampler=testsampler, num_workers=0)
        else:
            testloader = DataLoader(
                testset, batch_size=n_samples, shuffle=True, num_workers=0)
        data = {}
        dataiter = iter(testloader)
        while True:
            data, _, _ = dataiter.next()
            if all([mod in data.keys() for mod in modalities]):
                break
        for idx, mod in enumerate(modalities):
            data[mod] = Variable(data[mod]).to(device).float()
        test_size = len(data[mod])
        latents = model(data, sample_latents=False)["latents"]["joint"][0]
        latents = latents.cpu().detach().numpy()
        print_text(f"latents: {latents.shape}")
        n_scores = data["clinical"].shape[1]
        n_subjects = len(latents)
        cmat = data2cmat(latents)
        latent_disimilarities.append(cmat)
        print_text(f"(dis)similarity matrix: {cmat.shape}")
        scores_cmats = []
        for score_idx in range(n_scores):
            score_cmat = vec2cmat(data["clinical"][:, score_idx])
            scores_cmats.append(score_cmat)
            tau, pval = fit_rsa(cmat, score_cmat)
            kendalltaus[val_idx, score_idx, 0] = tau
            kendalltaus[val_idx, score_idx, 1] = pval
        scores_cmats = np.asarray(scores_cmats)
        scores_disimilarities.append(scores_cmats)
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
    data = {"score": [], "pval": [], "pval_std": [], "r": [], "r_std": []}
    for score_idx in range(n_scores):
        data["score"].append(clinical_names[score_idx])
        data["pval"].append(np.mean(kendalltaus[:, score_idx, 1]))
        data["pval_std"].append(np.std(kendalltaus[:, score_idx, 1]))
        data["r"].append(np.mean(kendalltaus[:, score_idx, 0]))
        data["r_std"].append(np.std(kendalltaus[:, score_idx, 0]))
    df = pd.DataFrame.from_dict(data)
    summary_file = os.path.join(rsadir, "kendalltau_summary.tsv")
    df.to_csv(summary_file, sep="\t", index=False)
    print_result(f"kendall tau summary: {summary_file}")
    print(df.groupby(["score"]).apply(lambda e: e[:]))


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


def daa_plot_exp(dataset, datasetdir, outdir, run):
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
    from plotting import plot_surf_mosaic
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    from nilearn import datasets
    import seaborn as sns

    print_title(f"PLOT DAA results: {dataset}")
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
    clinical_names = np.load(
        os.path.join(datasetdir, "clinical_names.npy"), allow_pickle=True)
    clinical_names = clinical_names.tolist()
    rois_names = np.load(
        os.path.join(datasetdir, "rois_names.npy"), allow_pickle=True)
    rois_names = rois_names.tolist()
    for dirname in simdirs:
        df = pd.read_csv(os.path.join(dirname, "significant_rois.tsv"),
                         sep="\t")
        coefs = np.load(os.path.join(dirname, "coefs.npy"))

        print_subtitle(f"Plot regression coefficients radar plots...")
        counts = collections.Counter(df["roi"].values)
        selected_rois = [item[0] for item in counts.most_common(3)]
        selected_roi_indices = {"left": [], "right": []}
        for name in selected_rois:
            roi_name, hemi = name.rsplit("_", 1)
            hemi = "left" if hemi == "lh" else "right"
            selected_roi_indices[hemi].append(destrieux_labels.index(roi_name))
        parcellations = []
        for hemi, indices in selected_roi_indices.items():
            _par = destrieux_atlas[f"map_{hemi}"]
            parcellations.append(np.isin(_par, indices).astype(int))
        filename = os.path.join(dirname, "three_selected_rois.png")
        plot_surf_mosaic([parcellations], [" - ".join(selected_rois)],
                         fsaverage, filename, label=True)
        for _metric, _df in df.groupby(["metric"]):
            selected_scores = []
            for _roi in selected_rois:
                roi_idx = rois_names.index(f"{_roi}_{_metric}")
                selected_scores.append(coefs[:, :, roi_idx].mean(axis=0))
            selected_scores = np.asarray(selected_scores)
            max_score = coefs.max()
            fig = go.Figure()
            for roi_idx, _roi in enumerate(selected_rois):
                _scores = selected_scores[roi_idx].tolist()
                fig.add_trace(
                    go.Scatterpolar(
                        r=_scores + _scores[:1],
                        theta=[
                            "<b>" + name + "</b>"
                            for name in clinical_names + clinical_names[:1]],
                        mode="lines+text",
                        legendgroup="roi",
                        legendgrouptitle = dict(
                            font=dict(size=30, family="Droid Serif"),
                            text="<b>ROIs</b>"),
                        name=_roi))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True, showticklabels=False, ticks="",
                        range=[0, max_score + 0.003])),
                font=dict(size=20, family="Droid Serif"))
            filename = os.path.join(
                dirname, f"three_selected_rois_{_metric}_polarplots.png")
            fig.write_image(filename)
            print_result(f"{_metric} regression coefficients for 3 selected "
                         f"ROIs: {filename}")

        print_subtitle(f"Plot significant ROIs per score...")
        data, titles = [], []
        for (_metric, _score), _df in df.groupby(["metric", "score"]):
            roi_indices = {"left": [], "right": []}
            for name in _df["roi"].values:
                roi_name, hemi = name.rsplit("_", 1)
                hemi = "left" if hemi == "lh" else "right"
                roi_indices[hemi].append(destrieux_labels.index(roi_name))
            parcellations = []
            for hemi, indices in roi_indices.items():
                _par = destrieux_atlas[f"map_{hemi}"]
                parcellations.append(np.isin(_par, indices).astype(int))
            data.append(parcellations)
            titles.append(f"{_metric} - {_score}")
        filename = os.path.join(dirname, "significant_rois.png")
        plot_surf_mosaic(data, titles, fsaverage, filename, label=True)

        print_subtitle(f"Plot significant scores/ROIs flows...")
        for _metric, _df in df.groupby(["metric"]):
            significant_scores = _df["score"].values
            significant_rois = _df["roi"].values
            significant_coefs = []
            colors = []
            for _roi, _score in zip(significant_rois, significant_scores):
                score_idx = clinical_names.index(_score)
                roi_idx = rois_names.index(f"{_roi}_{_metric}")
                significant_coefs.append(coefs[:, score_idx, roi_idx].mean())
            significant_coefs = np.asarray(significant_coefs)
            colors = ["rgba(255,0,0,0.4)" if coef > 0 else "rgba(0,0,255,0.4)"
                      for coef in significant_coefs]
            sankey_plot = go.Parcats(
                domain={"x": [0.05, 0.9], "y": [0, 1]},
                dimensions=[{"label": "Score", "values": significant_scores},
                            {"label": "ROI", "values": significant_rois}],
                counts=np.abs(significant_coefs),
                line={"color": colors, "shape": "hspline"},
                labelfont=dict(family="Droid Serif", size=28),
                tickfont=dict(family="Droid Serif", size=20))
            fig = go.Figure(data=[sankey_plot])
            filename = os.path.join(dirname, f"score2roi_{_metric}_flow.png")
            fig.write_image(filename)
            print_result(f"flow for the {_metric} metric: {filename}")
