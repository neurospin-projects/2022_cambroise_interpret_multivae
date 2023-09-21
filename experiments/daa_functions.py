# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define the digital avatar functions used during the analysis.
"""

# Imports
import os
import itertools
from joblib import Parallel, delayed
import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd
from tqdm import tqdm
from types import SimpleNamespace
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from multimodal_cohort.experiment import MultimodalExperiment
from multimodal_cohort.dataset import MissingModalitySampler
from stat_utils import make_regression
from color_utils import (print_title, print_subtitle, print_text)


def make_digital_avatars(outdir, run, params, additional_data):
    """ Build the digital avatars using clinical scores taverses
    to influence the imaging part.
    Parameters
    ----------
    outdir: str
        the destination folder.
    run: str
        the name of the experiment in the destination folder:
        `<dataset>_<timestamp>'.
    params: types.SimpleNamespace
        parameters values for the daa
    additional_data: types.SimpleNamespace
        additional usefull information
    """

    print_subtitle("Computing digital avatars.")
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

    modalities = ["clinical", "rois"]
    print_text(f"modalities: {modalities}")

    n_scores = len(additional_data.clinical_names)
    n_rois = len(additional_data.rois_names)

    if params.seed is not None:
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
    name = "_".join(["_".join([key, str(val)])
                     for key, val in params.__dict__.items()])
    resdir = os.path.join(daadir, name)

    da_file = os.path.join(resdir, "rois_digital_avatars.npy")
    sampled_scores_file = os.path.join(resdir, "sampled_scores.npy")
    metadata_file = os.path.join(resdir, "metadatas.npy")
    rois_reconstructions_file = os.path.join(resdir, "rois_reconstructions.npy")
    
    print_text(f"number of ROIs: {n_rois}")
    print_text(f"number of clinical scores: {n_scores}")
    all_sampled_scores, all_metadatas, all_rois_reconstructions = [], [], []
    shape=(n_models, params.n_validation,
           params.n_subjects, n_scores, params.n_samples,
           n_rois)
    rois_digital_avatars = open_memmap(
        da_file, dtype='float32', mode='w+',
        shape=shape)

    for model_idx in range(n_models):
        trainset = experiment.dataset_train
        testset = experiment.dataset_test
        model = experiment.models
        if n_models > 1:
            trainset = trainset[model_idx]
            testset = testset[model_idx]
            model = model[model_idx]
        print_text(f"train data: {len(trainset)}")
        if flags.allow_missing_blocks:
            trainsampler = MissingModalitySampler(
                trainset, batch_size=len(trainset))
            trainloader = DataLoader(
                trainset, batch_sampler=trainsampler, num_workers=0)
        else:
            trainloader = DataLoader(
                trainset, shuffle=True, batch_size=len(trainset), num_workers=0)

        print_text(f"test data: {len(testset)}")
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
                    data, labels, _ = next(dataiter)
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
        trainset = latents.datatrain
        clinical_values = trainset["clinical"].cpu().detach().numpy()

        print_subtitle("Create digital avatars models using artificial clinical scores...")
        if params.sampling != "likelihood":
            print_text("Build the artificial values using population level statistics")
            min_per_score, max_per_score = np.quantile(
                clinical_values, [0.05, 0.95], 0)
            print_text(f"min range per score: {min_per_score}")
            print_text(f"max range per score: {max_per_score}")
            if params.sampling == "linear":
                scores_values = torch.FloatTensor(np.repeat(
                    np.linspace(min_per_score, max_per_score,
                        params.n_samples)[np.newaxis, :], params.n_subjects, axis=0))
            elif params.sampling == "uniform":
                scores_values = torch.FloatTensor(np.random.uniform(
                    min_per_score, max_per_score,
                    size=(params.n_subjects, params.n_samples, n_scores)))
            else:
                scores_values = torch.FloatTensor(np.random.normal(
                    0, 1,
                    size=(params.n_subjects, params.n_samples, n_scores)))
        else:
            print_text("Build the artificial values for each score by "
                    "sampling in the estimated output distribution "
                    "for each subject.")

        sampled_scores, metadatas, rois_reconstructions = [], [], []
        for val_idx in tqdm(range(params.n_validation)):
            testloader = DataLoader(
                testset, batch_size=params.n_subjects, shuffle=True,
                num_workers=0)
            data = {}
            dataiter = iter(testloader)
            while True:
                data, _, metadata = next(dataiter)
                if all([mod in data.keys() for mod in modalities]):
                    break
            for idx, mod in enumerate(modalities):
                data[mod] = Variable(data[mod]).to(device).float()
            metadata_df = pd.DataFrame(columns=list(metadata.keys()))
            for column in metadata.keys():
                if type(metadata[column]) is list:
                    metadata_df[column] = np.array(metadata[column])
                else:
                    metadata_df[column] = metadata[column].cpu().detach().numpy()
            metadatas.append(metadata_df.to_numpy())
            test_size = len(data[mod])
            rois_avatars = np.zeros(
                (test_size, n_scores, params.n_samples, n_rois))

            clinical_loc_hats = []
            clinical_scale_hats = []
            rois_loc_hats = []
            for _ in range(params.M):
                reconstructions = model(data, sample_latents=True)["rec"]
                clinical_loc_hats.append(
                    reconstructions["clinical"].loc.unsqueeze(0))
                clinical_scale_hats.append(
                    reconstructions["clinical"].scale.unsqueeze(0))
                rois_loc_hats.append(
                    reconstructions["rois"].loc.unsqueeze(0))
            clinical_loc_hat = torch.cat(clinical_loc_hats).mean(0)
            clinical_scale_hat = torch.cat(clinical_scale_hats).mean(0)
            rois_reconstruction = torch.cat(rois_loc_hats).mean(0)
            rois_reconstructions.append(
                rois_reconstruction.cpu().detach().numpy())
            if params.sampling == "likelihood":
                dist = torch.distributions.Normal(
                    clinical_loc_hat, clinical_scale_hat)
                scores_values = dist.sample(
                    torch.Size([params.n_samples]))
            for sample_idx in range(params.n_samples):
                for idx, qname in enumerate(additional_data.clinical_names):
                    cdata = data["clinical"].clone()
                    if params.sampling == "likelihood":
                        cdata[:, idx] = scores_values[sample_idx, :, idx]
                    else:
                        cdata[:, idx] = scores_values[:, sample_idx, idx]
                    modified_data = {
                        "clinical": cdata,
                        "rois": data["rois"]}
                    reconstructions = model(
                        modified_data, sample_latents=params.sample_latents)["rec"]
                    rois_hat = reconstructions["rois"].loc.cpu().detach()
                    rois_avatars[:, idx, sample_idx] = rois_hat.numpy()
            if params.sampling == "likelihood":
                scores_values = np.swapaxes(
                    scores_values, 0, 1)
            rois_digital_avatars[model_idx, val_idx] = rois_avatars
            sampled_scores.append(scores_values.cpu().detach().numpy())
        all_sampled_scores.append(sampled_scores)
        all_metadatas.append(metadatas)
        all_rois_reconstructions.append(rois_reconstructions)

    all_sampled_scores = np.asarray(all_sampled_scores)
    all_rois_reconstructions = np.asarray(all_rois_reconstructions)
    del rois_digital_avatars
    np.save(sampled_scores_file, all_sampled_scores)
    np.save(metadata_file, all_metadatas)
    np.save(rois_reconstructions_file, all_rois_reconstructions)


def compute_daa_statistics(outdir, run, params, additional_data,
                           save_all_coefs=False):
    """ Perform the digital avatars analysis using clinical scores taverses
    to influence the imaging part.
    Parameters
    ----------
    outdir: str
        the destination folder.
    run: str
        the name of the experiment in the destination folder:
        `<dataset>_<timestamp>'.
    params: types.SimpleNamespace
        parameters values for the daa
    additional_data: types.SimpleNamespace
        additional usefull information
    """

    print_subtitle("Fitting regressions for statistics.")
    
    # Set environ variable for when computing lots of regressions (but with
    # parallel and MKL with big matrices, see github issue
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_DYNAMIC"] = "FALSE"

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

    modalities = ["clinical", "rois"]
    print_text(f"modalities: {modalities}")

    n_scores = len(additional_data.clinical_names)
    n_rois = len(additional_data.rois_names)

    name = "_".join(["_".join([key, str(val)])
                     for key, val in params.__dict__.items()])
    resdir = os.path.join(daadir, name)
    if not os.path.isdir(resdir):
        raise ValueError("You must first compute digital avatars before "
                         "computing statistics.")

    da_file = os.path.join(resdir, "rois_digital_avatars.npy")
    sampled_scores_file = os.path.join(resdir, "sampled_scores.npy")
    metadata_file = os.path.join(resdir, "metadatas.npy")
    rois_reconstructions_file = os.path.join(resdir, "rois_reconstructions.npy")
    coefs_file = os.path.join(resdir, "coefs.npy")
    pvals_file = os.path.join(resdir, "pvalues.npy")
    if params.reg_method == "hierarchical" and save_all_coefs:
        shape = (n_models, params.n_validation, n_scores)
        all_coefs = open_memmap(da_file, dtype=object, mode='w+', shape=shape)
        all_coefs_file = os.path.join(resdir, "all_coefs.npy")
    
    rois_digital_avatars = np.load(da_file, mmap_mode="r+")
    all_metadatas = np.load(metadata_file, allow_pickle=True)
    all_rois_reconstructions = np.load(rois_reconstructions_file,
                                       mmap_mode="r")
    all_sampled_scores = np.load(sampled_scores_file, mmap_mode="r")
    print_text(f"digital avatars rois: {rois_digital_avatars.shape}")
    print_text(f"sampled scores: {all_sampled_scores.shape}")
    print_text(f"metadata: {len(all_metadatas), all_metadatas[0].shape}")
    
    print_subtitle("Compute statistics (regression): digital avatar wrt "
                   "sampled scores...")
    
    participant_id_idx = additional_data.metadata_columns.index("participant_id")
    site_idx = additional_data.metadata_columns.index("site")
    base_dfs = []
    for model_idx in range(n_models):
        base_dfs.append([])
        for val_idx in range(params.n_validation):
            base_dfs[model_idx].append([])
            for score_idx in range(n_scores):
                base_df = pd.DataFrame(dict(
                    participant_id=np.repeat(
                        all_metadatas[model_idx][val_idx][:, participant_id_idx, np.newaxis],
                        params.n_samples, axis=1).flatten(),
                    sampled_score=all_sampled_scores[model_idx][val_idx][:, :, score_idx].flatten()))
                base_dfs[model_idx][val_idx].append(base_df)
                if params.reg_method == "hierarchical" and save_all_coefs:
                    local_metadata = pd.DataFrame(all_metadatas[model_idx][val_idx][:, [participant_id_idx, site_idx]],
                                        columns=["participant_id", "site"])
                    all_coefs[model_idx, val_idx, score_idx] = local_metadata

    coefs = np.zeros((n_models, params.n_validation, n_scores, n_rois))
    pvalues = np.zeros((n_models, params.n_validation, n_scores, n_rois))

    product_of_params = list(
        itertools.product(range(n_models), range(params.n_validation),
                          range(n_scores), range(n_rois)))
    all_results = Parallel(n_jobs=-2, verbose=1)(
        delayed(compute_regressions)(
            score_idx, roi_idx, base_dfs[model_idx][val_idx][score_idx],
            rois_digital_avatars[model_idx][val_idx],
            all_rois_reconstructions[model_idx][val_idx],
            params) for model_idx, val_idx, score_idx, roi_idx in
        product_of_params)

    print_subtitle("All regression computed. Saving results.")

    for param_index, param_values in enumerate(product_of_params):
        model_idx, val_idx, score_idx, roi_idx = param_values
    # for model_idx in range(n_models):
    #     for val_idx in range(params.n_validation):
    #         for score_idx in range(n_scores):
    #             for roi_idx in range(n_rois):
        # results = all_results[product_of_params.index((model_idx, val_idx, score_idx, roi_idx))]
        new_pvals, new_coefs, all_betas = all_results[param_index]
        pvalues[model_idx, val_idx, score_idx, roi_idx] = new_pvals
        coefs[model_idx, val_idx, score_idx, roi_idx] = new_coefs
        if params.reg_method == "hierarchical" and save_all_coefs:
            roi_name = additional_data.rois_names[roi_idx].replace(
                "&", "_").replace("-", "_")
            all_betas.rename(columns={"beta": roi_name}, inplace=True)
            all_coefs[model_idx, val_idx, score_idx] = all_coefs[model_idx, val_idx, score_idx].join(
                all_betas.set_index("participant_id"), on="participant_id")
    

    np.save(pvals_file, pvalues)
    np.save(coefs_file, coefs)
    if params.reg_method == "hierarchical" and save_all_coefs:
        del all_coefs
    print_text(f"p_values: {pvalues.shape}")
    print_text(f"regression coefficients: {coefs.shape}")


def compute_all_regressions(rois_avatars, scores_values, metadata,
                            rois_reconstruction, params, additional_data):
    """ Compute regressions for all scores and rois pairs
    """
    all_coefs = None
    if params.reg_method == "hierarchical":
        all_coefs = []
    n_scores = scores_values.shape[2]
    n_rois = rois_avatars.shape[3]
    coefs = np.zeros((n_scores, n_rois))
    pvalues = np.zeros((n_scores, n_rois))
    participant_id_idx = additional_data.metadata_columns.index("participant_id")
    site_idx = additional_data.metadata_columns.index("site")
    base_dfs = []
    for score_idx in range(n_scores):
        base_df = pd.DataFrame(dict(
            participant_id=np.repeat(
                metadata[:, participant_id_idx, np.newaxis],
                params.n_samples, axis=1).flatten(),
            sampled_score=scores_values[:, :, score_idx].flatten()))
        base_dfs.append(base_df)
        if params.reg_method == "hierarchical":
            local_metadata = pd.DataFrame(metadata[:, [participant_id_idx, site_idx]],
                                columns=["participant_id", "site"])
            all_coefs.append(local_metadata)

    product_of_params = list(itertools.product(range(n_scores), range(n_rois)))
    all_results = Parallel(n_jobs=15, verbose=1)(
        delayed(compute_regressions)(score_idx, roi_idx, base_dfs[score_idx],
                                     params, rois_avatars, rois_reconstruction)
        for score_idx, roi_idx in product_of_params)
    
    for score_idx in range(n_scores):
        for roi_idx in range(n_rois):
            results = all_results[product_of_params.index((score_idx, roi_idx))]
            new_pvals, new_coefs, all_betas = results
            pvalues[score_idx, roi_idx] = new_pvals
            coefs[score_idx, roi_idx] = new_coefs
            if params.reg_method == "hierarchical":
                roi_name = additional_data.rois_names[roi_idx].replace(
                    "&", "_").replace("-", "_")
                all_betas.rename(columns={"beta": roi_name}, inplace=True)
                all_coefs[score_idx] = all_coefs[score_idx].join(
                    all_betas.set_index("participant_id"), on="participant_id")
    return pvalues, coefs, all_coefs


def compute_regressions(score_idx, roi_idx, base_df, rois_avatars,
                        rois_reconstruction, params):
    """ Compute regressions for a given score idx and roi idx
    """
    df = base_df.copy()
    df["roi_avatar"] = (
        rois_avatars[:, score_idx, :, roi_idx].flatten())
    df["roi_reconstruction"] = np.repeat(
        rois_reconstruction[:, roi_idx, np.newaxis],
        params.n_samples, axis=1).flatten()
    y_name = "roi_avatar"
    if params.reg_method == "fixed":
        df["roi_avatar_diff"] = df["roi_avatar"] - df["roi_reconstruction"]
        y_name = "roi_avatar_diff"
    
    results = make_regression(df, "sampled_score", y_name,
                                groups_name="participant_id",
                                method=params.reg_method)
    return results


def compute_significativity(pvalues, trust_level, vote_prop, n_validation,
                            additional_data, threshold=0.05,
                            correct_threshold=True, verbose=True):
    """ Compute significative relationship indices
    """
    if verbose:
        print_subtitle("Compute statistics significativity...")

    n_scores = len(additional_data.clinical_names)
    n_rois = len(additional_data.rois_names)

    if correct_threshold:
        threshold = (threshold / n_rois / n_scores)

    scaled_trust_level = n_validation * trust_level
    if verbose:
        print_text(f"voting trust level: {scaled_trust_level} / {n_validation}")
    idx_sign = ((pvalues < threshold).sum(axis=1) >= scaled_trust_level)
    idx_sign = idx_sign.sum(0) >= vote_prop * len(pvalues)

    data = {"metric": [], "roi": [], "score": []}
    for idx, score in enumerate(additional_data.clinical_names):
        rois_idx = np.where(idx_sign[idx])
        for name in additional_data.rois_names[rois_idx]:
            name, metric = name.rsplit("_", 1)
            data["score"].append(score)
            data["metric"].append(metric)
            data["roi"].append(name)
    significant_assoc = pd.DataFrame.from_dict(data)
    if verbose:
        print(significant_assoc.groupby(["metric", "score"]).count())
    return idx_sign, significant_assoc
