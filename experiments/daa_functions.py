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
import glob
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


def make_digital_avatars(outdir, run, params, additional_data, permuted=False,
                         verbose=False):
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

    print_subtitle("Computing digital avatars.", verbose)
    expdir = os.path.join(outdir, run)
    daadir = os.path.join(expdir, "daa")
    if not os.path.isdir(daadir):
        os.mkdir(daadir)
    print_text(f"experimental directory: {expdir}", verbose)
    print_text(f"DAA directory: {daadir}", verbose)

    print_subtitle("Loading data...", verbose)
    flags_file = os.path.join(expdir, "flags.rar")
    if not os.path.isfile(flags_file):
        raise ValueError("You need first to train the model.")    
    checkpoints_dir = os.path.join(expdir, "checkpoints")
    experiment, flags = MultimodalExperiment.get_experiment(
        flags_file, checkpoints_dir)
    n_models = experiment.flags.num_models

    modalities = ["clinical", "rois"]
    print_text(f"modalities: {modalities}", verbose)

    n_scores = len(additional_data.clinical_names)
    n_rois = len(additional_data.rois_names)

    if params.seed is not None:
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
    name = "_".join(["_".join([key, str(val)])
                     for key, val in params.__dict__.items()])
    if permuted:
        name += "_permuted"
    resdir = os.path.join(daadir, name)

    da_file = os.path.join(resdir, "rois_digital_avatars.npy")
    sampled_scores_file = os.path.join(resdir, "sampled_scores.npy")
    metadata_file = os.path.join(resdir, "metadatas.npy")
    rois_reconstructions_file = os.path.join(resdir, "rois_reconstructions.npy")
    
    print_text(f"number of ROIs: {n_rois}", verbose)
    print_text(f"number of clinical scores: {n_scores}", verbose)
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
        print_text(f"train data: {len(trainset)}", verbose)
        if flags.allow_missing_blocks:
            trainsampler = MissingModalitySampler(
                trainset, batch_size=len(trainset))
            trainloader = DataLoader(
                trainset, batch_sampler=trainsampler, num_workers=0)
        else:
            trainloader = DataLoader(
                trainset, shuffle=True, batch_size=len(trainset), num_workers=0)

        print_text(f"test data: {len(testset)}", verbose)
        testloader = DataLoader(
            testset, shuffle=True, batch_size=len(testset), num_workers=0)

        print_subtitle("Evaluate model...", verbose)
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
                    if permuted:
                        data[mod] = data[mod][torch.randperm(data[mod].size()[0])]
                _data[f"z{phase}"] = model.inference(data)
                _data[f"data{phase}"] = data
        latents = SimpleNamespace(**_data)
        print_text(f"z train: {latents.ztrain['mus'].shape}", verbose)
        print_text(f"z test: {latents.ztest['mus'].shape}", verbose)
        subsets = list(latents.ztest["subsets"])
        print_text(f"subsets: {subsets}", verbose)
        trainset = latents.datatrain
        clinical_values = trainset["clinical"].cpu().detach().numpy()

        print_subtitle("Create digital avatars models using artificial clinical scores...", verbose)
        if params.sampling != "likelihood":
            print_text("Build the artificial values using population level statistics", verbose)
            min_per_score, max_per_score = np.quantile(
                clinical_values, [0.05, 0.95], 0)
            print_text(f"min range per score: {min_per_score}", verbose)
            print_text(f"max range per score: {max_per_score}", verbose)
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
                    "for each subject.", verbose)

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
                if permuted:
                    data[mod] = data[mod][torch.randperm(data[mod].size()[0])]
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
            if permuted:
                scores_values = scores_values[
                    torch.randperm(scores_values.size()[0])]
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
                           save_all_coefs=False, permuted=False):
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
    if permuted:
        name += "_permuted"
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
    y_name = "roi_avatar"
    if params.reg_method == "fixed":
        df["roi_reconstruction"] = np.repeat(
            rois_reconstruction[:, roi_idx, np.newaxis],
            params.n_samples, axis=1).flatten()
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


class Heuristic:
    def __init__(self, params, additional_data):
        granularities = ["overall", "metric", "score", "metric_score"]
        self.granularity = granularities[0]
        if any([g in params.keys() for g in granularities]):
            assert len(list(params.keys())) == 1
            self.granularity = next(iter(params.keys()))
            params = params[self.granularity]
        self.params = params
        self.additional_data = additional_data

    def __call__(self, coefs, pvalues, model_scores=None, return_agg=False):

        clinical_names = self.additional_data.clinical_names
        rois_names = self.additional_data.rois_names
        scores = self.additional_data.scores
        metrics = self.additional_data.metrics
        
        # Selection of model / validation indices of interest
        agg_values = np.zeros((len(clinical_names), len(rois_names)))
        if self.granularity == "overall":
            self._parse_params(self.params)
            higher_is_better = False
            if self.type == "pvalues":
                values = pvalues
            else:
                higher_is_better = True
                values = coefs

            # Aggregation
            agg_values, values_std = self.aggregate(values, model_scores)
            if self.aggregation == "vote":
                return agg_values

        rois = np.array(
            list(set([name.rsplit("_", 1)[0] for name in rois_names])))

        # Computing associations
        associations = {"metric": [], "roi": [], "score": []}
        for metric in metrics:
            metric_indices = np.array(
                [roi_idx for roi_idx, name in enumerate(rois_names)
                if metric in name])
            if self.granularity == "metric":
                self._parse_params(self.params[metric])
                higher_is_better = False
                if self.type == "pvalues":
                    values = pvalues
                else:
                    higher_is_better = True
                    values = coefs
                local_agg_values, values_std = self.aggregate(values, model_scores)
            for score in scores:
                score_idx = clinical_names.index(score)
                if self.granularity in ["score", "metric_score"]:
                    local_metric_score = (score if self.granularity == "score"
                                          else f"{metric}_{score}")
                    self._parse_params(self.params[local_metric_score])
                    higher_is_better = False
                    if self.type == "pvalues":
                        values = pvalues
                    else:
                        higher_is_better = True
                        values = coefs
                    local_agg_values, values_std = self.aggregate(values, model_scores)
                    agg_values[score_idx, metric_indices] = local_agg_values[score_idx, metric_indices]

                if self.type == "coefs":
                    local_values = np.absolute(agg_values[score_idx, metric_indices])
                    significance_indices = np.argsort(local_values)[::-1]
                else:
                    local_values = agg_values[score_idx, metric_indices]
                    significance_indices = np.argsort(local_values)
                stds = values_std[score_idx, metric_indices]
                if "-" not in self.strategy:
                    strat_param = self.strat_params[self.strategy]
                    if self.strategy == "num":
                        rois_indices = significance_indices[:strat_param]
                    elif self.strategy == "thr":
                        ordered_values = local_values[significance_indices]
                        rois_indices = significance_indices[
                            ordered_values >= strat_param if
                            higher_is_better else
                            ordered_values <= strat_param]
                    elif self.strategy == "var":
                        ordered_values = local_values[significance_indices]
                        ordered_stds = stds[significance_indices]
                        rois_indices = significance_indices[
                            ordered_values - strat_param * ordered_stds > 0]
                    area_idx = metric_indices[rois_indices]
                    areas = np.array(rois_names)[area_idx]
                    areas = [area.rsplit("_", 1)[0] for area in areas]

                    for area in areas:
                        associations["score"].append(score)
                        associations["metric"].append(metric)
                        associations["roi"].append(area)
                else:
                    second_param = self.strategy.split("-")[1]
                    num = self.strat_params["num"]
                    other_param = self.strat_params[second_param]

                    ordered_values = local_values[significance_indices]
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

                    for area in areas:
                        associations["score"].append(score)
                        associations["metric"].append(metric)
                        associations["roi"].append(area)
        if return_agg:
            return pd.DataFrame.from_dict(associations), agg_values
        return pd.DataFrame.from_dict(associations)
    
    def _parse_params(self, params):
        assert len(list(params.keys())) == 1
        self.name, other_params = next(iter(params.items()))
        self.type, self.aggregation = self.name.split("_", 1)
        self.strategy = other_params["strategy"]
        if "-" in self.strategy:
            first_param, second_param = self.strategy.split("-")
            self.strat_params = {first_param: other_params[first_param],
                                 second_param: other_params[second_param]}
        else:
            self.strat_params = {self.strategy : other_params[self.strategy]}
    
    def aggregate(self, values, model_scores=None):
        values_std = values.std((0, 1))
        apply_min_max = self.type == "coefs" and self.aggregation == "max"
        if hasattr(values, self.aggregation) and not apply_min_max:
            agg_values = getattr(values, self.aggregation)((0, 1))
        elif apply_min_max:
            agg_values_pos = values.max((0, 1))
            agg_values_neg = values.min((0, 1))
            agg_values = agg_values_pos.copy()
            idx_neg = agg_values_pos < -agg_values_neg
            agg_values[idx_neg] = agg_values_neg[idx_neg]
        elif self.aggregation == "vote":
            vote_prop = self.strat_params["vote_prop"]
            _, associations = compute_significativity(
                values, 1, vote_prop, 1, self.additional_data,
                correct_threshold=True, verbose=False)
            return associations, None
        elif "combine" in self.aggregation:
            method = self.aggregation.split("combine_")[-1]
            agg_values = combine_all_pvalues(values, method)
        elif "test" in self.aggregation:
            agg_values = non_nullity_coef(values)
        elif "weighted" in self.aggregation:
            method = self.aggregation.split("weighted_mean_")[-1]
            if method.startswith("rank"):
                sorted_idx = np.argsort(
                    model_scores).tolist()
                weights = []
                for idx in range(len(values)):
                    weights.append(sorted_idx.index(idx) + 1)
                weights = np.array(weights)
            elif method.startswith("score"):
                weights = model_scores
            if self.name.endswith("softmax"):
                weights = np.exp(weights)
            elif self.name.endswith("log"):
                weights = np.log(weights)
            weights = weights / weights.sum()
            agg_values = np.average(values.mean(1), axis=0, weights=weights)
        return agg_values, values_std

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


def score_models(dataset, datasetdir, outdir, run, scores=None,
                 latent_name="joint", plot_score_hist=False):
    """ Uses previously computed Representational Similarity Analysis (RSA) on 
    latent representations with input scores in order to score models.
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
    scores: list or None, default None
        the scores used to score the models
    latent_name: str, default joint
        the name of the latent representation space to use
    plot_score_hist: bool, default False,
        weither or not to plot the histogram of the computed scores

    Returns
    -------
    score_per_model: numpy.ndarray
        score per model
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
    if plot_score_hist:
        plt.hist(score_per_model, bins=20)
        plt.title("Model scores histogram")
        plt.show()
    return score_per_model

def combine_all_pvalues(pvalues, method="fisher"):
    combined_pvalues = np.ones(pvalues.shape[-2:])
    for score_idx in range(pvalues.shape[-2]):
        for roi_metric_idx in range(pvalues.shape[-1]):
            res = combine_pvalues(
                pvalues[:, :, score_idx, roi_metric_idx].flatten())
            combined_pvalues[score_idx, roi_metric_idx] = res[1]
    return combined_pvalues


def compute_all_associations(dataset, datasetdir, outdir, runs,
                             heuristics_params, metrics=None,
                             scores=None, model_indices=None, 
                             validation_indices=None, n_subjects=301,
                             sampling=None, sample_latents=None,
                             ensemble_models=False):

    global_results = []
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

    # Computing heuristics with various parameters for each metric / score
    # If ensemble models across runs instead of models for each run
    if ensemble_models:
        for run_idx, run in enumerate(runs):
            run_results = {}
            expdir = os.path.join(outdir, run)
            daadir = os.path.join(expdir, "daa")
            # print_text(f"experimental directory: {expdir}")
            # print_text(f"DAA directory: {daadir}")
            simdirs = [path for path in glob.glob(os.path.join(daadir, "*"))
                    if os.path.isdir(path)]
            # print_text(f"Simulation directories: {','.join(simdirs)}")

            # flags_file = os.path.join(expdir, "flags.rar")
            # if not os.path.isfile(flags_file):
            #     raise ValueError("You need first to train the model.")
            # checkpoints_dir = os.path.join(expdir, "checkpoints")
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
                                            rois_names=rois_names,
                                            scores=scores, metrics=metrics)

            for dirname in simdirs:
                # print_text(dirname)
                if not os.path.exists(os.path.join(dirname, "coefs.npy")):
                    continue
                coefs = np.load(os.path.join(dirname, "coefs.npy"))
                pvalues = np.load(os.path.join(dirname, "pvalues.npy"))

                if model_indices is None:
                    run_model_indices = range(len(pvalues))
                elif len(model_indices) >= 2 and type(model_indices[0]) is not int:
                    run_model_indices = model_indices[run_idx]
                elif np.array(model_indices).ndim == 2:
                    run_model_indices = model_indices[:, run_idx]
                else:
                    run_model_indices = model_indices

                n_validation = pvalues.shape[1]
                if validation_indices is not None:
                    n_validation = len(validation_indices)


                local_sampling = dirname.split("sampling_")[1].split("_sample")[0]
                local_sample_latents = dirname.split("latents_")[1].split("_seed")[0]
                local_n_subjects = int(dirname.split("subjects_")[1].split("_M")[0])

                if (local_n_subjects != n_subjects or
                    (sampling is not None and local_sampling != sampling) or
                    (sample_latents is not None and
                    local_sample_latents != str(sample_latents))):
                    continue

                
                # Selection of model / validation indices of interest
                # if validation_indices is not None:
                #     pvalues = pvalues[:, validation_indices]
                #     coefs = coefs[:, validation_indices]
                # else:
                #     pvalues = pvalues[run_model_indices]
                #     coefs = coefs[run_model_indices]
                
                
                # model_scores = None
                # if len(weighted_mean_heuristics) != 0:
                model_scores = score_models(dataset, datasetdir, outdir, run,
                                            scores=scores)
                # model_scores = model_scores[run_model_indices]

                dir_results = {"coefs": coefs, "pvalues": pvalues,
                               "model_scores": model_scores}
                
                run_results[f"{local_sampling}_{local_sample_latents}"] = dir_results

            global_results.append(run_results)
        weighted_mean_heuristics = [heuristic for heuristic in heuristics_params.keys()
                                    if "weighted_mean" in heuristic]

        rois = np.array(
            list(set([name.rsplit("_", 1)[0] for name in rois_names])))
        
        final_results = []
        coefs = np.array([res[f"{sampling}_{sample_latents}"]["coefs"] for res in global_results])
        pvalues = np.array([res[f"{sampling}_{sample_latents}"]["pvalues"] for res in global_results])
        model_scores = np.array([res[f"{sampling}_{sample_latents}"]["model_scores"] for res in global_results])

        N_splits = len(coefs[0])
        for split_idx in range(N_splits):
            split_results = {}
            local_results = {}
            for heuristic_name in heuristics_params.keys():
                if heuristic_name not in local_results.keys():
                    local_results[heuristic_name] = {}
                for strategy in heuristics_params[heuristic_name]["strategy"]:
                    if "-" in strategy:
                        first_param, second_param = strategy.split("-")
                        for first_value, second_value in itertools.product(
                            heuristics_params[heuristic_name][first_param],
                            heuristics_params[heuristic_name][second_param]):

                            params = {"strategy": strategy,
                                    first_param: first_value,
                                    second_param: second_value}
                            heuristic_param = {heuristic_name: params}
                            heuristic = Heuristic(heuristic_param,
                                                additional_data)
                            associations = heuristic(coefs[:,split_idx], pvalues[:,split_idx], model_scores[:,split_idx])
                            strat_param_name = f"strategy_{strategy}_values_{first_value}_{second_value}"
                            local_results[heuristic.name][strat_param_name] = associations
                    else:
                        for strat_param in heuristics_params[heuristic_name][strategy]:
                            params = {"strategy": strategy,
                                    strategy : strat_param}
                            heuristic_param = {heuristic_name: params}
                            heuristic = Heuristic(heuristic_param,
                                                additional_data)
                            associations = heuristic(coefs[:,split_idx], pvalues[:,split_idx], model_scores[:,split_idx])
                            strat_param_name = f"strategy_{strategy}_value_{strat_param}"
                            local_results[heuristic.name][strat_param_name] = associations
            split_results[f"{sampling}_{sample_latents}"] = local_results
            final_results.append(split_results)
        return final_results
    for run_idx, run in enumerate(runs):
        run_results = {}
        expdir = os.path.join(outdir, run)
        daadir = os.path.join(expdir, "daa")
        # print_text(f"experimental directory: {expdir}")
        # print_text(f"DAA directory: {daadir}")
        simdirs = [path for path in glob.glob(os.path.join(daadir, "*"))
                if os.path.isdir(path)]
        # print_text(f"Simulation directories: {','.join(simdirs)}")

        # flags_file = os.path.join(expdir, "flags.rar")
        # if not os.path.isfile(flags_file):
        #     raise ValueError("You need first to train the model.")
        # checkpoints_dir = os.path.join(expdir, "checkpoints")
        # experiment, flags = MultimodalExperiment.get_experiment(
        #     flags_file, checkpoints_dir)

        for dirname in simdirs:
            # print_text(dirname)
            if not os.path.exists(os.path.join(dirname, "coefs.npy")):
                continue
            coefs = np.load(os.path.join(dirname, "coefs.npy"))
            pvalues = np.load(os.path.join(dirname, "pvalues.npy"))

            if model_indices is None:
                run_model_indices = range(len(pvalues))
            elif len(model_indices) >= 2 and type(model_indices[0]) is not int:
                run_model_indices = model_indices[run_idx]
            elif np.array(model_indices).ndim == 2:
                run_model_indices = model_indices[:, run_idx]
            else:
                run_model_indices = model_indices

            n_validation = pvalues.shape[1]
            if validation_indices is not None:
                n_validation = len(validation_indices)


            local_sampling = dirname.split("sampling_")[1].split("_sample")[0]
            local_sample_latents = dirname.split("latents_")[1].split("_seed")[0]
            local_n_subjects = int(dirname.split("subjects_")[1].split("_M")[0])

            if (local_n_subjects != n_subjects or
                (sampling is not None and local_sampling != sampling) or
                (sample_latents is not None and
                 local_sample_latents != str(sample_latents))):
                continue

            dir_results = {}
            # Selection of model / validation indices of interest
            if validation_indices is not None:
                pvalues = pvalues[:, validation_indices]
                coefs = coefs[:, validation_indices]
            else:
                pvalues = pvalues[run_model_indices]
                coefs = coefs[run_model_indices]
            
            weighted_mean_heuristics = [heuristic for heuristic in heuristics_params.keys()
                                        if "weighted_mean" in heuristic]
            model_scores = None
            if len(weighted_mean_heuristics) != 0:
                model_scores = score_models(dataset, datasetdir, outdir, run,
                                            scores=scores)
                model_scores = model_scores[run_model_indices]

            rois = np.array(
                list(set([name.rsplit("_", 1)[0] for name in rois_names])))
            
            for heuristic_name in heuristics_params.keys():
                if heuristic_name not in dir_results.keys():
                    dir_results[heuristic_name] = {}
                for strategy in heuristics_params[heuristic_name]["strategy"]:
                    if "-" in strategy:
                        first_param, second_param = strategy.split("-")
                        for first_value, second_value in itertools.product(
                            heuristics_params[heuristic_name][first_param],
                            heuristics_params[heuristic_name][second_param]):

                            params = {"strategy": strategy,
                                      first_param: first_value,
                                      second_param: second_value}
                            heuristic_param = {heuristic_name: params}
                            heuristic = Heuristic(heuristic_param,
                                                  additional_data)
                            associations = heuristic(coefs, pvalues, model_scores)
                            strat_param_name = f"strategy_{strategy}_values_{first_value}_{second_value}"
                            dir_results[heuristic.name][strat_param_name] = associations
                    else:
                        for strat_param in heuristics_params[heuristic_name][strategy]:
                            params = {"strategy": strategy,
                                      strategy : strat_param}
                            heuristic_param = {heuristic_name: params}
                            heuristic = Heuristic(heuristic_param,
                                                  additional_data)
                            associations = heuristic(coefs, pvalues, model_scores)
                            strat_param_name = f"strategy_{strategy}_value_{strat_param}"
                            dir_results[heuristic.name][strat_param_name] = associations
            run_results[f"{local_sampling}_{local_sample_latents}"] = dir_results
        global_results.append(run_results)
    return global_results



def compute_stability(associations0, associations1, N_prior, eps=1e-8, measure="product"):
    assert measure in ["product", "dice", "jaccard", "tanimoto"]

    N0 = len(associations0)
    N1 = len(associations1)
    N_inter = len(set(associations0).intersection(associations1))
    N_outer = len(set(associations0).union(associations1))

    if measure == "dice":
        stability = 2 * N_inter / (N0 + N1 + eps)
    elif measure == "jaccard":
        stability = N_inter / (N_outer + eps)
    elif measure == "tanimoto":
        stability = 1 - (N_outer - N_inter) / (N_outer + eps)
    else:
        stability = N_inter * N_inter / (N0 * N1 + eps)
    penality = 2  / (N_inter / N_prior + N_prior / (N_inter + eps))
    penalized_stability = stability * penality
    return stability, penalized_stability


def compute_all_stability(results, daa_params, heuristic, strat_param_name,
                          ideal_N, metrics, scores, measure="product"):

    res0 = results[0][daa_params][heuristic][strat_param_name]
    res1 = results[1][daa_params][heuristic][strat_param_name]

    all_assoc0 = list(zip(res0["score"], res0["metric"], res0["roi"]))
    all_assoc1 = list(zip(res1["score"], res1["metric"], res1["roi"]))
    
    stability_per_score_metric = {
        "daa_params": [], "heuristic": [], "strat_param": [], "metric": [],
        "score": [], "stability": [], "penalized_stability": []}
    for metric_idx, metric in enumerate(metrics):
        metric_assoc0 = [assoc for assoc in all_assoc0 if metric in assoc]
        metric_assoc1 = [assoc for assoc in all_assoc1 if metric in assoc]
        for score_idx, score in enumerate(scores):
            local_assoc0 = [assoc for assoc in metric_assoc0 if score in assoc]
            local_assoc1 = [assoc for assoc in metric_assoc1 if score in assoc]
            stability, penalized_stability = compute_stability(
                local_assoc0, local_assoc1, ideal_N, measure=measure)
            stability_per_score_metric["daa_params"].append(daa_params)
            stability_per_score_metric["heuristic"].append(heuristic)
            stability_per_score_metric["strat_param"].append(strat_param_name)
            stability_per_score_metric["metric"].append(metric)
            stability_per_score_metric["score"].append(score)
            stability_per_score_metric["stability"].append(stability)
            stability_per_score_metric["penalized_stability"].append(penalized_stability)
    return stability_per_score_metric