# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Perofrm the Digital Avatars Analysis (DAA).
"""

# Imports
import os
import pickle
import numpy as np
import statsmodels.api as sm
from scipy.stats import kendalltau
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from multimodal_cohort.dataset import DataManager, MissingModalitySampler
from argparse import ArgumentParser


# Define command line arguments
# ToDo: add the code in the workflow directly.
parser = ArgumentParser()
parser.add_argument("--run", type=str)
parser.add_argument("--datasetdir", type=str)
parser.add_argument("--dir_experiment", type=str)
parser.add_argument("--test", type=str, default=None)
args = parser.parse_args()


name_dataset_train = args.run.split("_")[0]

with open(os.path.join(args.dir_experiment, args.run, "experiment.pickle"), "rb") as f:
    exp = pickle.load(f)

print(len(exp.dataset_train))
if "allow_missing_blocks" in vars(exp.flags) and exp.flags.allow_missing_blocks:
    sampler = MissingModalitySampler(exp.dataset_train, batch_size=2048)
    loader_train = DataLoader(exp.dataset_train, batch_sampler=sampler, num_workers=8)
else:
    loader_train = DataLoader(exp.dataset_train, shuffle=True, batch_size=2048, num_workers=8)
modalities = ["clinical", "rois"]

dataset_test = exp.dataset_test
if args.test is not None:
    transform = {mod: transforms.Compose([
            exp.unsqueeze_0,
            scaler.transform,
            transforms.ToTensor(),
            torch.squeeze]) for mod, scaler in exp.scalers.items()}
    path_to_test_data = args.datasetdir.replace(name_dataset_train, args.test)
    path_to_test_data = path_to_test_data[:-1] + "-notest"
    manager = DataManager(
        args.test,
        path_to_test_data,
        modalities, test_size=0,
        on_the_fly_transform=transform)
    dataset_test = manager.train_dataset
if "allow_missing_blocks" in vars(exp.flags) and exp.flags.allow_missing_blocks:
    sampler_test = MissingModalitySampler(dataset_test, batch_size=512)
    loader_test = DataLoader(dataset_test, batch_sampler=sampler_test, num_workers=8)
else:
    loader_test = DataLoader(dataset_test, shuffle=True, batch_size=512, num_workers=8)

for batch in loader_train:
    data = batch[0]
    labels = batch[1]
    if all([mod in data.keys() for mod in modalities]):
        for k, m_key in enumerate(modalities):
            data[m_key] = Variable(data[m_key]).to(exp.flags.device).float()
        latents_train = exp.mm_vae.inference(data)
        train_data = data
for batch in loader_test:
    data = batch
    if args.test is None:
        data = batch[0]
        labels = batch[1]
    if all([mod in data.keys() for mod in modalities]):
        for k, m_key in enumerate(modalities):
            data[m_key] = Variable(data[m_key]).to(exp.flags.device).float()
        latents_test = exp.mm_vae.inference(data)
        test_data = data
print(latents_test["subsets"].keys())
print(latents_test["weights"])
subsets = list(latents_test["subsets"])
clinical_names = np.load(os.path.join(args.datasetdir, "clinical_names.npy"), allow_pickle=True)
rois_names = np.load(os.path.join(args.datasetdir, "rois_names.npy"), allow_pickle=True)

gradient_over_scores = True


params = {}
params["euaims"] = {
    "validation": 50, "n_discretization_steps": 200,
    "n_samples": 50, "K": 1000, "trust_level": 3/4}
params["hbn"] = {
    "validation": 50, "n_discretization_steps": 200,
    "n_samples": 50, "K": 1000, "trust_level": 3/4}
validation = params[name_dataset_train]["validation"]
n_discretization_steps = params[name_dataset_train]["n_discretization_steps"]
n_samples = params[name_dataset_train]["n_samples"]

seed = 1037
np.random.seed(seed)
torch.manual_seed(seed)

if gradient_over_scores:

    val_size = n_samples / (len(exp.dataset_train) * (5/4))
    K = params[name_dataset_train]["K"]
    trust_level = params[name_dataset_train]["trust_level"]
    stat_params = {
        "runs": validation,
        "samples_per_score": n_discretization_steps,
        "samples_in_run": n_samples,
        "K": K,
        "pvalue_select_thr": trust_level
    }

    if not os.path.isdir(os.path.join(args.dir_experiment, args.run, "results")):
        os.makedirs(os.path.join(args.dir_experiment, args.run, "results"))
    
    dir_name = "_".join(["_".join([key, str(value)]) for key, value in stat_params.items()])    
    if not os.path.isdir(os.path.join(args.dir_experiment, args.run, "results", dir_name)):
        os.makedirs(os.path.join(args.dir_experiment, args.run, "results", dir_name))

        clinical_values = train_data["clinical"].cpu().detach().numpy()

        min_per_score, max_per_score = np.quantile(clinical_values, [0.05, 0.95], 0)

        std_per_score = np.std(clinical_values, axis=0)

        transform = {mod: transforms.Compose([
                exp.unsqueeze_0,
                scaler.transform,
                transforms.ToTensor(),
                torch.squeeze]) for mod, scaler in exp.scalers.items()}
        if args.test is not None:
            manager = DataManager(
                args.test,
                path_to_test_data,
                modalities, test_size=0,
                validation=validation,
                val_size=val_size,
                on_the_fly_transform=transform)

        n_scores = train_data["clinical"].cpu().detach().numpy().shape[1]
        n_rois = train_data["rois"].cpu().detach().numpy().shape[1]
        pvalues = np.zeros((validation, n_scores, n_rois))
        coefs = np.zeros((validation, n_scores, n_rois))
        for val_step in range(validation):
            if args.test is not None:
                dataset_test = manager.train_dataset[val_step]["valid"]
                test_size = len(dataset_test)
                loader_test = DataLoader(dataset_test, batch_size=test_size,
                                shuffle=True, num_workers=8)
            elif "allow_missing_blocks" not in vars(exp.flags):
                loader_test = DataLoader(dataset_test, batch_size=n_samples,
                                shuffle=True, num_workers=8)
            else:
                sampler_test = MissingModalitySampler(dataset_test, batch_size=n_samples,
                                                    stratify=["age", "sex", "site"],
                                                    discretize=["age"])
                loader_test = DataLoader(dataset_test, batch_sampler=sampler_test, num_workers=8)
            batch_idx = 0
            data = {}
            for idx, batch in enumerate(loader_test):
                if args.test is None:
                    batch = batch[0]
                if all([mod in batch.keys() for mod in modalities]) and batch_idx == 0:
                    batch_idx += 1
                    for k, m_key in enumerate(modalities):
                        data[m_key] = Variable(batch[m_key]).to(exp.flags.device).float()
                        test_size = len(data[m_key])

            values = torch.FloatTensor(np.linspace(min_per_score, max_per_score, n_discretization_steps))

            all_errors = np.zeros((test_size, n_scores, n_discretization_steps,
                                            n_rois))
            score_values = np.zeros((test_size, n_scores, n_discretization_steps))
            linear_gradient = False
            if not linear_gradient:
                clinical_loc_hats = []
                clinical_scale_hats = []
                for k in range(K):
                    reconstructions = exp.mm_vae(data, sample_latents=True)["rec"]
                    clinical_loc_hats.append(reconstructions["clinical"].loc.unsqueeze(0))
                    clinical_scale_hats.append(reconstructions["clinical"].scale.unsqueeze(0))
                dist = torch.distributions.Normal(
                    torch.cat(clinical_loc_hats).mean(0),
                    torch.cat(clinical_scale_hats).mean(0))
                score_values = dist.sample(torch.Size([n_discretization_steps]))
            for step in range(n_discretization_steps):
                for idx, score in enumerate(clinical_names):
                    clinical_copy = data["clinical"].clone()
                    if linear_gradient:
                        clinical_copy[:, idx] = values[step, idx]
                        score_values[:, idx, step] = values[step, idx]
                    else:
                        clinical_copy[:, idx] = score_values[step, :, idx]
                    new_data = {
                        "clinical": clinical_copy,
                        "rois": data["rois"]}
                    reconstructions = exp.mm_vae(new_data, sample_latents=False)["rec"]
                    rois_hat = reconstructions["rois"].loc.detach()
                    all_errors[:, idx, step] = (rois_hat - data["rois"]).cpu().detach().numpy()

            if linear_gradient:
                score_values = np.swapaxes(score_values, 1, 2)
            else:
                score_values = np.swapaxes(score_values.detach().numpy(), 0, 1)
            for score_idx in range(n_scores):
                for roi_idx in range(n_rois):
                    X = score_values[:, :, score_idx].flatten()
                    X2 = sm.add_constant(X)
                    est = sm.OLS(all_errors[:, score_idx, :, roi_idx].flatten(), X2)
                    res = est.fit()
                    pvalues[val_step, score_idx, roi_idx] = res.pvalues[1]
                    coefs[val_step, score_idx, roi_idx] = res.params[1]

        np.save(os.path.join(args.dir_experiment, args.run, "results", dir_name, "pvalues.npy"), pvalues)
        np.save(os.path.join(args.dir_experiment, args.run, "results", dir_name, "coefs.npy"), coefs)
    else:
        pvalues = np.load(os.path.join(args.dir_experiment, args.run, "results", dir_name, "pvalues.npy"))
        coefs = np.load(os.path.join(args.dir_experiment, args.run, "results", dir_name, "coefs.npy"))
    
    significativity_thr = (0.05 / 444 / 7)
    
    print((pvalues < significativity_thr).sum())

    for prop in np.linspace(0.5, 1, 6):
        print(prop)
        print(((pvalues < significativity_thr).sum(0) >= validation*prop).sum())
        print(((pvalues < significativity_thr).sum(0) >= validation*prop).sum(1))

    print(coefs[(pvalues < significativity_thr)].max())
    print(coefs[(pvalues < significativity_thr)].min())
    print(clinical_names)

    trust_level = validation * trust_level
    idx_sign = (pvalues < significativity_thr).sum(0) >= trust_level
    rois_names_no_metric = np.array([name.replace("_{}".format(name.split("_")[-1]), "") for name in rois_names])
    for idx, score in enumerate(clinical_names):
        print(score)
        rois_idx = np.where(idx_sign[idx])
        print(len(rois_names[rois_idx]))
        print(len(np.unique(rois_names_no_metric[rois_idx])))
    
##################### RSA
compute_similarity_matrices = True
if compute_similarity_matrices:
    correlations = np.zeros((validation, len(clinical_names)))
    kendalltaus = np.zeros((validation, len(clinical_names)))
    kendallpvalues = np.zeros((validation, len(clinical_names)))
    for val_step in range(validation):
        if args.test is not None:
            dataset_test = manager.train_dataset[val_step]["valid"]
            test_size = len(dataset_test)
            loader_test = DataLoader(dataset_test, batch_size=test_size,
                            shuffle=True, num_workers=8)
        elif "allow_missing_blocks" not in vars(exp.flags):
            loader_test = DataLoader(dataset_test, batch_size=n_samples,
                            shuffle=True, num_workers=8)
        else:
            sampler_test = MissingModalitySampler(dataset_test, batch_size=n_samples,
                                                  stratify=["age", "sex", "site"],
                                                  discretize=["age"])
            loader_test = DataLoader(dataset_test, batch_sampler=sampler_test, num_workers=8)
        batch_idx = 0
        data = {}
        for idx, batch in enumerate(loader_test):
            if args.test is None:
                batch = batch[0]
            # if "clinical" in batch.keys() and "rois" not in batch.keys() and batch_idx == 0:
            if all([mod in batch.keys() for mod in modalities]) and batch_idx == 0:
                batch_idx += 1
                for k, m_key in enumerate(modalities):
                    data[m_key] = Variable(batch[m_key]).to(exp.flags.device).float()
                    test_size = len(data[m_key])
        latents = exp.mm_vae(data, sample_latents=False)["latents"]["joint"][0]
        n_scores = data["clinical"].shape[1]
        n_subjects = len(latents)
        latent_dissimilarity = np.zeros((n_subjects, n_subjects))
        idx_triu = np.triu(np.ones((n_subjects, n_subjects), dtype=bool))
        for i in range(n_subjects):
            for j in range(n_subjects):
                # if i == j:
                #     latent_dissimilarity[i, j] = 0
                # else:
                    # latent_dissimilarity[i, j] -= np.corrcoef(torch.cat([latents[i].unsqueeze(0), latents[j].unsqueeze(0)]).detach().numpy(), rowvar=False)[0, 1]
                latent_dissimilarity[i, j] = np.linalg.norm((latents[i] - latents[j]).cpu().detach().numpy())
    
        scores_dissimilarity = np.zeros((n_scores, n_subjects, n_subjects))
        for idx, score in enumerate(clinical_names):
            for i in range(n_subjects):
                for j in range(n_subjects):
                    scores_dissimilarity[idx, i, j] = np.linalg.norm(data["clinical"][i, idx] - data["clinical"][j, idx])
            r = np.corrcoef(np.concatenate([
                scores_dissimilarity[idx][idx_triu].flatten()[np.newaxis,:],
                latent_dissimilarity[idx_triu].flatten()[np.newaxis,:]]))[0, 1]
            correlations[val_step, idx]= r
            order_scores = scores_dissimilarity[idx][idx_triu].argsort(axis=None)
            ranks_scores = np.empty_like(order_scores)
            ranks_scores[order_scores] = np.arange(len(order_scores))

            order_latent = latent_dissimilarity[idx_triu].argsort(axis=None)
            ranks_latent = np.empty_like(order_latent)
            ranks_latent[order_latent] = np.arange(len(order_latent))

            tau, pvalue = kendalltau(scores_dissimilarity[idx][idx_triu], latent_dissimilarity[idx_triu])
            # tau, pvalue = kendalltau(ranks_scores, ranks_latent)
            kendalltaus[val_step, idx] = tau
            kendallpvalues[val_step, idx] = pvalue
        np.save(os.path.join(args.dir_experiment, args.run, "results", dir_name, "latent_dissimilarity.npy"), latent_dissimilarity)
        np.save(os.path.join(args.dir_experiment, args.run, "results", dir_name, "scores_dissimilarity.npy"), scores_dissimilarity)
    print(correlations.mean(0))
    print(correlations.std(0))
    print(kendalltaus.mean(0))
    print(kendalltaus.std(0))
    print(kendallpvalues.mean(0))
    print(kendallpvalues.std(0))
    
