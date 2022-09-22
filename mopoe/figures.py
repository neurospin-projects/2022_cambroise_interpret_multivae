# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


from argparse import ArgumentParser
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import plotly.graph_objects as go
import pickle

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP

from multimodal_cohort.dataset import DataManager, MissingModalitySampler
from plot_utils import plot_latent_representations, plot_surf, plot_areas

parser = ArgumentParser()
parser.add_argument("--run", type=str)
parser.add_argument("--datasetdir", type=str)
parser.add_argument("--dir_experiment", type=str)
parser.add_argument("--test", type=str, default=None)
args = parser.parse_args()

#################################### Overall histogram

red = (254/255, 0, 4/255)
blue = (2/255, 112/255, 187/255)
green = (0, 175/255, 80/255)
yellow = (243/255, 208/255, 144/255)
purple_blue = (122/255, 123/255, 255/255)

control_color = purple_blue
asd_color = yellow
hbn_color = green

cohorts = ["euaims", "hbn"]
allow_missing_blocks = {
    "euaims": True,
    "hbn": True,
}

srs_name = {
    "euaims": "t1_srs_rawscore",
    "hbn": "SRS_Total"
}

modalities = ["clinical", "rois"]
batch_size = 128

kwargs = {}

name_dataset_train = args.run.split("_")[0]

stuff_to_hist = []
for cohort in cohorts:
    print(cohort)
    path = args.datasetdir.replace(name_dataset_train, "{}").format(cohort)
    manager = DataManager(dataset=cohort, datasetdir=path,
                        modalities=modalities, overwrite=True,
                        allow_missing_blocks=allow_missing_blocks[cohort],
                        **kwargs)

    if allow_missing_blocks[cohort]:
        sampler = MissingModalitySampler(manager["train"], batch_size=batch_size)
        loader = DataLoader(manager["train"], batch_sampler=sampler, num_workers=8)
        sampler_test = MissingModalitySampler(manager["test"], batch_size=batch_size)
        loader_test = DataLoader(manager["test"], batch_sampler=sampler_test, num_workers=8) 
    else:
        loader = DataLoader(manager["train"], shuffle=True, batch_size=batch_size, num_workers=8)
        loader_test = DataLoader(manager["test"], shuffle=True, batch_size=batch_size, num_workers=8)

    all_clinical_data = []
    all_metadata = []
    for data in loader:
        if "clinical" in data[0].keys():
            all_clinical_data.append(data[0]["clinical"])
            all_metadata.append(pd.DataFrame(data[2]))

    for data in loader_test:
        if "clinical" in data[0].keys():
            all_clinical_data.append(data[0]["clinical"])
            all_metadata.append(pd.DataFrame(data[2]))

    X = torch.cat(all_clinical_data, dim=0).cpu().detach().numpy()
    metadata = pd.concat(all_metadata, axis=0)

    print(X.shape)

    clinical_names = np.load(os.path.join(path, "clinical_names.npy"), allow_pickle=True)

    index_of_srs = np.where(clinical_names == srs_name[cohort])[0][0]

    if "asd" in metadata.columns:
        bins = range(int(X[:, index_of_srs].min()), int(X[:, index_of_srs].max()), 2)
        print(bins)
        control_scores = X[:, index_of_srs][metadata["asd"] == 1]
        asd_scores = X[:, index_of_srs][metadata["asd"] == 2]
        stuff_to_hist.append(control_scores)
        stuff_to_hist.append(asd_scores)
    else:
        stuff_to_hist.append(X[:, index_of_srs])
    print("Stats for {}".format(clinical_names[index_of_srs]))
    print("Minimum value : {}".format(min(X[:, index_of_srs])))
    print("Maximum value : {}".format(max(X[:, index_of_srs])))
    print("1st quantile : {}".format(np.quantile(X[:, index_of_srs], 0.25)))
    print("3rd quantile : {}".format(np.quantile(X[:, index_of_srs], 0.75)))
    print("Median value : {}".format(np.median(X[:, index_of_srs])))
    print("Average value : {}".format(np.mean(X[:, index_of_srs])))
    print("Standard deviation : {}".format(np.std(X[:, index_of_srs])))
    print("Only entire values : {}".format(np.array_equal(X[:, index_of_srs], X[:, index_of_srs] // 1)))
    print("Correlation wih age : {}".format(np.corrcoef(np.concatenate((X[:, index_of_srs][:, np.newaxis], metadata[["age"]].values), axis=1), rowvar=False)[0, 1]))
    print("Correlation wih sex : {}".format(np.corrcoef(np.concatenate((X[:, index_of_srs][:, np.newaxis], metadata[["sex"]].values), axis=1), rowvar=False)[0, 1]))
    if "asd" in metadata.columns:
        print("Correlation wih diagnostic : {}".format(np.corrcoef(np.concatenate((X[:, index_of_srs][:, np.newaxis], metadata[["asd"]].values), axis=1), rowvar=False)[0, 1]))
    print("\n")

alpha = 0.5
fig_width = 10
plt.figure(figsize=(fig_width, 3/4 * fig_width))
label = ("EUAIMS", "HBN")
color = (control_color, hbn_color)
if len(stuff_to_hist) == 3:
    label = ("EUAIMS Control", "EUAIMS ASD", "HBN")
    color = (control_color, asd_color, hbn_color)
plt.hist(stuff_to_hist,
        alpha=alpha, bins=20, color=color,
        density=True, label=label)
for x in stuff_to_hist:
    kde = stats.gaussian_kde(x)
    xx = np.linspace(x.min(), x.max(), 1000)
    plt.plot(xx, kde(xx))
plt.xlabel("SRS", size=13, family="serif")
plt.ylabel("Proportion of participants", size=13, family="serif")
plt.legend(title="Cohort", fontsize=12, prop={"family": "serif"},
            title_fontproperties={"family": "serif", "size": 13})



################################### From stat analysis
clinical_names = np.load(os.path.join(args.datasetdir, "clinical_names.npy"), allow_pickle=True)
rois_names = np.load(os.path.join(args.datasetdir, "rois_names.npy"), allow_pickle=True)

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
val_size = n_samples / (len(manager["train"]) + len(manager["test"]))
K = params[name_dataset_train]["K"]
trust_level = params[name_dataset_train]["trust_level"]
stat_params = {
    "runs": validation,
    "samples_per_score": n_discretization_steps,
    "samples_in_run": n_samples,
    "K": K,
    "pvalue_select_thr": trust_level
}

plot_meaningful_areas_per_score_per_metric = False

# carefull, if True, opens as many fig tabs as number of clinical names * number of score + 3-4
plot_latent_space = False

dir_name = "_".join(["_".join([key, str(value)]) for key, value in stat_params.items()])
path_to_save_fig = os.path.join(
    args.dir_experiment, args.run, "figures", dir_name)
if not os.path.isdir(path_to_save_fig):
    os.makedirs(path_to_save_fig)

pvalues = np.load(os.path.join(args.dir_experiment, args.run, "results", dir_name, "pvalues.npy"))
coefs = np.load(os.path.join(args.dir_experiment, args.run, "results", dir_name, "coefs.npy"))

significativity_thr = (0.05 / 444 / 7)

trust_level = validation * trust_level
idx_sign = (pvalues < significativity_thr).sum(0) >= trust_level
rois_names_no_metric = np.array([name.replace("_{}".format(name.split("_")[-1]), "") for name in rois_names])

rois_names_no_metric_unique = np.unique(rois_names_no_metric)

coefs_thr = coefs.copy()
coefs_thr[:, (pvalues < significativity_thr).sum(0) < trust_level] = 0
metrics = ["thickness", "meancurv", "area"]

sources_per_metric = {}
targets_per_metric = {}
values_per_metric = {}
colors_per_metric = {}
for metric in metrics:
    sources_per_metric[metric] = []
    targets_per_metric[metric] = []
    values_per_metric[metric] = []
    colors_per_metric[metric] = []
    for idx, score in enumerate(clinical_names):
        for roi_idx, roi_name in enumerate(rois_names_no_metric_unique):
            roi_with_metric_idx = np.where(rois_names == roi_name + "_{}".format(metric))
            coef = coefs_thr[:, idx, roi_with_metric_idx].mean()
            if coef != 0:
                sources_per_metric[metric].append(idx)
                targets_per_metric[metric].append(roi_idx + len(clinical_names))
                values_per_metric[metric].append(np.abs(coef))
                colors_per_metric[metric].append("rgba(255,0,0,0.4)" if coef > 0 else "rgba(0,0,255,0.4)")
        if plot_meaningful_areas_per_score_per_metric:
            fig = plot_surf(coefs[:, idx].mean(0), metric)
            fig.suptitle("Average influence of {} on {}".format(score, metric))
            plt.savefig(os.path.join(path_to_save_fig, "{}_on_{}".format(score, metric)))
        
#################################### Most linked brain areas across scores (radar) - plot areas petit brain

clinical_names = [name.replace("t1", "").replace("_", " ").strip() for name in clinical_names]
plotting_clinical_names = {
    "euaims": {
        "rbs total": "RBS",
        "srs rawscore": "SRS",
        "adhd hyperimpul parent": "ADHD hyperimpul",
        "adhd inattentiv parent": "ADHD inattentiv",
        "dawba anx": "DAWBA anx",
        "dawba dep": "DAWBA dep",
        "dawba behavdis": "DAWBA behavdis"
    }
}

textfont = textfont = dict(
            size=20,
            family="Droid Serif")
    
for metric in metrics:
    targets, counts = np.unique(targets_per_metric[metric], return_counts=True)
    counts_target = dict(zip(targets, counts))
    counts_target = {k: v for k, v in sorted(counts_target.items(), key=lambda item: item[1], reverse=True)}
    min_connections = 2
    n_most_connected = 7
    thr_count_target = {key: value for key, value in list(counts_target.items())[:n_most_connected] if value >= min_connections}
    areas_to_plot = [rois_names_no_metric_unique[idx - len(clinical_names)] for idx in thr_count_target]
    plot_areas(areas_to_plot, np.arange(len(thr_count_target)) + 0.01)
    fig = go.Figure()
    all_values = []
    all_markers = []
    marker_signif = "star"
    marker_non_signif = "circle"
    for area_idx, (area, count) in enumerate(thr_count_target.items()):
        sources = np.array(sources_per_metric[metric])[np.array(targets_per_metric[metric]) == area]
        values = np.array(values_per_metric[metric])[np.array(targets_per_metric[metric]) == area]
        colors = np.array(colors_per_metric[metric])[np.array(targets_per_metric[metric]) == area]
        if len(np.unique(colors)) > 1:
            print("Signs of the meaninful coefficients differ !")
        new_values = []
        markers = []
        for idx, name in enumerate(clinical_names):
            if idx not in sources:
                roi_name = rois_names_no_metric_unique[area - len(clinical_names)]
                roi_with_metric_idx = np.where(rois_names == roi_name + "_{}".format(metric))
                coef = coefs[:, idx, roi_with_metric_idx].mean()
                coef = np.abs(coef)
                new_values.append(coef)
                markers.append(marker_non_signif)
            else:
                new_values.append(values[sources == idx][0])
                markers.append(marker_signif)

        all_values += new_values
        all_markers += markers
        fig.add_trace(go.Scatterpolar(
            r=new_values + [new_values[0]],
            theta=["<b>" + plotting_clinical_names[name_dataset_train][name] + "</b>" for name in clinical_names + [clinical_names[0]]],
            # fill='toself',
            mode="lines+text",
            legendgroup="roi",
            legendgrouptitle = dict(
                font=dict(
                    size=30,
                    family="Droid Serif"),
                text="<b>Regions of interest</b>"),
            name=rois_names_no_metric_unique[area - len(clinical_names)]
        ))
    all_markers = np.array(all_markers)
    for marker, name in [(marker_non_signif, "non significative"), (marker_signif, "significative")]:
        fig.add_trace(go.Scatterpolar(
            r=np.array(all_values)[all_markers == marker],
            theta=np.array(["<b>" + plotting_clinical_names[name_dataset_train][name] + "</b>" for name in clinical_names*len(thr_count_target)])[all_markers == marker],
            # fill='toself',
            mode="markers",
            legendgroup="significativity",
            legendgrouptitle = dict(
                font=dict(
                    size=30,
                    family="Droid Serif"),
                text="<b>Significativity</b>"),
            marker_symbol=np.array(all_markers)[all_markers == marker],
            marker_size=10,
            marker_color="black",
            name=name
        ))

    fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    showticklabels=False, ticks='',
                    range=[0, max(all_values)+0.003],
                ),
            ),
            font=textfont,
        )
    fig.show()

############################# Meaningful associations plot (flow)

for metric in metrics:
    sorted_scores = [plotting_clinical_names[name_dataset_train][clinical_names[source_idx]] for source_idx in np.array(sources_per_metric[metric])]
    sorted_rois = [rois_names_no_metric_unique[target_idx - len(clinical_names)] for target_idx in np.array(targets_per_metric[metric])]
    sankey_plot = go.Parcats(
        domain={"x": [0.05, 0.9], "y": [0, 1]},
        dimensions=[{"label": "Score", "values": sorted_scores},
                    {"label": "ROI", "values": sorted_rois}],
        counts=np.array(values_per_metric[metric]),
        line={'color': colors_per_metric[metric], 'shape': 'hspline'},
        labelfont=dict(family="Droid Serif", size=28), tickfont=textfont)
    fig = go.Figure(data=[sankey_plot])

    fig.show()


##################################### Latent space plots
if plot_latent_space:
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
    subsets = list(latents_test["subsets"])
    n_components = 3
    n_neighbors = 20
    min_dist = 0.2
    reducer = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist)
    # reducer = PCA(n_components=n_components)
    # reducer = TSNE(n_components=n_components, init="pca")
    latent_to_observe = "clinical_rois"
    reducer.fit(latents_train["subsets"][latent_to_observe][0].cpu().detach().numpy())
    for show_test in (True, False):
        if show_test:
            latents = latents_test["subsets"][latent_to_observe][0].cpu().detach().numpy()
            latent_weigths = latents_test["weights"].cpu().detach().numpy()
            dataset = exp.dataset_test
            data = test_data
        else:
            latents = latents_train["subsets"][latent_to_observe][0].cpu().detach().numpy()
            latent_weigths = latents_train["weights"].cpu().detach().numpy()
            dataset = exp.dataset_train
            data = train_data
        reduced = reducer.transform(latents)

        orig_data = exp.scalers["clinical"].inverse_transform(data["clinical"].cpu().detach().numpy())
        clinical_scores = np.round(orig_data)
        metadatas = {key: [] for key in exp.dataset_test.metadata.columns}
        for idx in range(len(dataset)):
            idx_per_mod = {
                mod: dataset.idx_per_mod[mod][idx] for mod in dataset.modalities}
            ret = {
                mod: dataset.data[mod][idx_per_mod[mod]] for mod in dataset.modalities}
            ret["metadata"] = dataset.metadata.iloc[idx].to_dict()
            for key in dataset.metadata.columns:
                metadatas[key].append(ret["metadata"][key])

        metadata = pd.DataFrame(metadatas)
        metadata["number"] = metadata.index.values
        clinical_scores = pd.DataFrame(clinical_scores, columns=clinical_names)

        for score in clinical_names:
            plot_latent_representations(reduced, clinical_scores, score, True,
                                        "Common for {}".format("test" if show_test else "train"),
                                        n_dims=n_components)
        plot_latent_representations(reduced, metadata, "age", True,
                                    "Common for {}".format("test" if show_test else "train"),
                                    n_dims=n_components)
        plot_latent_representations(reduced, metadata, "sex", False,
                                    "Common for {}".format("test" if show_test else "train"),
                                    n_dims=n_components)
        plot_latent_representations(reduced, metadata, "site", False,
                                        "Common for {}".format("test" if show_test else "train"),
                                        n_dims=n_components)
        if "asd" in metadata.columns:
            plot_latent_representations(reduced, metadata, "asd", False,
                                        "Common for {}".format("test" if show_test else "train"),
                                        n_dims=n_components)

################################## RSA plots
latent_dissimilarity = np.load(os.path.join(args.dir_experiment, args.run, "results", dir_name, "latent_dissimilarity.npy"))
scores_dissimilarity = np.load(os.path.join(args.dir_experiment, args.run, "results", dir_name, "scores_dissimilarity.npy"))
plt.figure()
plt.imshow(latent_dissimilarity)
plt.title("Dissimilarity matrix of the latent space")
plt.savefig(os.path.join(args.dir_experiment, args.run, "figures", "dissimilarity_latent"))
for idx, score in enumerate(clinical_names):
    plt.figure()
    plt.imshow(scores_dissimilarity[idx])
    plt.title("Dissimilarity matrix of {}".format(score))
    plt.savefig(os.path.join(args.dir_experiment, args.run, "figures", "dissimilarity_{}".format(score)))

plt.show()
