import os
import glob
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable


import numpy as np
import pandas as pd
from argparse import ArgumentParser

import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from scipy.stats import kendalltau, pearsonr

from multimodal_cohort.dataset import DataManager, MissingModalitySampler
from multimodal_cohort.experiment import MultimodalExperiment


parser = ArgumentParser()
parser.add_argument("--run", type=str)
parser.add_argument("--datasetdir", type=str)
parser.add_argument("--dir_experiment", type=str)
parser.add_argument("--test", type=str, default=None)
args = parser.parse_args()


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def make_regression(df, x_name, y_name, other_cov_names=[], groups_name=None, method="fixed"):
    formula = "{} ~ {}".format(y_name, x_name)
    formula = " + ".join([formula] + other_cov_names)
    idx_of_beta = x_name
    subjects_betas = None
    if method == "fixed":
        est = sm.OLS.from_formula(formula, data=df)
    elif method == "mixed":
        est = sm.MixedLM.from_formula(formula, data=df, groups=groups_name)
    elif method == "hierarchical":
        lv1 = [[group_lab, sm.OLS.from_formula(formula, group_df).fit().params[x_name]]
               for group_lab, group_df in df.groupby(groups_name)]
        lv1 = pd.DataFrame(lv1, columns=[groups_name, 'beta'])
        subjects_betas = lv1
        est = sm.OLS.from_formula("beta ~ 1", data=lv1)
        idx_of_beta = "Intercept"
    results = est.fit()
    return results.pvalues[idx_of_beta], results.params[idx_of_beta], subjects_betas
    

name_dataset_train = args.run.split("_")[0]
modalities = ["clinical", "rois"]

# with open(os.path.join(args.dir_experiment, args.run, "experiment.pickle"), "rb") as f:
#     exp = pickle.load(f)
print("Loading data...")
flags_file = os.path.join(args.dir_experiment, args.run, "flags.rar")
if not os.path.isfile(flags_file):
    raise ValueError("You need first to train the model.")
alphabet_file = os.path.join(os.getcwd(), "alphabet.json")
checkpoints_files = glob.glob(
    os.path.join(args.dir_experiment, args.run, "checkpoints", "*", "mm_vae"))
if len(checkpoints_files) == 0:
    raise ValueError("You need first to train the model.")
checkpoints_files = sorted(
    checkpoints_files, key=lambda path: int(path.split(os.sep)[-2]))
checkpoint_file = checkpoints_files[-1]
print(f"Restoring weights: {checkpoint_file}")
all_particpants = []
all_idx = []
exp, flags = MultimodalExperiment.get_experiment(
    flags_file, alphabet_file, checkpoint_file)


print(len(exp.dataset_train))
print(len(exp.dataset_test))
if "allow_missing_blocks" in vars(exp.flags) and exp.flags.allow_missing_blocks:
    sampler = MissingModalitySampler(exp.dataset_train, batch_size=4096)
    loader_train = DataLoader(exp.dataset_train, batch_sampler=sampler, num_workers=8)
else:
    loader_train = DataLoader(exp.dataset_train, shuffle=True, batch_size=4096, num_workers=8)

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
    sampler_test = MissingModalitySampler(dataset_test, batch_size=1024)
    loader_test = DataLoader(dataset_test, batch_sampler=sampler_test, num_workers=8)
else:
    loader_test = DataLoader(dataset_test, shuffle=True, batch_size=1024, num_workers=8)

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
    "validation": 20, "n_discretization_steps": 200,
    "n_samples": 49, "K": 1000, "trust_level": 1,
    "method": "hierarchical"}
params["hbn"] = {
    "validation": 50, "n_discretization_steps": 200,
    "n_samples": 150, "K": 1000, "trust_level": 1,
    "method": "hierarchical"}
validation = params[name_dataset_train]["validation"]
n_discretization_steps = params[name_dataset_train]["n_discretization_steps"]
n_samples = params[name_dataset_train]["n_samples"]
reg_method = params[name_dataset_train]["method"]

seed = 1037
np.random.seed(seed)
torch.manual_seed(seed)

if not os.path.isdir(os.path.join(args.dir_experiment, args.run, "results")):
    os.makedirs(os.path.join(args.dir_experiment, args.run, "results"))

if gradient_over_scores:

    val_size = n_samples / (len(exp.dataset_train) * (5/4))
    K = params[name_dataset_train]["K"]
    trust_level = params[name_dataset_train]["trust_level"]
    stat_params = {
        "runs": validation,
        "samples_per_score": n_discretization_steps,
        "samples_in_run": n_samples,
        "K": K,
        "method": reg_method
    }
    
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
        if reg_method == "hierarchical":
            all_coefs = []
            anova_pvalues = np.zeros((validation, n_scores, n_rois))
        # correlations_between_init_error_and_mixed_param = np.zeros(validation)
        plot_stuff = False
        if validation == 1:
            plot_stuff = True
        for val_step in range(validation):
            print("Validation step : {}".format(val_step))
            if args.test is not None:
                dataset_test = manager.train_dataset[val_step]["valid"]
                test_size = len(dataset_test)
                loader_test = DataLoader(dataset_test, batch_size=test_size,
                                shuffle=True, num_workers=8)
            else:# "allow_missing_blocks" not in vars(exp.flags):
                loader_test = DataLoader(dataset_test, batch_size=n_samples,
                                shuffle=True, num_workers=8)
            # else:
            #     sampler_test = MissingModalitySampler(dataset_test, batch_size=n_samples,
            #                                           stratify=["age", "sex", "site"],
            #                                           discretize=["age"])
            #     loader_test = DataLoader(dataset_test, batch_sampler=sampler_test, num_workers=8)
            batch_idx = 0
            data = {}
            for idx, batch in enumerate(loader_test):
                if args.test is None:
                    local_metadata = batch[2]
                    batch = batch[0]
                if all([mod in batch.keys() for mod in modalities]) and batch_idx == 0:
                    batch_idx += 1
                    for k, m_key in enumerate(modalities):
                        data[m_key] = Variable(batch[m_key]).to(exp.flags.device).float()
                        test_size = len(data[m_key])
                    metadata = local_metadata
            values = torch.FloatTensor(np.linspace(min_per_score, max_per_score, n_discretization_steps))

            all_errors = np.zeros((test_size, n_scores, n_discretization_steps,
                                            n_rois))
            score_values = np.zeros((test_size, n_scores, n_discretization_steps))
            # initial_errors = np.zeros((test_size, n_scores, n_rois))
            # subject_errors = np.zeros((test_size, n_scores, n_rois))
            linear_gradient = False
            if not linear_gradient:
                clinical_loc_hats = []
                clinical_scale_hats = []
                for k in range(K):
                    reconstructions = exp.mm_vae(data, sample_latents=True)["rec"]
                    clinical_loc_hats.append(reconstructions["clinical"].loc.unsqueeze(0))
                    clinical_scale_hats.append(reconstructions["clinical"].scale.unsqueeze(0))
                reconstruction_stable = exp.mm_vae(data, sample_latents=False)["rec"]
                rois_hat = reconstruction_stable["rois"].loc
                clinical_hat_mean = torch.cat(clinical_loc_hats).mean(0)
                clinical_std_mean = torch.cat(clinical_scale_hats).mean(0)
                initial_rois_errors = np.repeat(
                    (data["rois"] - rois_hat).cpu().detach().numpy()[:, np.newaxis],
                    n_discretization_steps, axis=1)
                initial_clinical_errors = np.repeat(
                    (data["clinical"] - clinical_hat_mean).cpu().detach().numpy()[:, np.newaxis],
                    n_discretization_steps, axis=1)
                dist = torch.distributions.Normal(
                    clinical_hat_mean,
                    # torch.cat(clinical_loc_hats).std(0))
                    clinical_std_mean)
                score_values = dist.sample(torch.Size([n_discretization_steps]))
                score_reconstructions = clinical_hat_mean.repeat(1, n_discretization_steps, 1)
                clinical_errors = (score_values - data["clinical"].repeat(n_discretization_steps, 1, 1)).cpu().detach().numpy()
                
                if plot_stuff:
                    import matplotlib.pyplot as plt
                    from matplotlib import colors
                    import scipy
                    subj_idx = np.random.randint(test_size, size=8)
                    for score_idx in range(len(clinical_names)):
                        plt.figure()
                        sigmax = 0
                        for i, (mu, sig) in enumerate(zip(torch.cat(clinical_loc_hats).mean(0)[subj_idx, score_idx].cpu().detach().tolist(),
                                                        torch.cat(clinical_scale_hats).mean(0)[subj_idx, score_idx].cpu().detach().tolist())):
                                                        #   torch.cat(clinical_loc_hats).std(0)[subj_idx, score_idx].cpu().detach().tolist())):
                            x = np.linspace(mu - 3 * sig, mu + 3 * sig, 120)
                            y = scipy.stats.norm.pdf(x, mu, sig)
                            sigmax = max(sigmax, sig)
                            plt.plot(x, y, color=list(colors.TABLEAU_COLORS)[i])
                            
                            plt.vlines(data["clinical"][subj_idx[i], score_idx], 0, 1, color=list(colors.TABLEAU_COLORS)[i])
                            plt.vlines(reconstruction_stable["clinical"].loc.detach().numpy()[subj_idx[i], score_idx], 0, 1, color=list(colors.TABLEAU_COLORS)[i], linestyles="dashed")
                        plt.title(clinical_names[score_idx])
                        plt.xlim(data["clinical"][subj_idx, score_idx].min() - 3 * sigmax, data["clinical"][subj_idx, score_idx].max() + 3 * sigmax)
                        plt.figure()
                        stuff_to_hist = (data["clinical"].detach().numpy()[:, score_idx],
                                         clinical_hat_mean.detach().numpy()[:, score_idx],
                                         score_values.cpu().detach().numpy()[:, :, score_idx].flatten())
                        # color = ("b", "o")
                        label = (clinical_names[score_idx], "Reconstruction", "Simulated")
                        plt.hist(stuff_to_hist,
                                bins=15, label=label,
                                color=list(colors.TABLEAU_COLORS)[:3], density=True)
                        from scipy import stats
                        for i, x in enumerate(stuff_to_hist):
                            kde = stats.gaussian_kde(x)
                            xx = np.linspace(x.min(), x.max(), 1000)
                            plt.plot(xx, kde(xx), c=list(colors.TABLEAU_COLORS)[i], label=label[i])
                        plt.legend()
                # Writing tables
                index_of_srs = (clinical_names.tolist().index("SRS_Total") if name_dataset_train == "hbn"
                            else clinical_names.tolist().index("t1_srs_rawscore"))
                srs_modifications = dict(
                    participant_num=np.repeat(np.arange(test_size)[:, np.newaxis], n_discretization_steps, axis=1).flatten(),
                    score=np.repeat(data["clinical"].cpu().detach().numpy()[:, index_of_srs, np.newaxis], n_discretization_steps, axis=1).flatten(),
                    score_mean_reconstruction=score_reconstructions.cpu().detach().numpy()[:, :, index_of_srs].flatten(),
                    score_std_reconstruction=clinical_std_mean.repeat(1, n_discretization_steps, 1).cpu().detach().numpy()[:, :, index_of_srs].flatten(),
                    sampled_score=np.swapaxes(score_values.detach().numpy(), 0, 1)[:, :, index_of_srs].flatten())
                metadata_df = pd.DataFrame(columns=list(metadata.keys()))
                for column in metadata.keys():
                    if type(metadata[column]) is list:
                        metadata_df[column] = np.array(metadata[column])
                    else:
                        metadata_df[column] = metadata[column].cpu().detach().numpy()
                    srs_modifications[column] = np.repeat(np.array(metadata_df[column])[:, np.newaxis], n_discretization_steps, axis=1).flatten()

            new_rois_hats = np.zeros((test_size, n_discretization_steps, n_rois, n_scores))
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
                    reconstructions = exp.mm_vae(new_data, sample_latents=True)["rec"]
                    new_rois_hat = reconstructions["rois"].loc
                    new_rois_hats[:, step, :, idx] = new_rois_hat.cpu().detach().numpy()
                    all_errors[:, idx, step] = (new_rois_hat - data["rois"]).cpu().detach().numpy()
            if plot_stuff:
                for roi_idx, roi_name in enumerate(rois_names):
                    srs_modifications[roi_name + "_avatar"] = new_rois_hats[:, :, roi_idx, index_of_srs].flatten()
                    srs_modifications[roi_name + "_reconstruction"] = np.repeat(rois_hat.cpu().detach().numpy()[:, roi_idx, np.newaxis], n_discretization_steps, axis=1).flatten()
                    srs_modifications[roi_name] = np.repeat(data["rois"].cpu().detach().numpy()[:, roi_idx, np.newaxis], n_discretization_steps, axis=1).flatten()
                pd.DataFrame(srs_modifications).to_csv(os.path.join(args.dir_experiment, args.run, "results", dir_name, "srs_perturbation.tsv"), sep="\t", index=False)
            if linear_gradient:
                score_values = np.swapaxes(score_values, 1, 2)
            else:
                score_values = np.swapaxes(score_values.detach().numpy(), 0, 1)
                clinical_errors = np.swapaxes(clinical_errors, 0, 1)
            selected_scores = [0, 1, 3, 4, 5, 6]
            selected_rois = [0, 1, 2, 3, 4, 5]
            points_to_scatter = {
                clinical_names[score_idx]: dict() for score_idx in selected_scores}

            things_to_test = np.zeros((n_scores, n_rois, test_size, 2))
            other_stuff = np.zeros((n_scores, n_rois))
            if reg_method == "hierarchical":
                all_coefs.append([])
            for score_idx in range(n_scores):
                base_df = pd.DataFrame(dict(
                    participant_id=np.repeat(np.array(metadata["participant_id"])[:, np.newaxis], n_discretization_steps, axis=1).flatten(),
                    score=np.repeat(data["clinical"].cpu().detach().numpy()[:, score_idx, np.newaxis], n_discretization_steps, axis=1).flatten(),
                    score_mean_reconstruction=score_reconstructions.cpu().detach().numpy()[:, :, score_idx].flatten(),
                    score_std_reconstruction=clinical_std_mean.repeat(1, n_discretization_steps, 1).cpu().detach().numpy()[:, :, score_idx].flatten(),
                    sampled_score=score_values[:, :, score_idx].flatten()))
                base_df["sampled_true_score_diff"] = base_df["sampled_score"] - base_df["score"]
                base_df["reconstruction_true_score_diff"] = base_df["score_mean_reconstruction"] - base_df["score"]
                base_df["sampled_reconstruction_score_diff"] = base_df["sampled_score"] - base_df["score_mean_reconstruction"]
                if reg_method == "hierarchical":
                    all_coefs[val_step].append(metadata_df[["participant_id", "site"]].copy())
                for roi_idx in range(n_rois):
                    df = base_df.copy()
                    df["roi_avatar"] = new_rois_hats[:, :, roi_idx, score_idx].flatten()
                    df["roi_reconstruction"] = np.repeat(rois_hat.cpu().detach().numpy()[:, roi_idx, np.newaxis], n_discretization_steps, axis=1).flatten()
                    df["roi"] = np.repeat(data["rois"].cpu().detach().numpy()[:, roi_idx, np.newaxis], n_discretization_steps, axis=1).flatten()
                    # If needed add other columns to the df, containing differences between columns for instance
                    # df["avatar_true_roi_diff"] = df["roi_avatar"] - df["roi"]
                    # df["true_reconstruction_roi_diff"] = df["roi"] - df["roi_reconstruction"]
                    # df["avatar_reconstruction_roi_diff"] = df["roi_avatar"] - df["roi_reconstruction"]
                    
                    # Use df with variable names
                    results = make_regression(df, "sampled_score", "roi_avatar", groups_name="participant_id", method=reg_method)
                    pvalues[val_step, score_idx, roi_idx], coefs[val_step, score_idx, roi_idx], all_betas = results
                    if reg_method == "hierarchical":
                        roi_name = rois_names[roi_idx].replace("&", "_").replace("-", "_")
                        all_betas.rename(columns={"beta": roi_name}, inplace=True)
                        all_coefs[val_step][score_idx] = all_coefs[val_step][score_idx].join(all_betas.set_index("participant_id"), on="participant_id")

                        anova_ols = sm.OLS.from_formula("{} ~ C(site)".format(roi_name), data=all_coefs[val_step][score_idx]).fit()
                        anova_res = anova_lm(anova_ols, type=2)
                        anova_pvalues[val_step, score_idx, roi_idx] = anova_res["PR(>F)"]["C(site)"]
                    
                    # things_to_test[score_idx, roi_idx, :, 0] = [value.to_numpy()[0] for value in res.random_effects.values()]
                    # things_to_test[score_idx, roi_idx, :, 1] = rois_hat.cpu().detach().numpy()[:, roi_idx] - data["rois"].cpu().detach().numpy()[:, roi_idx]
                    if score_idx in selected_scores and roi_idx in selected_rois:
                        points_to_scatter[clinical_names[score_idx]][rois_names[roi_idx]] = dict(
                            x = score_values[:, :, score_idx].flatten(),
                            # x = clinical_errors[:, :, score_idx].flatten() + initial_clinical_errors[:, :, score_idx].flatten(),
                            # y = df["avatar_reconstruction_roi_diff"]#all_errors[:, score_idx, :, roi_idx].flatten() + initial_rois_errors[:, :, roi_idx].flatten()
                            y = df["roi_avatar"]
                        )
                    # other_stuff[score_idx, roi_idx] = res.params[2]
                    # print(res.params[2])
                    # subject_errors[:, score_idx, roi_idx] = res.params[2]
            # print(other_stuff.mean())
            # print(np.corrcoef(np.reshape(things_to_test, (n_scores * n_rois * test_size, 2)), rowvar=False))
        if plot_stuff:
            import matplotlib.pyplot as plt
            # plt.figure()
            # plt.hist(other_stuff.flatten())
            selected_scores = [index_of_srs]
            # selected_rois = [5]
            fig, axes = plt.subplots(len(selected_scores), len(selected_rois), sharey=True, figsize=(5 * len(selected_rois), 3 * len(selected_scores)))
            for score_idx, (score_name, rois) in enumerate(points_to_scatter.items()):
                for roi_idx, (roi_name, points) in enumerate(rois.items()):
                    if score_idx == selected_scores[0]:
                        # axes[roi_idx].scatter(points["x"], points["y"], c=np.repeat(np.arange(test_size)[:, np.newaxis], n_discretization_steps, axis=1).flatten())
                        axes[roi_idx].scatter(srs_modifications["sampled_score"],
                                            srs_modifications["{}_avatar".format(roi_name)] - srs_modifications["{}_reconstruction".format(roi_name)],
                                            c=np.repeat(np.arange(test_size)[:, np.newaxis], n_discretization_steps, axis=1).flatten())
                    # if score_idx == 0:
                    #     axes[score_idx, roi_idx].set_title(roi_name)
                        axes[roi_idx].set_title(roi_name)
                    # if roi_idx == 0:
                    #     axes[score_idx, roi_idx].set_ylabel(score_name)
            plt.show()
        np.save(os.path.join(args.dir_experiment, args.run, "results", dir_name, "pvalues.npy"), pvalues)
        np.save(os.path.join(args.dir_experiment, args.run, "results", dir_name, "coefs.npy"), coefs)
        np.save(os.path.join(args.dir_experiment, args.run, "results", dir_name, "all_coefs.npy"), all_coefs)
        np.save(os.path.join(args.dir_experiment, args.run, "results", dir_name, "anova_pvalues.npy"), anova_pvalues)
    else:
        pvalues = np.load(os.path.join(args.dir_experiment, args.run, "results", dir_name, "pvalues.npy"))
        coefs = np.load(os.path.join(args.dir_experiment, args.run, "results", dir_name, "coefs.npy"))
        if os.path.exists(os.path.join(args.dir_experiment, args.run, "results", dir_name, "all_coefs.npy")):
            all_coefs = np.load(os.path.join(args.dir_experiment, args.run, "results", dir_name, "all_coefs.npy"), allow_pickle=True)
            anova_pvalues = np.load(os.path.join(args.dir_experiment, args.run, "results", dir_name, "anova_pvalues.npy"))

    significativity_thr = (0.05 / 444 / 7)
    
    print((pvalues < significativity_thr).sum())

    for prop in np.linspace(0, 1, 11):
        print(prop)
        print(((pvalues < significativity_thr).sum(0) >= validation*prop).sum())
        print(((pvalues < significativity_thr).sum(0) >= validation*prop).sum(1))

    print()
    print(coefs.max((0, 2)))
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

    if os.path.exists(os.path.join(args.dir_experiment, args.run, "results", dir_name, "all_coefs.npy")):
        print(anova_pvalues.min())
        print(anova_pvalues.max())
        print(anova_pvalues.mean(0).min())
        print(anova_pvalues.mean(0).max())
        print(anova_pvalues[:, idx_sign].min())
        print(anova_pvalues[:, idx_sign].max())
        print(anova_pvalues[:, idx_sign].mean(0).min())
        print(anova_pvalues[:, idx_sign].mean(0).max())
    
    # Plots to investigate coefs
    # import matplotlib.pyplot as plt
    # for score_idx, score in enumerate(clinical_names):
    #     score_scale = exp.scalers["clinical"].scale_[score_idx]
    #     roi_scales = exp.scalers["rois"].scale_
    #     true_coefs = (roi_scales / score_scale * coefs[:, score_idx].mean(0))
    #     if idx_sign[score_idx].sum() > 0:
    #         for p in range(0, 6):
    #             print(np.sum(np.abs(true_coefs) > 1 / 10 ** p))
    #         for metric in ["thickness", "meancurv", "area"]:
    #             if sum([idx_sign[score_idx, roi_idx] for roi_idx in range(len(rois_names)) if metric in rois_names[roi_idx]]) > 0:
    #                 plt.figure()
    #                 plt.hist(true_coefs[np.array([roi_idx for roi_idx in range(len(rois_names)) if (metric in rois_names[roi_idx] and idx_sign[score_idx, roi_idx])])], bins=20)
    #                 plt.title("Histogram of effects of {} on {}".format(score, metric))
    # plt.show()
##################### RSA
compute_similarity_matrices = True
validation = 1
n_samples = len(dataset_test)
sample_latents = False
if compute_similarity_matrices:
    cov_names = ["age", "sex", "site"]
    if name_dataset_train == "euaims" or args.test == "euaims":
        cov_names.append("fsiq")
    latent_names = ["joint", "clinical_rois", "clinical_style", "rois_style"]
    correlations = np.zeros((len(latent_names), validation, len(clinical_names) + len(cov_names)))
    correlationpvalues = np.zeros((len(latent_names), validation, len(clinical_names) + len(cov_names)))
    kendalltaus = np.zeros((len(latent_names), validation, len(clinical_names) + len(cov_names)))
    kendallpvalues = np.zeros((len(latent_names), validation, len(clinical_names) + len(cov_names)))
    last_latents = np.zeros((len(latent_names), 0)).tolist()
    for val_step in range(validation):
        if args.test is not None:
            dataset_test = manager.train_dataset[val_step]["valid"]
            test_size = len(dataset_test)
            loader_test = DataLoader(dataset_test, batch_size=test_size,
                            shuffle=True, num_workers=8)
        # elif "allow_missing_blocks" not in vars(exp.flags):
        #     loader_test = DataLoader(dataset_test, batch_size=n_samples,
        #                     shuffle=True, num_workers=8)
        # else:
        #     sampler_test = MissingModalitySampler(dataset_test, batch_size=n_samples,
        #                                           stratify=["age", "sex", "site"],
        #                                           discretize=["age"])
        #     loader_test = DataLoader(dataset_test, batch_sampler=sampler_test, num_workers=8)
        else:
            loader_test = DataLoader(dataset_test, batch_size=n_samples,
                            shuffle=True, num_workers=8)
        batch_idx = 0
        data = {}
        for idx, batch in enumerate(loader_test):
            if args.test is None:
                local_metadata = batch[2]
                batch = batch[0]
            # if "clinical" in batch.keys() and "rois" not in batch.keys() and batch_idx == 0:
            if all([mod in batch.keys() for mod in modalities]) and batch_idx == 0:
                batch_idx += 1
                for k, m_key in enumerate(modalities):
                    data[m_key] = Variable(batch[m_key]).to(exp.flags.device).float()
                    test_size = len(data[m_key])
                metadata = local_metadata
        idx_triu = np.triu(np.ones((test_size, test_size), dtype=bool), 1)

        for latent_idx, latent_name in enumerate(latent_names):
            latents = exp.mm_vae(data, sample_latents=sample_latents)["latents"]
            if latent_name == "joint":
                latents = latents["joint"]
            elif "style" in latent_name:
                latents = latents["modalities"][latent_name]
            else:
                latents = latents["subsets"][latent_name]
            if sample_latents:
                latents = exp.mm_vae.reparameterize(latents[0], latents[1])
            else:
                latents = latents[0]
            # print(latents.shape)
            n_scores = data["clinical"].shape[1]
            latent_dissimilarity = np.zeros((test_size, test_size))
            for i in range(test_size):
                for j in range(i + 1, test_size):
                    latent_dissimilarity[i, j] = np.linalg.norm((latents[i].long() - latents[j].long()).cpu().detach().numpy())

            scores_dissimilarity = np.zeros((n_scores + len(cov_names), test_size, test_size))
            for idx, score in enumerate(clinical_names.tolist() + cov_names):
                for i in range(test_size):
                    for j in range(i + 1, test_size):
                        if score in clinical_names:
                            scores_dissimilarity[idx, i, j] = np.abs(data["clinical"][i, idx] - data["clinical"][j, idx])
                        elif score in ["sex", "site"]:
                            scores_dissimilarity[idx, i, j] = 0 if metadata[score][i] == metadata[score][j] else 1
                        else:
                            scores_dissimilarity[idx, i, j] = np.abs(metadata[score][i] - metadata[score][j])
                # print(scores_dissimilarity[idx])

                r_scipy = pearsonr(scores_dissimilarity[idx][idx_triu], latent_dissimilarity[idx_triu])
                r, r_pvalue = r_scipy
                correlations[latent_idx, val_step, idx] = r
                correlationpvalues[latent_idx, val_step, idx] = r_pvalue
                # order_scores = scores_dissimilarity[idx][idx_triu].argsort(axis=None)
                # ranks_scores = np.empty_like(order_scores)
                # ranks_scores[order_scores] = np.arange(len(order_scores))

                # order_latent = latent_dissimilarity[idx_triu].argsort(axis=None)
                # ranks_latent = np.empty_like(order_latent)
                # ranks_latent[order_latent] = np.arange(len(order_latent))

                tau, pvalue = kendalltau(scores_dissimilarity[idx][idx_triu], latent_dissimilarity[idx_triu])
                # tau, pvalue = kendalltau(ranks_scores, ranks_latent)
                kendalltaus[latent_idx, val_step, idx] = tau
                kendallpvalues[latent_idx, val_step, idx] = pvalue

            # np.save(os.path.join(args.dir_experiment, args.run, "results", dir_name, "latent_dissimilarity.npy"), latent_dissimilarity)
            # np.save(os.path.join(args.dir_experiment, args.run, "results", dir_name, "scores_dissimilarity.npy"), scores_dissimilarity)
    for latent_idx, latent_name in enumerate(latent_names):
        rsa_results = pd.DataFrame.from_dict(data={"correlation": correlations[latent_idx].mean(0),
                                                    "correlation_std": correlations[latent_idx].std(0),
                                                    "correlation_pvalue": correlationpvalues[latent_idx].mean(0),
                                                    "correlation_pvalue_std": correlationpvalues[latent_idx].std(0),
                                                    "kendalltau": kendalltaus[latent_idx].mean(0),
                                                    "kendalltau_std": kendalltaus[latent_idx].std(0),
                                                    "kendalltau_pvalue": kendallpvalues[latent_idx].mean(0),
                                                    "kendalltau_pvalue_std": kendallpvalues[latent_idx].std(0)},
                                            columns=clinical_names.tolist() + cov_names,
                                            orient="index")
                                    
        rsa_results.to_csv(os.path.join(args.dir_experiment, args.run, "results", "rsa_results_{}_{}_validation_{}_samples.tsv".format(latent_name, validation, test_size)), sep="\t")    
