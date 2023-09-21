import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from multimodal_cohort.experiment import MultimodalExperiment


dataset = "hbn"
datasetdir = "/neurospin/signatures/2020_deepint/data/fetchers/hbn/clinical-rois"
outdir = "/neurospin/signatures/2020_deepint/experiments/MoPoe/runs/tmp"
run = "hbn_2023_08_22_16_32"

expdir = os.path.join(outdir, run)
# Create a folder to store analysis results 
rsadir = os.path.join(expdir, "rsa")
if not os.path.isdir(rsadir):
    os.mkdir(rsadir)

# We load the flags for the experiments, then the experiment :
# Load the models and their weights, data split indices, scalers, etc
flags_file = os.path.join(expdir, "flags.rar")
if not os.path.isfile(flags_file):
    raise ValueError("You need first to train the model.")
checkpoints_dir = os.path.join(expdir, "checkpoints")
experiment, flags = MultimodalExperiment.get_experiment(
    flags_file, checkpoints_dir)

# Setting some variables
n_models = flags.num_models
clinical_names = np.load(
    os.path.join(datasetdir, "clinical_names.npy"), allow_pickle=True)
modalities = ["clinical", "rois"]

# Covariates in the metadata we want to study
cov_names = ["age", "sex", "site"]
if dataset == "euaims":
    cov_names.append("fsiq")
categorical_covs = ["sex", "site"]

# If we want to bootstrap test subjects for each models to have more robust
# metric, you can set n_validation to more than 1 and s_subjects to less than 301
n_validation = 1
n_subjects = 301

# Choose weither we sample the latent representations according to the
# posterior distribution or if we take the mean
sample_latents = False 

# Keys to access different parts of the latent space
latent_names = ["joint", "clinical_rois", "clinical_style", "rois_style"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
array_to_store_stuff = np.zeros((n_models, len(latent_names), n_validation, 
                                 len(clinical_names) + len(cov_names)))
for model_idx in range(n_models):
    trainset = experiment.dataset_train
    testset = experiment.dataset_test
    model = experiment.models
    if n_models > 1:
        trainset = trainset[model_idx]
        testset = testset[model_idx]
        model = model[model_idx]

    model = model.to(device)
    model.eval()
    for val_idx in tqdm(range(n_validation)):
        testloader = DataLoader(
            testset, batch_size=n_subjects, shuffle=True, num_workers=0)
        data = {}
        dataiter = iter(testloader)

        # We look for the first batch with all modalities. Here we only do it
        # for test data, but we could do it for train data, specially if we
        # want to train a predictor on train representations
        while True:
            data, _, metadata = next(dataiter)
            if all([mod in data.keys() for mod in modalities]):
                break
        
        # We cast them in tensor for input data, and to numpy for metadata
        for idx, mod in enumerate(modalities):
            data[mod] = Variable(data[mod]).to(device).float()
        for column in metadata.keys():
            if type(metadata[column]) is list:
                metadata[column] = np.array(metadata[column])
            else:
                metadata[column] = metadata[column].cpu().detach().numpy()

        # We extract the latents and compute the metrics we want on it
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
                # Do things on it, now it is a numpy array
print("Finished !")
