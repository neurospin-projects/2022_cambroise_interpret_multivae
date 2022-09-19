# MoPoE-VAE
This is the official code for the ISBI 2023 "blabla".

If you have any questions about the code or the paper, we are happy to help!

## Preliminaries

This code was developed and tested with:
- Python version 3.5.6
- PyTorch version 1.4.0
- CUDA version 11.0
- The conda environment defined in `environment.yml`

First, set up the conda enviroment as follows:
```bash
conda env create -f environment.yml  # create conda env
conda activate mopoe                 # activate conda env
```
or install the requirements in your own environment. 

In order to be able to run the experiments, you need to have access to HBN or EUAIMS data. Then, you must provide each script with some paths :
```
--datasetdir 
```
is the path to the folder that contains the data. It must contain at least 5 files :
- rois_data.npy, an array with 2 dimensions, the first corresponding to the subjects, the second to the different metric for each roi
- rois_subjects.npy, the list of subjects with the same ordering as in the previous file
- roi_names.npy, the list of feature names for the "roi_data" file, with the same ordering as its columns
- clinical_data.npy, an array with 2 dimensions, the first corresponding to the subjects, the second to the different score values
- clinical_subjects.npy, the list of subjects with the same ordering as in the previous file
- clinical_names.npy, the list of feature names for the "clinical_data" file, with the same ordering as its columns
- metadata.tsv, a table containing the metadata. It must contain at east 4 columns:
    "participant_id" with the id of the subjects, corresponding to the "_subjects" files.
    "sex" with numerically encoded sex
    "age" with continuous age
    "site" with acquisition site names
    for eauims, it can be good to have "asd" containing the 1-2 encoded diagnosis values (for the histogram)

## Experiments

Experiments can be started by running the scripts. 
To choose between running the MVAE, MMVAE, and MoPoE-VAE, one needs to
change the script's `METHOD` variabe to "poe", "moe", or "joint\_elbo"
respectively.  By default, each experiment uses `METHOD="joint_elbo"`.

### training the model
```bash
DATASET="$DATASET" DIR_EXPERIMENT="$DIR_EXPERIMENT" DATASETDIR="$DATASETDIR" ./train_mopoe
```

### running analysis
```bash
RUN="$RUN_NAME" DIR_EXPERIMENT="$DIR_EXPERIMENT" DATASETDIR="$DATASETDIR" ./launch_analysis
```

### create figures
```bash
RUN="$RUN_NAME" DIR_EXPERIMENT="$DIR_EXPERIMENT" DATASETDIR="$DATASETDIR" ./create_figures
```
