![Pep8](https://github.com/neurospin-projects/2022_cambroise_interpret_multivae/actions/workflows/pep8.yml/badge.svg)


# Multi-view variational autoencoders allow for interpretability leveraging Digital Avatars: application to the HBN cohort

\:+1: If you are using the code please add a star to the repository :+1:

The availability of multiple data types provides a rich source of information
and holds promise for learning representations that generalize well across
multiple modalities. Multimodal data naturally grants additional
self-supervision in the form of shared information connecting the
different data types. Further, the understanding of different modalities and
the interplaybetween data types are non-trivial research questions and
long-standing goals in machine learning research.

This is the official repository for our ISBI 2023 paper associated code.
If you have any question about the code or the paper, we are happy to help!


## Preliminaries

This code was developed and tested with:
- Python version 3.9.13
- PyTorch version 1.13.0
- CUDA version 11.0
- The conda environment defined in `environment.yml`

First, set up the conda enviroment as follows:

```
conda env create -f environment.yml
conda activate mopoe
```

or install the requirements in your own environment. 

In order to be able to run the experiments, you need to have access to HBN or
EUAIMS data. Then, you must provide each script the path to these data setting
the `--dataset` and `--datasetdir` parameters.
The data folder must contains at least 5 files:
- **rois_data.npy**: an array with 2 dimensions, the first corresponding to
  the subjects, the second to the different metric for each ROI.
- **rois_subjects.npy**: the list of subjects with the same ordering as
  in the previous file.
- **roi_names.npy**: the list of feature names for the `roi_data` file, with
  the same ordering as its columns.
- **clinical_data.npy**: an array with 2 dimensions, the first corresponding
  to the subjects, the second to the different score values.
- **clinical_subjects.npy**: the list of subjects with the same ordering as
  in the previous file.
- **clinical_names.npy** the list of feature names for the `clinical_data`
  file, with the same ordering as its columns.
- **metadata.tsv**: a table containing the metadata. It must contain at least
  4 columns: `participant_id` with the id of the subjects, corresponding
  to the `_subjects` files, `sex` with numerically encoded sex, `age` with
  continuous age, and `site` with acquisition site names. In EUAIMS, it can
  be good to have `asd` containing the 1-2 encoded diagnosis values (for the
  histogram).


## Experiments

To choose between running the MVAE, MMVAE, and MoPoE-VAE, one needs to
change the script's `--method` variabe to `poe`, `moe`, or `joint_elbo`
respectively. By default, `joint_elbo` is selected.


Perform the proposed Digital Avatars Aanalysis (DAA) on HBN by running
the following commands in a shell:

```
cd experiments
export DATASETDIR=/path/to/my/dataset
export OUTDIR=/path/to/the/output/directory

./experiments train --dataset hbn --datasetdir $DATASETDIR --outdir $OUTDIR
--latent_dim 20 --input_dims 7,444 --beta 1 --batch_size 256
--likelihood normal --learning_rate 0.002 --num_epochs 550
--learn_output_scale --allow_missing_blocks

export RUN=my_run_id

# /!\ Long run /!\
./experiments daa --dataset hbn --datasetdir $DATASETDIR --outdir $OUTDIR --run $RUN --n_samples 150 --n_validation 20 --trust_level 0.7
./experiments rse --dataset hbn --datasetdir $DATASETDIR --outdir $OUTDIR --run $RUN

./experiments daa-plot-most-connected --dataset hbn --datasetdir $DATASETDIR --outdir $OUTDIR --run $RUN
./experiments daa-plot-score-metric --dataset hbn --datasetdir $DATASETDIR --outdir $OUTDIR --run $RUN --score SRS_Total --metric thickness
./experiments rsa-plot --dataset hbn --datasetdir $DATASETDIR --outdir $OUTDIR --run $RUN
./experiments hist-plot --datasets hbn,euaims --datasetdirs $DATASETDIR1,$DATASETDIR2 --scores SRS_Total,t1_srs_rawscore --outdir $OUTDIR
```

Citation
========

If you are using this repository or building research on it, it would be great to cite our paper :
Corentin Ambroise, Antoine Grigis, Edouard Duchesnay, Vincent Frouin (2023).
[Multi-view variational autoencoders allow for interpretability leveraging Digital Avatars: application to the HBN cohort](https://ieeexplore.ieee.org/xpl/conhome/1000080/all-proceedings). ISBI 2023

This works is dervived from the following papers:

Thomas M. Sutter, Imant Daunhawer, Julia E Vogt (2021).
[Generalized Multimodal ELBO](https://openreview.net/pdf?id=5Y21V0RDBV). ICLR.

