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
import subprocess
from color_utils import print_title, print_command


def isbi23_train(dataset, datasetdir, outdir, python_cmd="python"):
    """ Train the ISBI23 model.

    Parameters
    ----------
    dataset: str
        the dataset name: euaims or hbn.
    datasetdir: str
        the path to the dataset associated data.
    outdir: str
        the destination folder.
    python_cmd: str, default 'python'
        the Python interpreter.
    """
    print_title(f"ISBI23 TRAIN: {dataset}")
    cmd = [python_cmd, "main.py", f"--dataset={dataset}",
           f"--datasetdir={datasetdir}", f"--dir_experiment={outdir}",
           "--method=joint_elbo", "--style_dim=0", "--class_dim=20",
           "--input_dim", "7", "444", "--beta=5", "--likelihood=normal",
           "--batch_size=256", "--initial_learning_rate=0.002",
           "--eval_freq=25", "--eval_freq_fid=100", "--data_multiplications=1",
           "--num_hidden_layers=1", "--end_epoch=2500", "--dropout_rate=0.",
           "--initial_out_logvar=-3", "--learn_output_scale",
           "--allow_missing_blocks"]
    print_command(cmd)
    subprocess.checkcall(cmd)


def daa(datasetdir, outdir, run, python_cmd="python"):
    """ Perform the digital avatars analysis.

    Parameters
    ----------
    datasetdir: str
        the path to the dataset associated data.
    outdir: str
        the destination folder.
    run: str
        the name of the experiment in the destination folder:
        `<dataset>_<timestamp>'.
    python_cmd: str, default 'python'
        the Python interpreter.
    """
    print_title(f"ISBI23 TRAIN: {dataset}")
    cmd = [python_cmd, "analysis.py", f"--datasetdir={datasetdir}",
           f"--dir_experiment={outdir}", f"--run={run}"]
    print_command(cmd)
    subprocess.checkcall(cmd)


def plotting(datasetdir, outdir, run, python_cmd="python"):
    """ Perform the digital avatars analysis.

    Parameters
    ----------
    datasetdir: str
        the path to the dataset associated data.
    outdir: str
        the destination folder.
    run: str
        the name of the experiment in the destination folder:
        `<dataset>_<timestamp>'.
    python_cmd: str, default 'python'
        the Python interpreter.
    """
    print_title(f"ISBI23 TRAIN: {dataset}")
    cmd = [python_cmd, "figures.py", f"--datasetdir={datasetdir}",
           f"--dir_experiment={outdir}", f"--run={run}"]
    print_command(cmd)
    subprocess.checkcall(cmd)
