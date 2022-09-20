import sys
import os
import json

import torch
import pickle

from run_epochs import run_epochs

from utils.filehandling import create_dir_structure
from multimodal_cohort.flags import parser
from multimodal_cohort.experiment import MultimodalExperiment

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    FLAGS.device = torch.device('cuda' if use_cuda else 'cpu')
    if FLAGS.method == 'poe':
        FLAGS.modality_poe = True
        FLAGS.poe_unimodal_elbos = True
    elif FLAGS.method == 'moe':
        FLAGS.modality_moe = True
    elif FLAGS.method == 'jsd':
        FLAGS.modality_jsd = True
    elif FLAGS.method == 'joint_elbo':
        FLAGS.joint_elbo = True
    else:
        print('method implemented...exit!')
        sys.exit()

    # postprocess flags
    FLAGS.num_mods = len(FLAGS.input_dim)  # set number of modalities dynamically
    if FLAGS.div_weight_uniform_content is None:
        FLAGS.div_weight_uniform_content = 1 / (FLAGS.num_mods + 1)
    FLAGS.alpha_modalities = [FLAGS.div_weight_uniform_content]
    if FLAGS.div_weight is None:
        FLAGS.div_weight = 1 / (FLAGS.num_mods + 1)
    FLAGS.alpha_modalities.extend([FLAGS.div_weight for _ in range(FLAGS.num_mods)])
    create_dir_structure(FLAGS)

    alphabet_path = os.path.join(os.getcwd(), 'alphabet.json')
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    mst = MultimodalExperiment(FLAGS, alphabet)
    mst.set_optimizer()

    run_epochs(mst)

    with open(os.path.join(FLAGS.dir_experiment_run, "experiment.pickle"), "wb") as f:
        pickle.dump(mst, f, pickle.HIGHEST_PROTOCOL)