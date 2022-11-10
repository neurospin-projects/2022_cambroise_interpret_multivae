

import argparse

parser = argparse.ArgumentParser()

# TRAINING
parser.add_argument('--batch_size', type=int, default=256, help="batch size for training")
parser.add_argument('--initial_learning_rate', type=float, default=0.001, help="starting learning rate")
parser.add_argument('--beta_1', type=float, default=0.9, help="default beta_1 val for adam")
parser.add_argument('--beta_2', type=float, default=0.999, help="default beta_2 val for adam")
parser.add_argument('--start_epoch', type=int, default=0, help="flag to set the starting epoch for training")
parser.add_argument('--end_epoch', type=int, default=100, help="flag to indicate the final epoch of training")

# DATA DEPENDENT
parser.add_argument('--class_dim', type=int, default=20, help="dimension of common factor latent space")

# SAVE and LOAD
parser.add_argument('--mm_vae_save', type=str, default='mm_vae', help="model save for vae_bimodal")
parser.add_argument('--load_saved', type=bool, default=False, help="flag to indicate if a saved model will be loaded")

# DIRECTORIES
# data
parser.add_argument('--dir_data', type=str, default='../data', help="directory where data is stored")
# experiments
parser.add_argument('--dir_experiment', type=str, default='/tmp/multilevel_multimodal_vae_swapping', help="directory to save generated samples in")
# fid
parser.add_argument('--dir_fid', type=str, default=None, help="directory to save generated samples for fid score calculation")
#fid_score
parser.add_argument('--inception_state_dict', type=str, default='../inception_state_dict.pth', help="path to inception v3 state dict")

# EVALUATION
parser.add_argument('--calc_nll', default=False, action="store_true",
                    help="flag to indicate calculation of nll")
parser.add_argument('--calc_prd', default=False, action="store_true",
                    help="flag to indicate calculation of prec-rec for gen model")
parser.add_argument('--save_figure', default=False, action="store_true",
                    help="flag to indicate if figures should be saved to disk (in addition to tensorboard logs)")
parser.add_argument('--eval_freq', type=int, default=10, help="frequency of evaluation of latent representation of generative performance (in number of epochs)")
parser.add_argument('--eval_freq_fid', type=int, default=10, help="frequency of evaluation of latent representation of generative performance (in number of epochs)")
parser.add_argument('--num_samples_fid', type=int, default=10000,
                    help="number of samples the calculation of fid is based on")
parser.add_argument('--num_training_samples_lr', type=int, default=500,
                    help="number of training samples to train the lr clf")

#multimodal
parser.add_argument('--method', type=str, default='poe', help='choose method for training the model')
parser.add_argument('--modality_jsd', type=bool, default=False, help="modality_jsd")
parser.add_argument('--modality_poe', type=bool, default=False, help="modality_poe")
parser.add_argument('--modality_moe', type=bool, default=False, help="modality_moe")
parser.add_argument('--joint_elbo', type=bool, default=False, help="modality_moe")
parser.add_argument('--poe_unimodal_elbos', type=bool, default=True, help="unimodal_klds")
parser.add_argument('--factorized_representation', action='store_true', default=False, help="factorized_representation")

# LOSS TERM WEIGHTS
parser.add_argument('--beta', type=float, default=5.0, help="default weight of sum of weighted divergence terms")
parser.add_argument('--beta_style', type=float, default=1.0, help="default weight of sum of weighted style divergence terms")
parser.add_argument('--beta_content', type=float, default=1.0, help="default weight of sum of weighted content divergence terms")




