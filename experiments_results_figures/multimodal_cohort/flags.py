from utils.BaseFlags import parser as parser

parser.add_argument('--dataset', type=str, default='euaims', help="name of the dataset")

parser.add_argument('--style_dim', type=int, default=0, help="style dimensionality")
parser.add_argument('--dropout_rate', "-dr", type=float, default=0, help="dropout rate")
parser.add_argument('--num_classes', type=int, default=2, help="number of classes on which the data set trained")
parser.add_argument('--len_sequence', type=int, default=8, help="length of sequence")
parser.add_argument('--img_size_m1', type=int, default=28, help="img dimension (width/height)")
parser.add_argument('--num_channels_m1', type=int, default=1, help="number of channels in images")
parser.add_argument('--img_size_m2', type=int, default=32, help="img dimension (width/height)")
parser.add_argument('--num_channels_m2', type=int, default=3, help="number of channels in images")
parser.add_argument('--dim', type=int, default=64, help="number of classes on which the data set trained")
parser.add_argument('--data_multiplications', type=int, default=1, help="number of pairs per sample")
parser.add_argument('--num_hidden_layers', type=int, default=1, help="number of channels in images")
parser.add_argument('--likelihood', type=str, default='normal', help="output distribution")

# data
parser.add_argument('--datasetdir', type=str, help="directory where data is stored")

# multimodal
parser.add_argument('--subsampled_reconstruction', default=True, help="subsample reconstruction path")
parser.add_argument('--include_prior_expert', action='store_true', default=False, help="factorized_representation")

# weighting of loss terms
parser.add_argument('--div_weight', type=float, default=None, help="default weight divergence per modality, if None use 1/(num_mods+1).")
parser.add_argument('--div_weight_uniform_content', type=float, default=None, help="default weight divergence term prior, if None use (1/num_mods+1)")

# annealing
parser.add_argument('--kl_annealing', type=int, default=0, help="number of kl annealing steps; 0 if no annealing should be done")

parser.add_argument('--input_dim', nargs="+", type=int, default=[7, 444, 24], help="input dimension for each modality")
parser.add_argument('--initial_out_logvar', type=float, default=0, help="initial output logvar")
parser.add_argument('--learn_output_scale', action='store_true', default=False, help="allows for different scales per feature")
parser.add_argument('--allow_missing_blocks', action='store_true', default=False, help="allows for missing modalities")




