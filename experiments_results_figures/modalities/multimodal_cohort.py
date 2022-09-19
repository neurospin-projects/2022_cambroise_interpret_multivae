
import torch
import numpy as np

from modalities.modality import Modality


class Clinical(Modality):
    def __init__(self, n_scores, enc, dec, class_dim, style_dim, lhood_name):
        super().__init__("clinical", enc, dec, class_dim, style_dim, lhood_name)
        self.data_size = torch.Size([n_scores])
        self.gen_quality_eval = True
        self.file_suffix = ".npy"


    def save_data(self, d, fn, args):
        np.save(d.tolist(), fn)



    def plot_data(self, d):
        p = d.repeat(1, 3, 1, 1)
        return p;

class Rois(Modality):
    def __init__(self, n_rois, enc, dec, class_dim, style_dim, lhood_name):
        super().__init__("rois", enc, dec, class_dim, style_dim, lhood_name)
        self.data_size = torch.Size([n_rois])
        self.gen_quality_eval = True
        self.file_suffix = '.npy'


    def save_data(self, d, fn, args):
        np.save(d.tolist(), fn)



    def plot_data(self, d):
        p = d.repeat(1, 3, 1, 1)
        return p