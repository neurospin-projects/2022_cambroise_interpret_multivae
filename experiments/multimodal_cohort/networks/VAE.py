import torch.nn as nn

from utils.BaseMMVae import BaseMMVae


class VAE(BaseMMVae, nn.Module):
    def __init__(self, flags, modalities, subsets):
        super().__init__(flags, modalities, subsets);
