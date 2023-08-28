import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """
    def __init__(self, flags, mod_num):
        super(Encoder, self).__init__()

        self.flags = flags
        self.shared_encoder = nn.Sequential()
        input_dim = flags.input_dim[mod_num]
        output_dim = 256
        for _ in range(flags.num_hidden_layer_encoder):
            self.shared_encoder.append(nn.Linear(input_dim, output_dim))
            self.shared_encoder.append(nn.ReLU())
            self.shared_encoder.append(nn.Dropout(flags.dropout_rate))
            input_dim = output_dim
        self.style_dim = flags.style_dim[mod_num]
        # content branch
        self.class_mu = nn.Linear(input_dim, flags.class_dim)
        self.class_logvar = nn.Linear(input_dim, flags.class_dim)
        # optional style branch
        if flags.factorized_representation and self.style_dim > 0:
            self.style_mu = nn.Linear(input_dim, self.style_dim)
            self.style_logvar = nn.Linear(input_dim, self.style_dim)

    def forward(self, h):
        h = self.shared_encoder(h)
        if self.flags.factorized_representation and self.style_dim > 0:
            return self.style_mu(h), self.style_logvar(h), self.class_mu(h), \
                   self.class_logvar(h)
        else:
            return None, None, self.class_mu(h), self.class_logvar(h)


class Decoder(nn.Module):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """
    def __init__(self, flags, mod_num, mod_name):
        super(Decoder, self).__init__()
        self.flags = flags
        self.shared_decoder = nn.Sequential()
        self.style_dim = flags.style_dim[mod_num]
        input_dim = self.style_dim + flags.class_dim
        output_dim = 256
        for _ in range(flags.num_hidden_layer_decoder):
            self.shared_decoder.append(nn.Linear(input_dim, output_dim))
            self.shared_decoder.append(nn.ReLU())
            self.shared_decoder.append(nn.Dropout(flags.dropout_rate))
            input_dim = output_dim
        self.out_mu =  nn.Linear(input_dim, flags.input_dim[mod_num])

        if flags.learn_output_sample_scale:
            self.logvar = nn.Linear(input_dim, flags.input_dim[mod_num])
        else:
            logvar_dim = [1, flags.input_dim[mod_num]]
            if mod_name in flags.learn_output_covmatrix:
                logvar_dim.append(flags.input_dim[mod_num])
            self.logvar = nn.Parameter(
                data=torch.FloatTensor(
                    *logvar_dim).fill_(flags.initial_out_logvar),
                requires_grad=(flags.learn_output_scale or
                               mod_name in flags.learn_output_covmatrix))

    def forward(self, style_latent_space, class_latent_space):
        if self.flags.factorized_representation and self.style_dim > 0:
            z = torch.cat((style_latent_space, class_latent_space), dim=1)
        else:
            z = class_latent_space
        h = self.shared_decoder(z)
        x_hat = self.out_mu(h)
        if self.flags.learn_output_sample_scale:
            logvar = self.logvar(h)
        else:
            logvar = self.logvar
        if self.logvar.ndim == 3:
            logvar = torch.matmul(logvar, logvar.transpose(1, 2))
        return x_hat, (logvar * 0.5).exp().to(z.device)


