import os
import numpy as np
import random
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from divergence_measures.kl_div import calc_kl_divergence

from eval_metrics.sample_quality import calc_prd_score
from eval_metrics.likelihood import estimate_likelihoods

from utils import utils
from utils.TBLogger import TBLogger

from multimodal_cohort.dataset import MissingModalitySampler

# global variables
SEED = None 
if SEED is not None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED) 


def calc_log_probs(exp, result, batch):
    mods = exp.modalities
    log_probs = dict()
    weighted_log_prob = 0.0
    for m, m_key in enumerate(mods.keys()):
        mod = mods[m_key]
        if mod.name in batch[0].keys():
            target = batch[0][mod.name]
            if mod.name in exp.flags.learn_output_covmatrix:
                cov_matrix = exp.flags.cov_matrices[mod.name].copy()
                np.fill_diagonal(cov_matrix, target.detach().cpu().numpy())
                target = torch.from_numpy(cov_matrix)
            log_probs[mod.name] = -mod.calc_log_prob(result["rec"][mod.name],
                                                     target,
                                                     len(batch[0][mod.name]))
            weighted_log_prob += exp.rec_weights[mod.name]*log_probs[mod.name]
    return log_probs, weighted_log_prob


def calc_klds(exp, result):
    latents = result["latents"]["subsets"]
    klds = dict()
    for m, key in enumerate(latents.keys()):
        mu, logvar = latents[key]
        klds[key] = calc_kl_divergence(mu, logvar,
                                       norm_value=len(mu))
    return klds


def calc_klds_style(exp, result):
    latents = result["latents"]["modalities"]
    klds = dict()
    for m, key in enumerate(latents.keys()):
        if key.endswith("style") and latents[key][0] is not None:
            mu, logvar = latents[key]
            klds[key] = calc_kl_divergence(mu, logvar,
                                           norm_value=len(mu))
    return klds


def calc_style_kld(exp, klds):
    mods = exp.modalities
    style_weights = exp.style_weights
    weighted_klds = 0.0
    for m, m_key in enumerate(mods.keys()):
        if m_key + "_style" in klds.keys():
            weighted_klds += style_weights[m_key]*klds[m_key + "_style"]
    return weighted_klds



def basic_routine_epoch(exp, model_idx, batch):
    # set up weights
    beta_style = exp.flags.beta_style
    beta_content = exp.flags.beta_content
    beta = exp.flags.beta
    rec_weight = 1.0

    model = exp.models
    if exp.flags.num_models > 1:
        model = model[model_idx]
    batch_d = batch[0]
    mods = exp.modalities
    for k, m_key in enumerate(batch_d.keys()):
        batch_d[m_key] = Variable(batch_d[m_key]).to(exp.flags.device).float()
    results = model(batch_d)

    log_probs, weighted_log_prob = calc_log_probs(exp, results, batch)
    group_divergence = results["joint_divergence"]

    klds = calc_klds(exp, results)
    if exp.flags.factorized_representation:
        klds_style = calc_klds_style(exp, results)
    if (exp.flags.modality_jsd or exp.flags.modality_moe
        or exp.flags.joint_elbo):
        if exp.flags.factorized_representation:
            kld_style = calc_style_kld(exp, klds_style)
        else:
            kld_style = 0.0
        kld_content = group_divergence
        kld_weighted = beta_style * kld_style + beta_content * kld_content
        total_loss = rec_weight * weighted_log_prob + beta * kld_weighted
    elif exp.flags.modality_poe:
        klds_joint = {"content": group_divergence,
                      "style": dict()}
        elbos = dict()
        for m, m_key in enumerate(batch_d.keys()):
            mod = mods[m_key]
            if exp.flags.factorized_representation:
                kld_style_m = klds_style[m_key + "_style"]
            else:
                kld_style_m = 0.0
            klds_joint["style"][m_key] = kld_style_m
            if exp.flags.poe_unimodal_elbos:
                i_batch_mod = {m_key: batch_d[m_key]}
                r_mod = model(i_batch_mod)
                log_prob_mod = -mod.calc_log_prob(r_mod["rec"][m_key],
                                                  batch_d[m_key],
                                                  len(batch_d[m_key]))
                log_prob = {m_key: log_prob_mod}
                klds_mod = {"content": klds[m_key],
                            "style": {m_key: kld_style_m}}
                elbo_mod = utils.calc_elbo(exp, m_key, log_prob, klds_mod)
                elbos[m_key] = elbo_mod
        elbo_joint = utils.calc_elbo(exp, "joint", log_probs, klds_joint)
        elbos["joint"] = elbo_joint
        total_loss = sum(elbos.values())

    out_basic_routine = dict()
    out_basic_routine["results"] = results
    out_basic_routine["log_probs"] = log_probs
    out_basic_routine["total_loss"] = total_loss
    out_basic_routine["klds"] = klds
    return out_basic_routine


def train(model_idx, epoch, exp, tb_logger):
    model = exp.models
    dataset = exp.dataset_train
    optimizer = exp.optimizers
    grad_scaler = exp.grad_scalers
    sub_indices = None
    if exp.flags.num_models > 1:
        model = model[model_idx]
        model.train()
        exp.models[model_idx] = model
        dataset = dataset[model_idx]
        sub_indices = dataset.indices
        optimizer = optimizer[model_idx]
        grad_scaler = grad_scaler[model_idx]
    else:
        model.train()
        exp.models = model
    sampler = MissingModalitySampler(dataset, batch_size=exp.flags.batch_size,
                                     indices=sub_indices)
    d_loader = DataLoader(dataset, batch_sampler=sampler, num_workers=8)
    for iteration, batch in enumerate(d_loader):
        # with torch.autocast(exp.flags.device.type):
        basic_routine = basic_routine_epoch(exp, model_idx, batch)
        results = basic_routine["results"]
        total_loss = basic_routine["total_loss"]
        klds = basic_routine["klds"]
        log_probs = basic_routine["log_probs"]
        # backprop
        if exp.flags.grad_scaling:
            grad_scaler.scale(total_loss).backward()

            # Unscales the gradients of optimizer's assigned params in-place
            # grad_scaler.unscale_(optimizer)
            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)

            # optimizer's gradients are already unscaled, so grad_scaler.step does not unscale them,
            # although it still skips optimizer.step() if the gradients contain infs or NaNs.
            grad_scaler.step(optimizer)
            # Updates the scale for next iteration.
            grad_scaler.update()
        else:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        tb_logger.write_training_logs(results, total_loss, log_probs, klds)


def test(model_idx, epoch, exp, tb_logger):
    with torch.no_grad():
        model = exp.models
        dataset = exp.dataset_test
        if exp.flags.num_models > 1:
            model = model[model_idx]
            model.eval()
            exp.models[model_idx] = model
            dataset = dataset[model_idx]
        else:
            model.eval()
            exp.models = model

        # sampler = MissingModalitySampler(dataset, batch_size=exp.flags.batch_size)
        d_loader = DataLoader(dataset, batch_size=exp.flags.batch_size, num_workers=8)

        for _, batch in enumerate(d_loader):
            basic_routine = basic_routine_epoch(exp, model_idx, batch)
            results = basic_routine["results"]
            total_loss = basic_routine["total_loss"]
            klds = basic_routine["klds"]
            log_probs = basic_routine["log_probs"]
            tb_logger.write_testing_logs(results, total_loss, log_probs, klds)

        if (epoch + 1) % exp.flags.eval_freq == 0 or (epoch + 1) == exp.flags.end_epoch:

            if exp.flags.calc_nll:
                lhoods = estimate_likelihoods(exp)
                tb_logger.write_lhood_logs(lhoods)

            if exp.flags.calc_prd and ((epoch + 1) % exp.flags.eval_freq_fid == 0):
                prd_scores = calc_prd_score(exp)
                tb_logger.write_prd_scores(prd_scores)


def run_epochs(exp):
    # initialize summary writer
    
    str_flags = utils.save_and_log_flags(exp.flags)

    print("training epochs progress:")
    for model_idx in range(exp.flags.num_models):
        log_path = exp.flags.str_experiment
        dir_logs = exp.flags.dir_logs
        if exp.flags.num_models > 1:
            log_path = os.path.join(log_path, f"model_{model_idx}")
            dir_logs = dir_logs[model_idx]
        writer = SummaryWriter(dir_logs)
        tb_logger = TBLogger(log_path, writer)
        tb_logger.writer.add_text("FLAGS", str_flags, 0)

        exp.flags.cov_matrices = {}
        for mod in exp.modalities:
            if mod in exp.flags.learn_output_covmatrix:
                mod_path = os.path.join(
                    exp.flags.datasetdir, f"{mod}_data.npy")
                mod_idx = np.load(os.path.join(
                    exp.flags.datasetdir, "multiblock_idx_train.npz"))[mod]
                mod_data = np.load(mod_path, mmap_mode="r")[mod_idx]
                cov_matrix = np.cov(mod_data.reshape((len(mod_data, -1))))
                exp.flags.cov_matrices[mod] = cov_matrix

        for epoch in range(exp.flags.start_epoch, exp.flags.end_epoch):
            utils.printProgressBar(epoch, exp.flags.end_epoch)
            # one epoch of training and testing
            train(model_idx, epoch, exp, tb_logger)
            test(model_idx, epoch, exp, tb_logger)
            # save checkpoints after every 5 epochs
            if (epoch + 1) % 5 == 0 or (epoch + 1) == exp.flags.end_epoch:
                dir_network_epoch = os.path.join(exp.flags.dir_checkpoints,
                                                 str(epoch).zfill(4))
                model = exp.models
                if exp.flags.num_models > 1:
                    dir_network_epoch = os.path.join(exp.flags.dir_checkpoints,
                                                     f"model_{model_idx}",
                                                     str(epoch).zfill(4))
                    model = model[model_idx]
                if not os.path.exists(dir_network_epoch):
                    os.makedirs(dir_network_epoch)
                model.save_networks()
                torch.save(model.state_dict(),
                    os.path.join(dir_network_epoch, exp.flags.model_save))
