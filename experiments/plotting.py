# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Plotting utility functions.
"""

# Imports
import os
import torch
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind as ttest
from utils import plot
from color_utils import print_result
import nilearn.plotting as plotting


def plot_cmat(key, cmat, ax=None, figsize=None, dpi=None, fontsize=None,
              fontweight=None, title=None):
    if figsize is None:
        figsize = (5, 2)
    if dpi is None:
        dpi = 300
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    if fontsize is None:
        fontsize = 16
    if fontweight is None:
        fontweight = "bold"
    ax.imshow(cmat, aspect="auto", cmap="Reds")
    if title is None:
        plt.title(key, fontsize=fontsize * 1.5, pad=2, fontweight=fontweight)
    else:
        plt.title(title, fontsize=fontsize * 1.5, pad=2, fontweight=fontweight)


def plot_bar(key, rsa, ax=None, figsize=None, dpi=None, fontsize=None,
             fontsize_star=None, fontweight=None, line_width=None,
             marker_size=None, title=None, report_t=False,
             do_pairwise_stars=False, do_one_sample_stars=True):
    pallete = sns.color_palette()
    pallete_new = sns.color_palette()
    if figsize is None:
        figsize = (5, 2)
    if dpi is None:
        dpi = 300
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    pallete_new[1] = pallete[0]
    pallete_new[0] = pallete[1]
    pallete_new[0] = tuple(np.array((.5, .5, .5)))
    data = rsa[key]
    n = data.shape[0]
    c = data.shape[1]
    if fontsize is None:
        fontsize = 16
    if fontsize_star is None:
        fontsize_star = 25
    if fontweight is None:
        fontweight = "bold"
    if line_width is None:
        line_width = 2.5
    if marker_size is None:
        marker_size = .1

    for i in range(c):
        plot_data = np.zeros(data.shape)
        plot_data[:, i] = data[:, i]
        xs = np.repeat(i, n) + (np.random.rand(n) - .5) * .25
        plt.scatter(xs, data[:, i], c="k", s=marker_size)
        sns.barplot(data=plot_data, errcolor="r", linewidth=line_width,
                    errwidth=line_width,
                    facecolor=np.hstack((np.array(pallete_new[i]), .3)),
                    edgecolor=np.hstack((np.array(pallete_new[i]), 1)))

    locs, labels = plt.yticks()
    new_y = locs
    new_y = np.linspace(locs[0], locs[-1], 6)
    plt.yticks(new_y, labels=[f"{yy:.2f}" for yy in new_y], fontsize=fontsize,
               fontweight=fontweight)
    plt.ylabel("model fit (r)", fontsize=fontsize, fontweight=fontweight)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(line_width)
    xlbls = ["MoPoe", "Scores\nsMCVAE", "ROIs\nsMCVAE"]
    plt.xticks(np.arange(3), labels=xlbls, fontsize=fontsize,
               fontweight=fontweight)

    if do_one_sample_stars:
        one_sample = np.array([ttest_1samp(data[:, i], 0) for i in range(3)])
        one_sample_thresh = np.array((1, .05, .001, .0001))
        one_sample_stars = np.array(("n.s.", "*", "**", "***"))
        xlbls = ["MoPoe", "Scores\nsMCVAE", "ROIs\nsMCVAE"]
        for i in range(c):
            these_stars = one_sample_stars[
                max(np.nonzero(one_sample[i, 1] < one_sample_thresh)[0])]
            xlbls[i] = f"{xlbls[i]}\n({these_stars})"
        plt.xticks(np.arange(3), labels=xlbls, fontsize=fontsize,
                   fontweight=fontweight, horizontalalignment="center",
                   multialignment="center")

    pairwise_t = np.zeros((3, 3))
    pairwise_p = np.zeros((3, 3))
    pairwise_sample_thresh = np.array((1, .05, .001, .0001))
    pairwise_sample_stars = np.array(("n.s.", "*", "**", "***"))
    if report_t:
        for i in range(c):
            for j in range(c):
                t, p = ttest(data[:, i], data[:, j])
                mnames = ["MoPoe", "Scores sMCVAE", "ROIs sMCVAE"]
                if p > .001:
                    print(f"{key} {mnames[i]} >  {mnames[j]} | "
                          f"t({data.shape[0]-1}) = {t:.2f} p = {p:.2f}")
                else:
                    print(f"{key} {mnames[i]} >  {mnames[j]} | "
                          f"t({data.shape[0]-1}) = {t:.2f} p < .001")
                pairwise_t[i, j] = t
                pairwise_p[i, j] = p
    comps = [[1, 2]]
    if do_pairwise_stars:
        for comp_idx in range(len(comps)):
            this_comp = comps[comp_idx]
            sig_idx = max(np.nonzero(
                pairwise_p[this_comp[0], this_comp[1]] <
                pairwise_sample_thresh)[0])
            max_y = new_y[-1] + comp_idx * .05
            xs = np.array(this_comp)
            stars = pairwise_sample_stars[sig_idx]
            plt.plot(xs, [max_y, max_y], "k", linewidth=line_width)
            plt.text(xs.mean(), max_y, stars, fontsize=fontsize_star,
                     horizontalalignment="center", fontweight=fontweight)

    ylim = plt.ylim()
    plt.ylim(np.array(ylim) * (1, 1.1))
    if title is None:
        plt.title(key, fontsize=fontsize * 1.5, pad=2, fontweight=fontweight)
    else:
        plt.title(title, fontsize=fontsize * 1.5, pad=2, fontweight=fontweight)


def plot_surf_mosaic(data, titles, fsaverage, filename, label=True):
    n_plots = len(data)
    size = n_plots * 10 / 4.
    fig = plt.figure(figsize=(10, size))
    subfigs = fig.subfigures(nrows=n_plots, ncols=1)
    for idx in tqdm(range(n_plots)):
        if n_plots == 1:
            subfig = subfigs
        else:
            subfig = subfigs[idx]
        subfig.suptitle(f"{titles[idx]}")
        axs = subfig.subplots(nrows=1, ncols=4,
                              subplot_kw={"projection": "3d"})
        for ax in axs:
            ax.axis("off")
        textures = data[idx]
        for hidx, hemi in enumerate(("left", "right")):
            if label:
                plotting.plot_surf_roi(
                    fsaverage[f"infl_{hemi}"], roi_map=textures[0], hemi=hemi,
                    view="lateral", bg_map=fsaverage[f"sulc_{hemi}"],
                    bg_on_data=True, darkness=.5, axes=axs[hidx * 2])
                plotting.plot_surf_roi(
                    fsaverage[f"infl_{hemi}"], roi_map=textures[1], hemi=hemi,
                    view="medial", bg_map=fsaverage[f"sulc_{hemi}"],
                    bg_on_data=True, darkness=.5, axes=axs[hidx * 2 + 1])
            else:
                plotting.plot_surf_stat_map(
                    fsaverage[f"infl_{hemi}"], stat_map=textures[0], hemi=hemi,
                    view="medial", bg_map=fsaverage[f"sulc_{hemi}"],
                    bg_on_data=True, darkness=.5, cmap="jet", colorbar=False,
                    axes=axs[hidx * 2])
                plotting.plot_surf_stat_map(
                    fsaverage[f"infl_{hemi}"], stat_map=textures[1], hemi=hemi,
                    view="lateral", bg_map=fsaverage[f"sulc_{hemi}"],
                    bg_on_data=True, darkness=.5, cmap="jet", colorbar=False,
                    axes=axs[hidx * 2 + 1])
    plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98,
                        wspace=0.02, hspace=0.02)
    plt.savefig(filename)
    print_result(f"surface mosaic: {filename}")


def plot_mosaic(images, filename, n_cols=8, image_size=(28, 28), scaler=None):
    n_images = len(images)
    if scaler is not None:
        images = scaler.inverse_transform(images.reshape(n_images, -1))
        images = images.reshape(n_images, *image_size)
    n_rows = n_images // n_cols
    if n_images % n_cols != 0:
        n_rows += 1
    arr = np.zeros((image_size[0] * n_rows, image_size[1] * n_cols))
    for idx, _arr in enumerate(images):
        j = idx % n_cols
        i = idx // n_cols
        arr[i * image_size[0]: (i + 1) * image_size[0],
            j * image_size[1]: (j + 1) * image_size[1]] = _arr
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(arr, cmap="Greys_r")
    plt.savefig(filename)
    print_result(f"mosaic: {filename}")


def generate_plots(exp, epoch):
    plots = dict();
    if exp.flags.factorized_representation:
        # mnist to mnist: swapping content and style intra modal
        swapping_figs = generate_swapping_plot(exp, epoch)
        plots['swapping'] = swapping_figs;

    for k in range(len(exp.modalities.keys())):
        cond_k = generate_conditional_fig_M(exp, epoch, k+1)
        plots['cond_gen_' + str(k+1).zfill(2)] = cond_k;

    plots['random'] = generate_random_samples_plots(exp, epoch);
    return plots;


def generate_random_samples_plots(exp, epoch):
    model = exp.mm_vae;
    mods = exp.modalities;
    num_samples = 100;
    random_samples = model.generate(num_samples)
    random_plots = dict();
    for k, m_key_in in enumerate(mods.keys()):
        mod = mods[m_key_in];
        samples_mod = random_samples[m_key_in];
        rec = torch.zeros(exp.plot_img_size,
                          dtype=torch.float32).repeat(num_samples,1,1,1);
        for l in range(0, num_samples):
            rand_plot = mod.plot_data(samples_mod[l]);
            rec[l, :, :, :] = rand_plot;
        random_plots[m_key_in] = rec;

    for k, m_key in enumerate(mods.keys()):
        fn = os.path.join(exp.flags.dir_random_samples, 'random_epoch_' +
                             str(epoch).zfill(4) + '_' + m_key + '.png');
        mod_plot = random_plots[m_key];
        p = plot.create_fig(fn, mod_plot, 10, save_figure=exp.flags.save_figure);
        random_plots[m_key] = p;
    return random_plots;


def generate_swapping_plot(exp, epoch):
    model = exp.mm_vae;
    mods = exp.modalities;
    samples = exp.test_samples;
    swap_plots = dict();
    for k, m_key_in in enumerate(mods.keys()):
        mod_in = mods[m_key_in];
        for l, m_key_out in enumerate(mods.keys()):
            mod_out = mods[m_key_out];
            rec = torch.zeros(exp.plot_img_size,
                              dtype=torch.float32).repeat(121,1,1,1);
            rec = rec.to(exp.flags.device);
            for i in range(len(samples)):
                c_sample_in = mod_in.plot_data(samples[i][mod_in.name]);
                s_sample_out = mod_out.plot_data(samples[i][mod_out.name]);
                rec[i+1, :, :, :] = c_sample_in;
                rec[(i + 1) * 11, :, :, :] = s_sample_out;
            # style transfer
            for i in range(len(samples)):
                for j in range(len(samples)):
                    i_batch_s = {mod_out.name: samples[i][mod_out.name].unsqueeze(0)}
                    i_batch_c = {mod_in.name: samples[i][mod_in.name].unsqueeze(0)}
                    l_style = model.inference(i_batch_s,
                                              num_samples=1)
                    l_content = model.inference(i_batch_c,
                                                num_samples=1)
                    l_s_mod = l_style['modalities'][mod_out.name + '_style'];
                    l_c_mod = l_content['modalities'][mod_in.name];
                    s_emb = model.reparameterize(l_s_mod[0], l_s_mod[1]);
                    c_emb = model.reparameterize(l_c_mod[0], l_c_mod[1]);
                    style_emb = {mod_out.name: s_emb}
                    emb_swap = {'content': c_emb, 'style': style_emb};
                    swap_sample = model.generate_from_latents(emb_swap);
                    swap_out = mod_out.plot_data(swap_sample[mod_out.name].squeeze(0));
                    rec[(i+1) * 11 + (j+1), :, :, :] = swap_out;
                    fn_comb = (mod_in.name + '_to_' + mod_out.name + '_epoch_'
                               + str(epoch).zfill(4) + '.png');
                    fn = os.path.join(exp.flags.dir_swapping, fn_comb);
                    swap_plot = plot.create_fig(fn, rec, 11, save_figure=exp.flags.save_figure);
                    swap_plots[mod_in.name + '_' + mod_out.name] = swap_plot;
    return swap_plots;


def generate_conditional_fig_M(exp, epoch, M):
    model = exp.mm_vae;
    mods = exp.modalities;
    samples = exp.test_samples;
    subsets = exp.subsets;

    # get style from random sampling
    random_styles = model.get_random_styles(10);

    cond_plots = dict();
    for k, s_key in enumerate(subsets.keys()):
        subset = subsets[s_key];
        num_mod_s = len(subset);

        if num_mod_s == M:
            s_in = subset;
            for l, m_key_out in enumerate(mods.keys()):
                mod_out = mods[m_key_out];
                rec = torch.zeros(exp.plot_img_size,
                                  dtype=torch.float32).repeat(100 + M*10,1,1,1);
                for m, sample in enumerate(samples):
                    for n, mod_in in enumerate(s_in):
                        c_in = mod_in.plot_data(sample[mod_in.name]);
                        rec[m + n*10, :, :, :] = c_in;
                cond_plots[s_key + '__' + mod_out.name] = rec;

            # style transfer
            for i in range(len(samples)):
                for j in range(len(samples)):
                    i_batch = dict();
                    for o, mod in enumerate(s_in):
                        i_batch[mod.name] = samples[j][mod.name].unsqueeze(0);
                    latents = model.inference(i_batch, num_samples=1)
                    c_in = latents['subsets'][s_key];
                    c_rep = model.reparameterize(mu=c_in[0], logvar=c_in[1]);

                    style = dict();
                    for l, m_key_out in enumerate(mods.keys()):
                        mod_out = mods[m_key_out];
                        if exp.flags.factorized_representation:
                            style[mod_out.name] = random_styles[mod_out.name][i].unsqueeze(0);
                        else:
                            style[mod_out.name] = None;
                    cond_mod_in = {'content': c_rep, 'style': style};
                    cond_gen_samples = model.generate_from_latents(cond_mod_in);

                    for l, m_key_out in enumerate(mods.keys()):
                        mod_out = mods[m_key_out];
                        rec = cond_plots[s_key + '__' + mod_out.name];
                        squeezed = cond_gen_samples[mod_out.name].squeeze(0);
                        p_out = mod_out.plot_data(squeezed);
                        rec[(i+M) * 10 + j, :, :, :] = p_out;
                        cond_plots[s_key + '__' + mod_out.name] = rec;

    for k, s_key_in in enumerate(subsets.keys()):
        subset = subsets[s_key_in];
        if len(subset) == M:
            s_in = subset;
            for l, m_key_out in enumerate(mods.keys()):
                mod_out = mods[m_key_out];
                rec = cond_plots[s_key_in + '__' + mod_out.name];
                fn_comb = (s_key_in + '_to_' + mod_out.name + '_epoch_' +
                           str(epoch).zfill(4) + '.png')
                fn_out = os.path.join(exp.flags.dir_cond_gen, fn_comb);
                plot_out = plot.create_fig(fn_out, rec, 10, save_figure=exp.flags.save_figure);
                cond_plots[s_key_in + '__' + mod_out.name] = plot_out;
    return cond_plots;