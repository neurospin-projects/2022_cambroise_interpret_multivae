from curses import color_pair
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from nilearn import plotting, datasets
import plotly.express as px

def plot_latent_representations(representations, phenotypes, stratification_column, continuous,
                                channel_name, supp_labels=None, supp_labels_names=None,
                                n_dims=3, cmap_name='plasma'):
    
    assert n_dims in [2, 3]

    if continuous:
        cmap = plt.get_cmap(cmap_name)
    else:
        cmap = plt.cm.get_cmap(cmap_name, len(phenotypes[stratification_column].unique()))
    
    if supp_labels is not None:
        n_labels = len(supp_labels_names)
        unique_labels = sorted(supp_labels.unique())

        assert len(unique_labels) == n_labels
        
        if n_labels > 10:
            raise ValueError('You must provides supplementary labels with less than 10 different values,\
                otherwise you will not see anything in the representation.')

    fig = plt.figure(figsize=(400/72, 400/72 * (3/4)), dpi=72)
    point_sizes = 30
    if n_dims == 2:
        ax = fig.add_subplot(111)

        if supp_labels is None:
            scat = ax.scatter(representations[:, 0], representations[:, 1], c=phenotypes[stratification_column], cmap=cmap, s=point_sizes)
            scats = [scat]
        else:
            scats = [0] * n_labels
            markers = ['o', 'v', '^', '>', '<', 'p', '*', '+', 'd', '2']
            for i, label in enumerate(unique_labels):

                scats[i] = ax.scatter(representations[supp_labels == label, 0], representations[supp_labels == label, 1],
                                      c=phenotypes[stratification_column][supp_labels == label], cmap=cmap, marker=markers[i], s=point_sizes)

    else:
        ax = fig.add_subplot(111, projection='3d')

        if supp_labels is None:    
            scat = ax.scatter(representations[:, 0], representations[:, 1], 
                            representations[:, 2], c=phenotypes[stratification_column], cmap=cmap, s=point_sizes)
            scats = [scat]
        else:
            scats = [0] * n_labels
            markers = ['o', 'v', '^', '>', '<', 'p', '*', '+', 'd', '2']
            for i, label in enumerate(unique_labels):
                scats[i] = ax.scatter(representations[supp_labels == label, 0], representations[supp_labels == label, 1], representations[supp_labels == label, 2],
                                     c=phenotypes[stratification_column][supp_labels == label], cmap=cmap, marker=markers[i], alpha=0.5, s=point_sizes)

        xlabel = '1st dimension'
        ylabel = '2nde dimension'
        zlabel = '3rd dimension'

        ax.set_xlabel(xlabel, labelpad=-10)
        ax.set_ylabel(ylabel, labelpad=-10)
        ax.set_zlabel(zlabel, labelpad=-10)

    if continuous:
        norm = matplotlib.colors.Normalize(
            vmin=phenotypes[stratification_column].min(),
            vmax=phenotypes[stratification_column].max(),
            )
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    else:
        handles, labels = [], []
        subgroups = ['Controls','AnxDep', 'Attn', 'Emot']
        for i, label in enumerate(np.sort(phenotypes[stratification_column].unique())):
            artist = matplotlib.lines.Line2D([0], [0], marker='s', color=cmap(i), linestyle='None')
            handles.append(artist)
            if stratification_column != 'labels_modified':
                labels.append(label)
            else:
                labels.append(subgroups[label])
        legend = ax.legend(handles, labels, loc='upper right')#, loc=1, ncol=2, bbox_to_anchor=(0.5, 0.5))

    
    if supp_labels is not None:
        handles, labels = [], []
        for i, label in enumerate(unique_labels):
            artist = matplotlib.lines.Line2D([0], [0], marker=markers[i], color='k', linestyle='None')
            handles.append(artist)
            labels.append(supp_labels_names[i])
        ax.legend(handles, labels, loc='upper left')
        if not continuous:
            ax.add_artist(legend)
    plt.title('Representations of the {} channel,\nlabeled with {}'.format(channel_name, stratification_column))
    # plt.title('Projections of test subjects in the {} channel,\nlabeled with ASD subgroups'.format(channel_name))
    plt.rcParams.update({'font.size': 11})
    fig.tight_layout()
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    if n_dims == 3:
        ax.zaxis.set_ticklabels([])

    return ax

    
def nilearn_labels_to_feature_names_with_metric(labels, metric):
    features = [label.decode().replace("_and_", "&")
                for label in labels]
    lh_features = ["{}_lh_{}".format(item, metric) for item in features]
    rh_features = ["{}_rh_{}".format(item, metric) for item in features]
    return lh_features, rh_features

def nilearn_labels_to_feature_names(labels):
    features = [label.decode().replace("_and_", "&")
                for label in labels]
    lh_features = ["{}_lh".format(item) for item in features]
    rh_features = ["{}_rh".format(item) for item in features]
    return lh_features, rh_features

def plot_surf(data, metric):
    destrieux = datasets.fetch_atlas_surf_destrieux()    
    fsaverage = datasets.fetch_surf_fsaverage()
    lh_features, rh_features = nilearn_labels_to_feature_names_with_metric(
        destrieux["labels"], metric)
    print("-- features", len(lh_features), len(rh_features))
    metric_names = [name
        for name in rois_names if name.endswith(metric)]
    metric_indices = [
        idx for idx, item in enumerate(rois_names)
        if item.endswith(metric)]
    print("-- metrics rois", metric, len(metric_names))
    metric_data = data[metric_indices]
    print("-- metric data", metric, metric_data.shape)
    lh_texture = np.zeros(destrieux["map_left"].shape)
    rh_texture = np.zeros(destrieux["map_right"].shape)
    vmax = np.abs(metric_data).max() + 1e-8
    n_points_per_roi = []
    for roi_name, roi_val in zip(metric_names, metric_data):
        if "lh" in roi_name:
            roi_index = lh_features.index(roi_name)
            roi_surface_indices = (destrieux["map_left"] == roi_index)
            lh_texture[roi_surface_indices] = roi_val
            n_points_per_roi.append(roi_surface_indices.sum())
        else:
            roi_index = rh_features.index(roi_name)
            roi_surface_indices = (destrieux["map_right"] == roi_index)
            rh_texture[roi_surface_indices] = roi_val
            n_points_per_roi.append(roi_surface_indices.sum())
    fig, axs = plt.subplots(2, 2, subplot_kw={"projection": "3d"})
    mymap = plt.get_cmap("jet").copy()
    # lh_texture[lh_texture == 0] = vmax + 1
    # rh_texture[rh_texture == 0] = vmax + 1
    # mymap.set_over((0.1, 0.1, 0.1, 0.1))
    plotting.plot_surf_stat_map(
        fsaverage["infl_left"], lh_texture, cmap=mymap, threshold=1e-8,
        vmax=vmax, colorbar=True, hemi="left", view="lateral", axes=axs[0, 0],
        alpha=0.1)
    plotting.plot_surf_stat_map(
        fsaverage["infl_left"], lh_texture, cmap=mymap, threshold=1e-8,
        vmax=vmax, colorbar=True, hemi="left", view="medial", axes=axs[0, 1],
        alpha=0.1)
    plotting.plot_surf_stat_map(
        fsaverage["infl_right"], rh_texture, cmap=mymap, threshold=1e-8,
        vmax=vmax, colorbar=True, hemi="right", view="lateral", axes=axs[1, 0],
        alpha=0.1)
    plotting.plot_surf_stat_map(
        fsaverage["infl_right"], rh_texture, cmap=mymap, threshold=1e-8,
        vmax=vmax, colorbar=True, hemi="right", view="medial", axes=axs[1, 1],
        alpha=0.1)
    return fig

def plot_areas(areas, colors, color_name="Plotly", inflated=False):
    destrieux = datasets.fetch_atlas_surf_destrieux()    
    fsaverage = datasets.fetch_surf_fsaverage()
    lh_features, rh_features = nilearn_labels_to_feature_names(
        destrieux["labels"])
    print("-- features", len(lh_features), len(rh_features))
    lh_map = np.zeros(destrieux["map_left"].shape)
    rh_map = np.zeros(destrieux["map_right"].shape)
    # mymap = mcolors.LinearSegmentedColormap.from_list("plotly", px.colors.qualitative.Plotly)
    color_palette = getattr(px.colors.qualitative, color_name, None)
    if color_palette is None:
        mymap = plt.get_cmap(color_name)
        if type(mymap) is mcolors.ListedColormap:
            color_palette = mymap.colors
        else:
            color_palette = [mymap(idx / len(areas)) for idx in range(len(areas))]
    n_colors = len(color_palette)
    for color_idx, color in enumerate(color_palette):
        if type(color) is str and "rgb" in color:
            color_tuple = tuple(color.split("(")[1].split(")")[0].split(","))
            new_color_tuple = tuple(float(color) if float(color) <= 1 else float(color) / 256 for color in color_tuple)
            color_palette[color_idx] = new_color_tuple
    mymap = mcolors.ListedColormap(color_palette)
    # bounds = np.arange(len(px.colors.qualitative.Alphabet) + 1)
    # norm = mcolors.BoundaryNorm(bounds, mymap.N)
    # mymap = cm.ScalarMappable(cmap=cmap, norm=norm)
    for idx, roi_name in enumerate(areas):
        if "lh" in roi_name:
            roi_index = lh_features.index(roi_name)
            roi_surface_indices = (destrieux["map_left"] == roi_index)
            lh_map[roi_surface_indices] = colors[idx]
        else:
            roi_index = rh_features.index(roi_name)
            roi_surface_indices = (destrieux["map_right"] == roi_index)
            rh_map[roi_surface_indices] = colors[idx]
    fig, axs = plt.subplots(2, 2, subplot_kw={"projection": "3d"})
    # lh_texture[lh_texture == 0] = vmax + 1
    # rh_texture[rh_texture == 0] = vmax + 1
    # mymap.set_over((0.1, 0.1, 0.1, 0.1))*
    alpha = 1
    bg_darkness = 0.4
    template = "pial"
    if inflated:
        template = "infl"
    plotting.plot_surf_roi(fsaverage['{}_left'.format(template)], roi_map=lh_map,
                       hemi='left', view='lateral', cmap=mymap,
                       bg_map=fsaverage['sulc_left'], bg_on_data=True,
                       axes=axs[0, 0], alpha=alpha,
                       vmin=0, vmax=n_colors,
                       darkness=bg_darkness)
    plotting.plot_surf_roi(fsaverage['{}_left'.format(template)], roi_map=lh_map,
                       hemi='left', view='medial', cmap=mymap,
                       bg_map=fsaverage['sulc_left'], bg_on_data=True,
                       axes=axs[0, 1], alpha=alpha,
                       vmin=0, vmax=n_colors,
                       darkness=bg_darkness)
    plotting.plot_surf_roi(fsaverage['{}_right'.format(template)], roi_map=rh_map,
                       hemi='right', view='lateral', cmap=mymap,
                       bg_map=fsaverage['sulc_right'], bg_on_data=True,
                       axes=axs[1, 0], alpha=alpha,
                       vmin=0, vmax=n_colors,
                       darkness=bg_darkness)
    plotting.plot_surf_roi(fsaverage['{}_right'.format(template)], roi_map=rh_map,
                       hemi='right', view='medial', cmap=mymap,
                       bg_map=fsaverage['sulc_right'], bg_on_data=True,
                       axes=axs[1, 1], alpha=alpha,
                       vmin=0, vmax=n_colors,
                       darkness=bg_darkness)
    # axs[0, 0].set_title("Left hemisphere", fontdict=fontdict)
    # axs[0, 1].set_title("Right hemisphere", fontdict=fontdict)
    return fig
