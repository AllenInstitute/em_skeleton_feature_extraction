import pandas as pd
import numpy as np
from standard_transform import v1dd_ds
from meshparty import meshwork
import skeleton_plot as skplt
import matplotlib.pyplot as plt
import seaborn as sns

# Set up directories and layers

skel_dir = "../skeletonization/skeletons/v1dd"

layer_bnds = [100, 270, 400, 550, 750]

plot_dir = "plots/v1dd"

import os

try:
    os.makedirs(plot_dir)
except:
    pass

root_ids = [864691132602579885, 864691132609091256, 864691132777470624]


# Do plot

root_id = root_ids[2]
fn = f"{plot_dir}/example_{root_id}.pdf"

nrn = meshwork.load_meshwork(f"{skel_dir}/{root_id}.h5")

# Convert the vertices and annotations to a rotated and streamline-straightened version.
# The [1,1,1] in the annotatoin is because the annotations came in in that resolution.


nrn_tr = v1dd_ds.streamline_nm.transform_meshwork_vertices(nrn)
nrn_tr = v1dd_ds.streamline_res([1, 1, 1]).transform_meshwork_annotations(
    nrn_tr,
    {
        "pre_syn": ["pre_pt_position", "ctr_pt_position", "post_pt_position"],
        "post_syn": ["pre_pt_position", "ctr_pt_position", "post_pt_position"],
    },
    root_loc=nrn.skeleton.root_position / [1, 1, 1],
)

# Set up color maps and bounds for the panels

hue = np.invert(nrn.anno.is_axon.skel_mask).astype(int)

cmap = {
    0: "tomato",
    1: "navy",
}

height_bounds = [-25, 850]
spacing = 0

bbox = np.array(
    [nrn_tr.skeleton.vertices.min(axis=0), nrn_tr.skeleton.vertices.max(axis=0)]
)

dims = np.diff(bbox, axis=0).squeeze()
baseline_height = int(np.diff(height_bounds))
height_ratios = (dims[2], baseline_height)

zdim = dims[2] + 2 * spacing

net_height = zdim + baseline_height

y_inches = 4
fig_height = net_height * (y_inches / baseline_height)
fig_width = dims[0] * (y_inches / baseline_height)

# Build figure with proper scaling

fig, axes = plt.subplots(
    figsize=(fig_width, fig_height),
    nrows=2,
    height_ratios=height_ratios,
    gridspec_kw={"hspace": 0, "wspace": 0},
    sharex=True,
    dpi=300,
)

# Plot skeleton and synapses of top-down view (x,z)

ax = axes[0]

# Plot skeleton
skplt.plot_tools.plot_mw_skel(
    ax,
    nrn_tr,
    plot_soma=True,
    soma_size=10,
    pull_radius=True,
    skel_colors=hue,
    skel_color_map=cmap,
    line_width=1.5,
    x="x",
    y="z",
)
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

## Plot pre and postsynaptic sites
sns.scatterplot(
    x=nrn_tr.anno.pre_syn.df["ctr_pt_position"].apply(lambda x: x[0]),
    y=nrn_tr.anno.pre_syn.df["ctr_pt_position"].apply(lambda x: x[2]),
    s=1,
    color="maroon",
    alpha=0.5,
    ax=ax,
)

sns.scatterplot(
    x=nrn_tr.anno.post_syn.df["ctr_pt_position"].apply(lambda x: x[0]),
    y=nrn_tr.anno.post_syn.df["ctr_pt_position"].apply(lambda x: x[2]),
    s=1,
    color="midnightblue",
    alpha=0.2,
)

# Plot side view

ax = axes[1]

## Plot skeleton
skplt.plot_tools.plot_mw_skel(
    ax,
    nrn_tr,
    plot_soma=True,
    soma_size=10,
    pull_radius=True,
    skel_colors=hue,
    skel_color_map=cmap,
    radius_scaling=1.5,
    x="x",
    y="y",
)

## Plot upper and lower data bounds plus layer heights

ax.set_ylim(*height_bounds)
ax.invert_yaxis()
ax.hlines(
    layer_bnds,
    ax.get_xlim()[0],
    ax.get_xlim()[1],
    color=[0.5, 0.5, 0.5],
    linewidth=0.5,
    alpha=0.5,
    linestyle=":",
)
ax.hlines(
    height_bounds,
    ax.get_xlim()[0],
    ax.get_xlim()[1],
    color=[0, 0, 0],
    linewidth=1,
    alpha=0.5,
    linestyle="-",
)

## Plot synapses

sns.scatterplot(
    x=nrn_tr.anno.post_syn.df["ctr_pt_position"].apply(lambda x: x[0]),
    y=nrn_tr.anno.post_syn.df["ctr_pt_position"].apply(lambda x: x[1]),
    s=1,
    color="midnightblue",
    alpha=0.2,
)

sns.scatterplot(
    x=nrn_tr.anno.pre_syn.df["ctr_pt_position"].apply(lambda x: x[0]),
    y=nrn_tr.anno.pre_syn.df["ctr_pt_position"].apply(lambda x: x[1]),
    s=1,
    color="maroon",
    alpha=0.5,
)

ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

axes[0].set_xlim(bbox[0, 0], bbox[1, 0])
axes[1].set_xlim(bbox[0, 0], bbox[1, 0])
axes[0].set_ylim(bbox[0, 2] - spacing, bbox[1, 2] + spacing)

plt.tight_layout()
fig.savefig(fn)


# If running a lot of these, important to close the figures

# plt.close(fig)
