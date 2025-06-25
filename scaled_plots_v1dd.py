# %%
import pandas as pd
import numpy as np
from standard_transform import v1dd_ds
from meshparty import meshwork
import skeleton_plot as skplt
import matplotlib.pyplot as plt
import seaborn as sns

skel_dir = '../skeletonization/skeletons/v1dd'
feat_df = pd.read_feather('excitatory_feature_d150.feather')

layer_bnds = [100, 270, 400, 550, 750]

# %%
plot_dir = 'plots/v1dd'

import os
try:
    os.makedirs(plot_dir)
except:
    pass

# %%
from caveclient import CAVEclient

ss_df = client.materialize.query_view('single_somas')

client = CAVEclient('v1dd')

# %%
root_id = feat_df.query('proofread').sort_values(by=['clust', 'soma_depth']).iloc[
    2
].root_id



# %%
for _, row in tqdm.tqdm(feat_df.query('proofread').sort_values(by=['clust', 'soma_depth']).iterrows()):
    root_id = row['root_id']
    clust = row['clust']
    sd = int(row['soma_depth'])
    fn = f"{plot_dir}/syn_clust_{clust}_sd_{sd}_oid_{root_id}.png"
    
    nrn = meshwork.load_meshwork(f"{skel_dir}/{root_id}.h5")
    
    syn_df = nrn.anno.pre_syn.df.copy()
    syn_df['dtr'] = nrn.distance_to_root(nrn.anno.pre_syn.mesh_index) / 1_000

    syn_df['pt_um'] = v1dd_ds.transform_res([9.7, 9.7, 45]).apply(syn_df['ctr_pt_position'])

    syn_df['syn_depth'] = v1dd_ds.transform_res([9.7, 9.7, 45]).apply_project('y', syn_df['ctr_pt_position'])

    syn_df['post_pt_root_id'] = client.chunkedgraph.get_roots(
        syn_df['post_pt_supervoxel_id'],
        timestamp=client.materialize.get_timestamp(),
    )

    ss_targ = syn_df.merge(
        ss_df[['pt_root_id', 'pt_position']],
        left_on='post_pt_root_id',
        right_on='pt_root_id',
    )

    ss_targ['targ_loc'] = v1dd_ds.transform_nm.apply(ss_targ['pt_position'])

    ss_targ['targ_depth'] = v1dd_ds.transform_nm.apply_project('y', ss_targ['pt_position'])

    source_loc = v1dd_ds.transform_nm.apply(nrn.skeleton.root_position)

    ss_targ['source_loc'] = [source_loc.tolist()] * len(ss_targ)

    ss_targ['dist_to_targ_soma'] = np.linalg.norm(np.vstack(ss_targ['targ_loc'].values)-np.vstack(ss_targ['pt_um'].values), axis=1)
    ss_targ['vdist_to_targ_soma'] = np.vstack(ss_targ['targ_loc'].values)[:,1] - np.vstack(ss_targ['pt_um'].values)[:,1]

    fig, ax = plt.subplots(figsize=(5,3), dpi=300)
    sns.scatterplot(
        x='targ_depth',
        y='syn_depth',
        data=ss_targ,
        size='size',
        size_norm=(100,10_000),
        sizes=(0, 20),
        hue='dtr',
        hue_norm=(0,1000),
        palette='viridis',
        ax=ax,
        legend=False,
        edgecolor='k',
    )

    ax.hlines(
        layer_bnds,
        height_bounds[0],
        height_bounds[1],
        color=[0.5, 0.5, 0.5],
        linewidth=0.5,
        alpha=0.5,
        linestyle=":",
    )
    ax.vlines(
        layer_bnds,
        height_bounds[0],
        height_bounds[1],
        color=[0.5, 0.5, 0.5],
        linewidth=0.5,
        alpha=0.5,
        linestyle=":",
    )
    sns.despine(ax=ax, trim=True)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig(fn)
    plt.close(fig)

import tqdm

# %%
root_ids = [864691132602579885, 864691132609091256, 864691132777470624]

# %%
root_id = root_ids[2]
fn = f"{plot_dir}/example_{root_id}.pdf"

nrn = meshwork.load_meshwork(f"{skel_dir}/{root_id}.h5")

nrn_tr = v1dd_ds.streamline_nm.transform_meshwork_vertices(nrn)
nrn_tr = v1dd_ds.streamline_res([9.7, 9.7, 45]).transform_meshwork_annotations(
    nrn_tr,
    {
        "pre_syn": ["pre_pt_position", "ctr_pt_position", "post_pt_position"],
        "post_syn": ["pre_pt_position", "ctr_pt_position", "post_pt_position"],
    },
    root_loc=nrn.skeleton.root_position / [9.7, 9.7, 45],
)

hue = np.invert(nrn.anno.is_axon.skel_mask).astype(int)
cmap = {
    0: 'tomato',
    1: 'navy',
}

height_bounds = [-25, 850]
spacing = 0

bbox = np.array(
    [nrn_tr.skeleton.vertices.min(axis=0), nrn_tr.skeleton.vertices.max(axis=0)]
)

dims = np.diff(bbox, axis=0).squeeze()
baseline_height = int(np.diff(height_bounds))
height_ratios = (dims[2], baseline_height)

zdim = dims[2] + 2*spacing

net_height = zdim+baseline_height

y_inches = 4
fig_height = net_height * (y_inches / baseline_height)
fig_width = dims[0] * (y_inches / baseline_height)

fig, axes = plt.subplots(
    figsize=(fig_width, fig_height),
    nrows=2,
    height_ratios=height_ratios,
    gridspec_kw={"hspace": 0, "wspace": 0},
    sharex=True,
    dpi=300,
)

ax = axes[0]
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

# sns.scatterplot(
#     x=nrn_tr.anno.post_syn.df["ctr_pt_position"].apply(lambda x: x[0]),
#     y=nrn_tr.anno.post_syn.df["ctr_pt_position"].apply(lambda x: x[2]),
#     s=1,
#     color="midnightblue",
#     alpha=0.2,
#     ax=ax,
# )

sns.scatterplot(
    x=nrn_tr.anno.pre_syn.df["ctr_pt_position"].apply(lambda x: x[0]),
    y=nrn_tr.anno.pre_syn.df["ctr_pt_position"].apply(lambda x: x[2]),
    s=1,
    color="maroon",
    alpha=0.5,
    ax=ax,
)

ax = axes[1]
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

# sns.scatterplot(
#     x=nrn_tr.anno.post_syn.df["ctr_pt_position"].apply(lambda x: x[0]),
#     y=nrn_tr.anno.post_syn.df["ctr_pt_position"].apply(lambda x: x[1]),
#     s=1,
#     color="midnightblue",
#     alpha=0.2,
# )

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
# plt.close(fig)


# %%
# root_id = feat_df.query('proofread and clust==11').sample(1).root_id.values[0]
for _, row in tqdm.tqdm(feat_df.query('proofread').sort_values(by=['clust', 'soma_depth']).iterrows()):
    root_id = row['root_id']
    clust = row['clust']
    sd = int(row['soma_depth'])
    fn = f"{plot_dir}/clust_{clust}_sd_{sd}_oid_{root_id}.png"
    
    try:
        nrn = meshwork.load_meshwork(f"{skel_dir}/{root_id}.h5")

        nrn_tr = v1dd_ds.streamline_nm.transform_meshwork_vertices(nrn)
        nrn_tr = v1dd_ds.streamline_res([9.7, 9.7, 45]).transform_meshwork_annotations(
            nrn_tr,
            {
                "pre_syn": ["pre_pt_position", "ctr_pt_position", "post_pt_position"],
                "post_syn": ["pre_pt_position", "ctr_pt_position", "post_pt_position"],
            },
            root_loc=nrn.skeleton.root_position / [9.7, 9.7, 45],
        )

        hue = np.invert(nrn.anno.is_axon.skel_mask).astype(int)
        cmap = {
            0: 'tomato',
            1: 'navy',
        }

        height_bounds = [-25, 850]
        spacing = 0

        bbox = np.array(
            [nrn_tr.skeleton.vertices.min(axis=0), nrn_tr.skeleton.vertices.max(axis=0)]
        )

        dims = np.diff(bbox, axis=0).squeeze()
        baseline_height = int(np.diff(height_bounds))
        height_ratios = (dims[2], baseline_height)

        zdim = dims[2] + 2*spacing

        net_height = zdim+baseline_height

        y_inches = 4
        fig_height = net_height * (y_inches / baseline_height)
        fig_width = dims[0] * (y_inches / baseline_height)

        fig, axes = plt.subplots(
            figsize=(fig_width, fig_height),
            nrows=2,
            height_ratios=height_ratios,
            gridspec_kw={"hspace": 0, "wspace": 0},
            sharex=True,
            dpi=300,
        )

        ax = axes[0]
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
            y="z",
        )
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        sns.scatterplot(
            x=nrn_tr.anno.post_syn.df["ctr_pt_position"].apply(lambda x: x[0]),
            y=nrn_tr.anno.post_syn.df["ctr_pt_position"].apply(lambda x: x[2]),
            s=1,
            color="midnightblue",
            alpha=0.2,
            ax=ax,
        )

        sns.scatterplot(
            x=nrn_tr.anno.pre_syn.df["ctr_pt_position"].apply(lambda x: x[0]),
            y=nrn_tr.anno.pre_syn.df["ctr_pt_position"].apply(lambda x: x[2]),
            s=1,
            color="maroon",
            alpha=0.5,
            ax=ax,
        )

        ax = axes[1]
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
        plt.close(fig)
    except Exception as e:
        print(e)

# %%
from caveclient import CAVEclient
client = CAVEclient('v1dd')

# %%
soma_df = client.materialize.tables.nucleus_detection_v0().query(desired_resolution=[1,1,1], split_positions=True)

# %%
soma_df = soma_df.drop_duplicates(subset='pt_root_id', keep=False)

# %%
soma_df['soma_depth'] = v1dd_ds.transform_nm.apply_dataframe('pt_position', df=soma_df, projection='y')

# %%
sns.stripplot?

# %%
targ_dfs = []
for ind in range(3):
    root_id = root_ids[ind]
    
    nrn = meshwork.load_meshwork(f"{skel_dir}/{root_id}.h5")
    
    nrn_tr = v1dd_ds.streamline_nm.transform_meshwork_vertices(nrn)
    nrn_tr = v1dd_ds.streamline_res([9.7, 9.7, 45]).transform_meshwork_annotations(
        nrn_tr,
        {
            "pre_syn": ["pre_pt_position", "ctr_pt_position", "post_pt_position"],
            "post_syn": ["pre_pt_position", "ctr_pt_position", "post_pt_position"],
        },
        root_loc=nrn.skeleton.root_position / [9.7, 9.7, 45],
    )
    
    targ_df = nrn.anno.pre_syn.df
    
    targ_df = targ_df.merge(
        soma_df[['pt_root_id', 'soma_depth']],
        left_on='post_pt_root_id',
        right_on='pt_root_id',
    )
    
    targ_df['index_val'] = ind
    targ_dfs.append(targ_df)

# %%
targ_df_all = pd.concat(targ_dfs)

# %%
sns.stripplot

# %%
fig, ax = plt.subplots(figsize=(1,5), dpi=300)

sns.stripplot(
    x='index_val',
    y='soma_depth',
    data=targ_df_all,
    jitter=0.35,
    size=2,
    color='k',
    ax=ax,
    alpha=0.15,
)

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
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
sns.despine(ax=ax, trim=True, left=True, bottom=True)

fig.savefig(f'{plot_dir}/target_distribution.pdf', bbox_inches='tight')

# %%



