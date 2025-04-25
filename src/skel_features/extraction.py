import os
import orjson
import json
import numpy as np
from .io_utils import load_root_id
from .skel_filtering import (
    anno_mask_dict,
    pre_transform_neuron,
    output_synapses_per_compartment,
)
import pandas as pd
from scipy import sparse
from pathos.pools import ProcessPool

radius_bins = np.arange(20, 300, 30)
radius_bin_width = 10

spine_column = "tag"


def make_depth_bins(height_bounds, spacing=50):
    return np.linspace(height_bounds[0], height_bounds[1], spacing)


def make_egocentric_bins(max_d, nbins=14):
    return np.linspace(-max_d, max_d, nbins)


def make_dist_bins(max_d, nbins=5):
    return np.concatenate([np.linspace(0, max_d, nbins), [np.inf]])


def tip_len_dist(nrn, compartment=None):
    eps = nrn.end_points
    if compartment is not None:
        eps = eps[nrn.anno[anno_mask_dict[compartment]].mesh_mask[eps]]
    if len(eps) == 0:
        return None
    return nrn.distance_to_root(eps)


def tip_tort(nrn, compartment=None):
    eps = nrn.end_points
    if compartment is not None:
        eps = eps[nrn.anno[anno_mask_dict[compartment]].mesh_mask[eps]].to_skel_index
    if len(eps) == 0:
        return None
    dtr = nrn.skeleton.distance_to_root[eps]
    euc_dist = np.linalg.norm(
        (
            nrn.skeleton.vertices[eps]
            - np.atleast_2d(nrn.skeleton.vertices[nrn.skeleton.root])
        ),
        axis=1,
    )
    tort = dtr / euc_dist
    return tort


def num_syn(nrn, compartment=None, subset_query=None):
    if compartment is None:
        df = nrn.anno.post_syn.df
    else:
        df = nrn.anno.post_syn.filter_query(
            nrn.anno[anno_mask_dict[compartment]].mesh_mask
        ).df
    if subset_query is not None:
        df = df.query(subset_query)
    return len(df)


def syn_size_distribution(nrn, compartment=None, subset_query=None):
    if compartment is None:
        df = nrn.anno.post_syn.df
    else:
        df = nrn.anno.post_syn.filter_query(
            nrn.anno[anno_mask_dict[compartment]].mesh_mask
        ).df
    if subset_query is not None:
        df = df.query(subset_query)
    return df["size"].values


def syn_dist_distribution(nrn, compartment=None, subset_query=None):
    if compartment is None:
        filt = nrn.anno.post_syn
    else:
        filt = nrn.anno.post_syn.filter_query(
            nrn.anno[anno_mask_dict[compartment]].mesh_mask
        )

    minds = filt.mesh_index
    if subset_query is not None:
        subset = np.isin(
            filt.df.index,
            filt.df.query(subset_query).index,
        )
        minds = minds[subset]
    return nrn.distance_to_root(minds)


def syn_depth_dist(nrn, compartment=None, subset_query=None):
    if compartment is None:
        df = nrn.anno.post_syn.df
    else:
        df = nrn.anno.post_syn.filter_query(
            nrn.anno[anno_mask_dict[compartment]].mesh_mask
        ).df
    if subset_query is not None:
        df = df.query(subset_query)
    if len(df) > 0:
        return np.vstack(df["ctr_pt_position"])[:, 1].copy(order="C")
    else:
        return np.ascontiguousarray([])


def _is_between(xs, a, b):
    return np.logical_and(xs > a, xs <= b)


def _branches_between(nrn, d_a, d_b, min_thresh):
    gap = _is_between(nrn.skeleton.distance_to_root, d_a, d_b)
    G = nrn.skeleton.csgraph_binary_undirected
    _, ccs = sparse.csgraph.connected_components(G[:, gap][gap])
    _, nvs = np.unique(ccs, return_counts=True)
    return sum(nvs > min_thresh)


def branches_between(nrn, d_a, d_b, min_thresh=1, compartment=None):
    if compartment is None:
        return _branches_between(nrn, d_a, d_b, min_thresh)
    else:
        try:
            with nrn.mask_context(
                nrn.anno[anno_mask_dict[compartment]].mesh_mask
            ) as nmc:
                return _branches_between(nmc, d_a, d_b, min_thresh)
        except:
            return None


def branches_with_distance(nrn, radius_bins, radius_bin_width, compartment=None):
    return [
        branches_between(nrn, d_a, (d_a + radius_bin_width), compartment=compartment)
        for d_a in radius_bins
    ]


def path_length(nrn, compartment=None):
    try:
        with nrn.mask_context(nrn.anno[anno_mask_dict[compartment]].mesh_mask) as nmc:
            return nmc.path_length()
    except:
        return None


def horizontal_extent(nrn, rdist_func, compartment=None):
    try:
        with nrn.mask_context(nrn.anno[anno_mask_dict[compartment]].mesh_mask) as nmc:
            d = rdist_func(
                nrn.skeleton.root_position,
                nmc.skeleton.vertices,
                transform_points=False,
            )
            if len(d) == 0:
                return None
        return np.percentile(d, 97).copy(order="C")
    except:
        return None


def soma_depth(nrn):
    return nrn.skeleton.root_position[1]


def _node_weight(nrn):
    return np.squeeze(np.array(np.sum(nrn.skeleton.csgraph_undirected, axis=0)) / 2)


def _path_length_binned(nrn, depth_bins):
    ws = _node_weight(nrn)
    sk_vert_y = nrn.skeleton.vertices[:, 1]
    lens = []
    for d_a, d_b in zip(depth_bins[:-1], depth_bins[1:]):
        lens.append(np.sum(ws[_is_between(sk_vert_y, d_a, d_b)]))
    return np.array(lens)


def path_length_binned(nrn, depth_bins, compartment=None):
    if compartment is None:
        return _path_length_binned(nrn, depth_bins)
    else:
        try:
            with nrn.mask_context(
                nrn.anno[anno_mask_dict[compartment]].mesh_mask
            ) as nmc:
                return _path_length_binned(nmc, depth_bins)
        except:
            return None
    pass


def _syn_count_binned(nrn, depth_bins, subset_query):
    df = nrn.anno.post_syn.df
    if subset_query is not None:
        df = df.query(subset_query)
    syn_y = np.vstack(df["ctr_pt_position"])[:, 1]
    n_syn = []
    for ymin, ymax in zip(depth_bins[:-1], depth_bins[1:]):
        n_syn.append(sum(_is_between(syn_y, ymin, ymax)))
    return np.array(n_syn)


def syn_count_binned(nrn, depth_bins, compartment=None, subset_query=None):
    if compartment is None:
        return _syn_count_binned(nrn, depth_bins, subset_query=subset_query)
    else:
        try:
            with nrn.mask_context(
                nrn.anno[anno_mask_dict[compartment]].mesh_mask
            ) as nmc:
                return _syn_count_binned(nmc, depth_bins, subset_query=subset_query)
        except:
            return None


def syn_count_dist_binned(nrn, dist_bins, subset_query=None):
    syn_df = nrn.anno.post_syn.df
    if subset_query is not None:
        syn_df = syn_df.query(subset_query).reset_index(drop=True)
    syn_df["dist_to_root"] = nrn.distance_to_root(syn_df["post_pt_mesh_ind_filt"])
    syn_df["dist_bin"] = pd.cut(syn_df["dist_to_root"], dist_bins)
    return syn_df.groupby("dist_bin", observed=False).size().values


def path_length_dist_binned(nrn, dist_bins):
    pls = []
    for d_a, d_b in zip(dist_bins[:-1], dist_bins[1:]):
        pls.append(nrn.path_length(_is_between(nrn.distance_to_root(), d_a, d_b)))
    return np.array(pls)


def syn_count_egocentric(nrn, soma_depth, rel_bins, subset_query=None):
    df = nrn.anno.post_syn.df
    if subset_query is not None:
        df = df.query(subset_query)
    syn_y = np.vstack(df["ctr_pt_position"])[:, 1] - soma_depth
    n_syn = []
    for ymin, ymax in zip(rel_bins[:-1], rel_bins[1:]):
        n_syn.append(sum(_is_between(syn_y, ymin, ymax)))
    return np.array(n_syn)


def median_radius_close(nrn, rad_anno="r_eff", anno_name="segment_properties", dist=65):
    with nrn.mask_context(nrn.anno.is_dendrite.mesh_mask) as nrnf:
        seg_df = nrnf.anno[anno_name].df
        seg_df["dtr"] = nrnf.distance_to_root(nrnf.anno[anno_name].mesh_index)
    return seg_df.query("dtr < @dist and dtr > 20")[rad_anno].median()


def median_radius_distal(
    nrn, rad_anno="r_eff", anno_name="segment_properties", dist=65
):
    with nrn.mask_context(nrn.anno.is_dendrite.mesh_mask) as nrnf:
        seg_df = nrnf.anno[anno_name].df
        seg_df["dtr"] = nrnf.distance_to_root(nrnf.anno[anno_name].mesh_index)
    if len(seg_df.query("dtr > @dist")) > 0:
        return seg_df.query("dtr > @dist")[rad_anno].median()
    else:
        return 0


def area_factor(
    nrn,
    area_factor="area_factor",
    anno_name="segment_properties",
    dist=20,
):
    with nrn.mask_context(nrn.anno.is_dendrite.mesh_mask) as nrnf:
        seg_df = nrnf.anno[anno_name].df
        seg_df["dtr"] = nrnf.distance_to_root(nrnf.anno[anno_name].mesh_index)
    return seg_df.query("dtr > @dist")[area_factor].median()


def extract_features_dict(
    nrn,
    radius_bins,
    radius_bin_width,
    depth_bins,
    egocentric_bins,
    sl_dataset,
    synapse_transform="nm",
    dist_bins=5,
    max_dist=200,
):
    nrn = pre_transform_neuron(nrn, sl_dataset, synapse_transform=synapse_transform)
    dist_bins = make_dist_bins(max_dist, dist_bins)
    return {
        "root_id": nrn.seg_id,
        "soma_depth": soma_depth(nrn),
        "tip_len_dist_dendrite": tip_len_dist(nrn, "dendrite"),
        "tip_tort_dendrite": tip_tort(nrn, "dendrite"),
        "num_syn_dendrite": num_syn(nrn, "dendrite"),
        "num_spine_syn_dendrite": num_syn(nrn, "dendrite", 'tag=="spine"'),
        "num_shaft_syn_dendrite": num_syn(nrn, "dendrite", 'tag=="shaft"'),
        "num_syn_soma": num_syn(nrn, "soma"),
        "num_spine_syn_soma": num_syn(nrn, "soma", 'tag=="spine"'),
        "num_surface_syn_soma": num_syn(nrn, "soma", 'tag=="shaft" or tag=="soma"'),
        "syn_size_distribution_soma": syn_size_distribution(nrn, "soma"),
        "syn_size_distribution_dendrite": syn_size_distribution(nrn, "dendrite"),
        "syn_size_distribution_spine_dendrite": syn_size_distribution(
            nrn, "dendrite", 'tag=="spine"'
        ),
        "syn_size_distribution_shaft_dendrite": syn_size_distribution(
            nrn, "dendrite", 'tag=="shaft"'
        ),
        "syn_dist_distribution_dendrite": syn_dist_distribution(nrn, "dendrite"),
        "syn_dist_distribution_shaft_dendrite": syn_dist_distribution(
            nrn, "dendrite", subset_query='tag=="shaft"'
        ),
        "syn_dist_distribution_spine_dendrite": syn_dist_distribution(
            nrn, "dendrite", subset_query='tag=="spine"'
        ),
        "syn_depth_dist_all": syn_depth_dist(nrn, "dendrite"),
        "syn_depth_dist_spine": syn_depth_dist(
            nrn, "dendrite", subset_query='tag=="spine"'
        ),
        "syn_depth_dist_shaft": syn_depth_dist(
            nrn, "dendrite", subset_query='tag=="shaft"'
        ),
        "radial_extent_dendrite": horizontal_extent(
            nrn, sl_dataset.streamline_nm.radial_distance, "dendrite"
        ),
        "path_length_dendrite": path_length(nrn, "dendrite"),
        "branches_dist": branches_with_distance(
            nrn, radius_bins, radius_bin_width, compartment="dendrite"
        ),
        "path_length_depth_dendrite": path_length_binned(
            nrn, depth_bins, compartment="dendrite"
        ),
        "syn_count_depth_dendrite": syn_count_binned(
            nrn, depth_bins, compartment="dendrite"
        ),
        "syn_count_depth_spine_dendrite": syn_count_binned(
            nrn,
            depth_bins,
            compartment="dendrite",
            subset_query='tag=="spine"',
        ),
        "syn_count_depth_shaft_dendrite": syn_count_binned(
            nrn, depth_bins, compartment="dendrite", subset_query='tag=="shaft"'
        ),
        "syn_count_dist_binned_shaft": syn_count_dist_binned(
            nrn, dist_bins, subset_query='tag=="shaft"'
        ),
        "syn_count_dist_binned_spine": syn_count_dist_binned(
            nrn, dist_bins, subset_query='tag=="spine"'
        ),
        "dendrite_length_binned": path_length_dist_binned(nrn, dist_bins),
        "radius_dist": median_radius_distal(nrn, dist=30),
        "area_factor": area_factor(nrn),
        "egocentric_bins": syn_count_egocentric(nrn, soma_depth(nrn), egocentric_bins),
        "success": True,
    }


def extract_features(
    nrn,
    height_bounds,
    sl_dataset,
    feature_dir=None,
    filename=None,
    n_egocentric_bins=14,
    dist_bins=5,
    max_dist=200,
    synapse_transform="nm",
):
    try:
        depth_bins = make_depth_bins(height_bounds)
        egocentric_bins = make_egocentric_bins(100, n_egocentric_bins)
        features = extract_features_dict(
            nrn,
            radius_bins=radius_bins,
            radius_bin_width=radius_bin_width,
            depth_bins=depth_bins,
            egocentric_bins=egocentric_bins,
            sl_dataset=sl_dataset,
            synapse_transform=synapse_transform,
            dist_bins=dist_bins,
            max_dist=max_dist,
        )
    except:
        features = {
            "root_id": nrn.seg_id,
            "success": False,
        }
    if feature_dir:
        if filename is None:
            filename = f"{nrn.seg_id}"
        with open(f"{feature_dir}/{filename}.json", "wb") as f:
            f.write(orjson.dumps(features, option=orjson.OPT_SERIALIZE_NUMPY))
    return features


def extract_features_root_id(
    root_id,
    skel_dir,
    height_bounds,
    sl_dataset,
    feature_dir,
    dist_bins,
    max_dist,
    n_egocentric_bins=14,
    rerun=False,
    peel_threshold=0,
    synapse_transform="nm",
    save_synapse_count=False,
):
    if os.path.exists(f"{feature_dir}/{root_id}.json") and not rerun:
        with open(f"{feature_dir}/{root_id}.json") as f:
            dat = json.load(f)
            if dat["success"] is True:
                return True
    try:
        nrn = load_root_id(root_id, skel_dir, peel_threshold=peel_threshold)
        if save_synapse_count:
            output_by_compartment = output_synapses_per_compartment(nrn)
            with open(f"{feature_dir}/{root_id}_syn_count.json", "w") as f:
                json.dump(output_by_compartment, f)
        features = extract_features(
            nrn,
            height_bounds,
            sl_dataset,
            feature_dir=feature_dir,
            n_egocentric_bins=n_egocentric_bins,
            synapse_transform=synapse_transform,
            dist_bins=dist_bins,
            max_dist=max_dist,
        )
        return features["success"]
    except Exception as e:
        print(e)
        return False


def extract_features_mp(
    root_ids,
    skel_dir,
    height_bounds,
    sl_dataset,
    feature_dir,
    dist_bins=5,
    max_dist=200,
    n_egocentric_bins=14,
    peel_threshold=0,
    synapse_transform="nm",
    rerun=False,
    save_synapse_count=False,
    nodes=8,
):
    pool = ProcessPool(nodes=nodes)
    return np.array(
        pool.map(
            extract_features_root_id,
            root_ids,
            [skel_dir] * len(root_ids),
            [height_bounds] * len(root_ids),
            [sl_dataset] * len(root_ids),
            [feature_dir] * len(root_ids),
            [dist_bins] * len(root_ids),
            [max_dist] * len(root_ids),
            [n_egocentric_bins] * len(root_ids),
            [rerun] * len(root_ids),
            [peel_threshold] * len(root_ids),
            [synapse_transform] * len(root_ids),
            [save_synapse_count] * len(root_ids),
        )
    )
