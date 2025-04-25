import numpy as np
from sklearn import decomposition, preprocessing
from scipy import stats


def percentile_above_zero(X, percentile=50):
    out = []
    for x in X:
        try:
            out.append(np.percentile(x[x > 0], percentile))
        except:
            out.append(0)
    return np.array(out)


# bring spine features into this collection
def assemble_features_from_data(
    df,
    n_syn_comp=5,
    n_branch_comp=3,
    n_syn_ego=5,
    model_dict=None,
    use_spines=False,
):
    df = df.copy()
    df["tip_len_dist_dendrite_p50"] = df["tip_len_dist_dendrite"].apply(
        lambda x: np.percentile(x, 75)
    )

    df["tip_tort_dendrite_p50"] = df["tip_tort_dendrite"].apply(
        lambda x: np.percentile(x, 75)
    )

    df["syn_size_distribution_soma_p50"] = df["syn_size_distribution_soma"].apply(
        np.median
    )
    df["syn_dist_distribution_dendrite_p50"] = df[
        "syn_dist_distribution_dendrite"
    ].apply(np.median)

    if use_spines:
        df["syn_dist_distribution_dendrite_spine_p50"] = df[
            "syn_dist_distribution_spine_dendrite"
        ].apply(np.median)
        df["syn_dist_distribution_dendrite_shaft_p50"] = df[
            "syn_dist_distribution_shaft_dendrite"
        ].apply(np.median)
        df["dend_spine_shaft_offset"] = (
            df["syn_dist_distribution_dendrite_spine_p50"]
            - df["syn_dist_distribution_dendrite_shaft_p50"]
        )

    df["syn_size_distribution_dendrite_p50"] = df[
        "syn_size_distribution_dendrite"
    ].apply(np.median)
    df["syn_size_distribution_dendrite_p10"] = df[
        "syn_size_distribution_dendrite"
    ].apply(lambda x: np.percentile(x, 10))
    df["syn_size_distribution_dendrite_p90"] = df[
        "syn_size_distribution_dendrite"
    ].apply(lambda x: np.percentile(x, 90))
    df["syn_size_dendrite_cv"] = df["syn_size_distribution_dendrite"].apply(
        np.std
    ) / df["syn_size_distribution_dendrite"].apply(np.mean)

    if use_spines:
        df["syn_size_distribution_spine_dendrite_p50"] = df[
            "syn_size_distribution_spine_dendrite"
        ].apply(np.median)
        df["syn_size_distribution_spine_dendrite_p10"] = df[
            "syn_size_distribution_spine_dendrite"
        ].apply(lambda x: np.percentile(x, 10))
        df["syn_size_distribution_spine_dendrite_p90"] = df[
            "syn_size_distribution_spine_dendrite"
        ].apply(lambda x: np.percentile(x, 90))
        df["syn_size_spine_dendrite_cv"] = df[
            "syn_size_distribution_spine_dendrite"
        ].apply(np.std) / (
            df["syn_size_distribution_spine_dendrite"].apply(np.mean) + 1
        )

        df["syn_size_distribution_shaft_dendrite_p50"] = df[
            "syn_size_distribution_shaft_dendrite"
        ].apply(np.median)
        df["syn_size_distribution_shaft_dendrite_p10"] = df[
            "syn_size_distribution_shaft_dendrite"
        ].apply(lambda x: np.percentile(x, 10))
        df["syn_size_distribution_shaft_dendrite_p90"] = df[
            "syn_size_distribution_shaft_dendrite"
        ].apply(lambda x: np.percentile(x, 90))
        df["syn_size_shaft_dendrite_cv"] = df[
            "syn_size_distribution_shaft_dendrite"
        ].apply(np.std) / (
            df["syn_size_distribution_shaft_dendrite"].apply(np.mean) + 1
        )

    df["syn_size_distribution_soma_p50"] = df["syn_size_distribution_soma"].apply(
        np.median
    )
    dist_binned_cols = []

    for ii in range(len(df["dendrite_length_binned"].iloc[0])):
        df[f"dendrite_length_binned_{ii}"] = np.vstack(
            df["dendrite_length_binned"].values
        )[:, ii]
        dist_binned_cols.append(f"dendrite_length_binned_{ii}")
    if use_spines:
        dist_binned_cols_spine = []
        df["frac_syn_spine_soma"] = df["num_spine_syn_soma"] / df["num_syn_soma"]
        df["frac_syn_spine_dendrite"] = (
            df["num_spine_syn_dendrite"] / df["num_syn_dendrite"]
        )
        df["syn_spine_shaft_ratio_dendrite"] = np.log2(
            (df["num_spine_syn_dendrite"] + 1) / (df["num_shaft_syn_dendrite"] + 1)
        )
        for ii in range(len(df["syn_count_dist_binned_shaft"].iloc[0])):
            df[f"syn_count_dist_binned_shaft_{ii}"] = np.vstack(
                df["syn_count_dist_binned_shaft"].values
            )[:, ii]
            df[f"syn_count_dist_binned_spine_{ii}"] = np.vstack(
                df["syn_count_dist_binned_spine"].values
            )[:, ii]
            df[f"syn_count_dist_binned_ratio_{ii}"] = np.log2(
                (df[f"syn_count_dist_binned_spine_{ii}"] + 1)
                / (df[f"syn_count_dist_binned_shaft_{ii}"] + 1)
            )

            df[f"spine_density_binned_{ii}"] = df[
                f"syn_count_dist_binned_spine_{ii}"
            ] / (df[f"dendrite_length_binned_{ii}"] + 0.001)
            df[f"shaft_density_binned_{ii}"] = df[
                f"syn_count_dist_binned_shaft_{ii}"
            ] / (df[f"dendrite_length_binned_{ii}"] + 0.001)
            dist_binned_cols_spine.extend(
                [
                    f"syn_count_dist_binned_shaft_{ii}",
                    f"syn_count_dist_binned_spine_{ii}",
                    f"syn_count_dist_binned_ratio_{ii}",
                    f"spine_density_binned_{ii}",
                    f"shaft_density_binned_{ii}",
                ]
            )
    df["syn_depth_dist_p5"] = df["syn_depth_dist_all"].apply(
        lambda x: np.percentile(x, 2.5)
    )
    df["syn_depth_dist_p95"] = df["syn_depth_dist_all"].apply(
        lambda x: np.percentile(x, 97.5)
    )
    df["syn_depth_extent"] = df["syn_depth_dist_p95"] - df["syn_depth_dist_p5"]

    dbr = np.vstack(df["branches_dist"].values)
    if model_dict is None:
        svd_br = decomposition.TruncatedSVD(n_branch_comp)
        Xbr = svd_br.fit_transform(dbr)
    else:
        svd_br = model_dict.get("svd_br")
        Xbr = svd_br.transform(dbr)
    for ii in range(Xbr.shape[1]):
        df[f"branch_svd{ii}"] = Xbr[:, ii]

    pl_dat = np.vstack(df["syn_count_depth_dendrite"].values)
    pl_dat_norm = pl_dat / np.atleast_2d(pl_dat.sum(axis=1)).T
    if model_dict is None:
        syn_pca_pproc = preprocessing.StandardScaler()
        keep_dat_cols = np.sum(pl_dat, axis=0) > 0
        pl_dat_z = syn_pca_pproc.fit_transform(pl_dat_norm[:, keep_dat_cols])
        syn_pca = decomposition.SparsePCA(n_components=n_syn_comp)
        X = syn_pca.fit_transform(pl_dat_z)
    else:
        keep_dat_cols = model_dict.get("keep_dat_cols")
        syn_pca_pproc = model_dict.get("syn_pca_pproc")
        pl_dat_z = syn_pca_pproc.transform(pl_dat[:, keep_dat_cols])
        syn_pca = model_dict.get("syn_pca")
        X = syn_pca.transform(pl_dat_z)
    for ii in range(X.shape[1]):
        df[f"syn_count_pca{ii}"] = X[:, ii]

    pl_dat = np.vstack(df["egocentric_bins"].values)
    pl_dat_norm = pl_dat / np.atleast_2d(pl_dat.sum(axis=1)).T
    if model_dict is None:
        ego_syn_pca = decomposition.SparsePCA(n_components=n_syn_ego)
        ego_pproc = preprocessing.StandardScaler()
        pl_dat_norm_z = ego_pproc.fit_transform(pl_dat_norm)
        Xego = ego_syn_pca.fit_transform(pl_dat_norm_z)
    else:
        ego_syn_pca = model_dict.get("ego_syn_pca")
        ego_pproc = model_dict.get("ego_pproc")
        pl_dat_norm_z = ego_pproc.transform(pl_dat_norm)
        Xego = ego_syn_pca.transform(pl_dat_norm_z)
    for ii in range(Xego.shape[1]):
        df[f"ego_count_pca{ii}"] = Xego[:, ii]

    if model_dict is None:
        model_dict = {}
        model_dict["svd_br"] = svd_br
        model_dict["keep_dat_cols"] = keep_dat_cols
        model_dict["syn_pca_pproc"] = syn_pca_pproc
        model_dict["syn_pca"] = syn_pca
        model_dict["ego_syn_pca"] = ego_syn_pca
        model_dict["ego_pproc"] = ego_pproc

    pl_depth = np.vstack(df["path_length_depth_dendrite"].values)
    sc_depth = np.vstack(df["syn_count_depth_dendrite"].values)
    keep_cols = pl_depth.sum(axis=0) > 0
    density_nan = sc_depth[:, keep_cols] / pl_depth[:, keep_cols]
    density_nan[np.isnan(density_nan)] = 0
    density_nan[np.isinf(density_nan)] = 0
    df["median_density"] = percentile_above_zero(density_nan, 50)
    if use_spines:
        sc_depth_spine = np.vstack(df["syn_count_depth_spine_dendrite"].values)
        density_nan_spine = sc_depth_spine[:, keep_cols] / pl_depth[:, keep_cols]
        density_nan_spine[np.isnan(density_nan_spine)] = 0
        density_nan_spine[np.isinf(density_nan_spine)] = 0
        df["median_density_spine"] = percentile_above_zero(density_nan_spine, 50)

        sc_depth_shaft = np.vstack(df["syn_count_depth_shaft_dendrite"].values)
        density_nan_shaft = sc_depth_shaft[:, keep_cols] / pl_depth[:, keep_cols]
        density_nan_shaft[np.isnan(density_nan_shaft)] = 0
        density_nan_shaft[np.isinf(density_nan_shaft)] = 0
        df["median_density_shaft"] = percentile_above_zero(density_nan_shaft, 50)

    dat_cols = [
        "tip_len_dist_dendrite_p50",
        "tip_tort_dendrite_p50",
        "num_syn_dendrite",
        "num_syn_soma",
        "path_length_dendrite",
        "radial_extent_dendrite",
        "syn_dist_distribution_dendrite_p50",
        "syn_size_distribution_soma_p50",
        "syn_size_distribution_dendrite_p50",
        "syn_size_distribution_dendrite_p10",
        "syn_size_distribution_dendrite_p90",
        "syn_size_dendrite_cv",
        "syn_depth_dist_p5",
        "syn_depth_dist_p95",
        "syn_depth_extent",
        "median_density",
        "radius_dist",
        "area_factor",
    ] + dist_binned_cols
    if use_spines:
        dat_cols += [
            "syn_dist_distribution_dendrite_spine_p50",
            "syn_dist_distribution_dendrite_shaft_p50",
            "dend_spine_shaft_offset",
            "syn_size_distribution_spine_dendrite_p50",
            "syn_size_distribution_spine_dendrite_p10",
            "syn_size_distribution_spine_dendrite_p90",
            "syn_size_spine_dendrite_cv",
            "syn_size_distribution_shaft_dendrite_p50",
            "syn_size_distribution_shaft_dendrite_p10",
            "syn_size_distribution_shaft_dendrite_p90",
            "syn_size_shaft_dendrite_cv",
            "frac_syn_spine_soma",
            "frac_syn_spine_dendrite",
            "median_density_spine",
            "median_density_shaft",
            "syn_spine_shaft_ratio_dendrite",
            "num_spine_syn_dendrite",
            "num_shaft_syn_dendrite",
            "num_spine_syn_soma",
        ] + dist_binned_cols_spine

    for ii in range(X.shape[1]):
        dat_cols.append(f"syn_count_pca{ii}")
    for ii in range(Xbr.shape[1]):
        dat_cols.append(f"branch_svd{ii}")
    for ii in range(Xego.shape[1]):
        dat_cols.append(f"ego_count_pca{ii}")

    return_cols = ["root_id", "soma_depth"] + dat_cols
    return (
        df[return_cols],
        dat_cols,
        syn_pca,
        svd_br,
        keep_dat_cols,
        ego_syn_pca,
        model_dict,
    )
