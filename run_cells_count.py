import skel_features
from standard_transform import minnie_ds
import pandas as pd
import numpy as np
from cloudpathlib import AnyPath as Path
from caveclient import CAVEclient


skel_dir = Path("../skeletonization/skeletons/minnie65_phase3_v1")
feature_dir = "features/minnie65_phase3_v1"

height_bnds = [0, 950]
layer_bnds = np.load(
    "/Users/caseysm/Work/Projects/MinnieColumn/data/layer_bounds_v3.npy"
)

skel_dir = (
    "s3://bossdb-open-data/iarpa_microns/minnie/minnie65/skeletons/v661/meshworks"
)

client = CAVEclient("minnie65_phase3_v1")

soma_df = client.materialize.tables.aibs_metamodel_mtypes_v661_v2(
    classification_system=["excitatory_neuron", "inhibitory_neuron"]
).query(
    desired_resolution=[1, 1, 1],
)
soma_df["pt_root_id"] = client.chunkedgraph.get_roots(
    soma_df["pt_supervoxel_id"],
    timestamp=client.materialize.get_timestamp(795),
)
soma_df["filename"] = soma_df.apply(
    lambda row: f"{row['pt_root_id']}_{row['id']}.h5", axis=1
)

xyz0 = np.array([178400, 143248, 21234]) * [4, 4, 40]
soma_df["dist_from_ctr"] = minnie_ds.streamline_nm.radial_distance(
    xyz0,
    np.vstack(soma_df["pt_position"]),
)
run_df = soma_df.copy()
run_df["pt_root_id"] = run_df["pt_root_id"].astype(int)
print(run_df.query("pt_root_id == 864691134885792506"))
# print(len(run_df.query('pt_root_id != 0')))
