import skel_features
from caveclient import CAVEclient
from cloudpathlib import AnyPath as Path
import cloudfiles
from io import BytesIO
import numpy as np
from joblib import Parallel, delayed
import json
from cloudpathlib import AnyPath


def convert_dendrite_to_swc(root_id, skel_dir, cf):
    if cf.exists(f"{root_id}.swc"):
        pass
    try:
        nrn = skel_features.io_utils.load_root_id(root_id, skel_dir, peel_threshold=1)
        nrn.apply_mask(nrn.anno.is_dendrite.mesh_mask)

        with BytesIO() as bio:
            nrn.export_to_swc(
                bio,
                resample_spacing=1000,
                dendrite_label="is_dendrite",
                radius=nrn.anno.segment_properties.df["r_eff"].values,
            )
            bio.seek(0)
            cf.put(f"{root_id}.swc", bio.read())
        with open(f"swc_conversion/{root_id}.json", "w") as f:
            json.dump({"success": True, "root_id": root_id}, f)
    except Exception as e:
        with open(f"swc_conversion/{root_id}.json", "w") as f:
            json.dump({"success": False, "root_id": root_id, "error": str(e)}, f)
        return False


client = CAVEclient("minnie65_public")
client.version = 1300


skel_dir = Path("gs://csm-skel/skel-test/")
dend_files = "gs://microns-static-links/skel/swc/dendrite"
cf = cloudfiles.CloudFiles(dend_files)

all_files = list(AnyPath(dend_files).glob("*.swc"))
done_ids = [int(fn.name.split(".")[0]) for fn in all_files]
print("Already done:", len(done_ids))

if __name__ == "__main__":
    tbl_qry = client.materialize.tables
    rids = tbl_qry.aibs_metamodel_mtypes_v661_v2().get_all().pt_root_id.values
    rids = np.setdiff1d(rids, done_ids)
    print("To do:", len(rids))

    success = Parallel(n_jobs=25)(
        delayed(convert_dendrite_to_swc)(int(root_id), skel_dir, cf) for root_id in rids
    )
