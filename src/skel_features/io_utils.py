import json
import sys
from pathlib import Path

import pandas as pd
from cloudpathlib import AnyPath, GSClient, S3Client
from google.cloud import storage
from meshparty import meshwork

from . import skel_filtering as filtering

path = Path(__file__)
sys.path.append(str(path.absolute().parent))

# Configure cloudpathlib to use non-token based clients for public buckets
storage_client = storage.Client.create_anonymous_client()
GSClient(storage_client=storage_client).set_as_default_client()
S3Client(no_sign_request=True).set_as_default_client()


def load_root_id(oid, nrn_dir, peel_threshold=0):
    "Load and apply dendrite labels to a neuron based on pre-classified synapse apical labels and more"
    cpath = AnyPath(nrn_dir)
    if isinstance(oid, str):
        filename = oid
    else:
        filename = f"{oid}.h5"
    with open(cpath / filename, "rb") as f:
        nrn = meshwork.load_meshwork(f)
        nrn.reset_mask()
    filtering.additional_component_masks(nrn, peel_threshold=peel_threshold)
    return nrn


def load_features(root_ids, feature_dir, peel_threshold=0):
    "Load a feature dataframe from a directory"
    dats = []
    dpath = AnyPath(feature_dir)
    for root_id in root_ids:
        filename = f"{root_id}.json"
        try:
            with open(dpath / filename, "r") as f:
                dats.append(json.load(f))
        except:
            continue
    return pd.DataFrame.from_records(dats)
