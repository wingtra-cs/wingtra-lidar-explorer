"""
lidar_r2.py — R2 access layer for the LiDAR Survey Explorer.

Read-only. Uses the same .streamlit/secrets.toml as the crop health tool:
    [r2]
    account_id = "..."
    access_key = "..."   # read-only token
    secret_key = "..."
    bucket     = "ptpn-bucket"

Public API:
    list_surveys()       -> [{slug, name, meta}]  — ready bundles only
    load_bundle(slug)    -> {slug, meta, layers, dtm, dtm_meta}
    refresh_surveys()    -> clears discovery cache
    presigned_url(key)   -> temporary download URL
"""

import io, json
import numpy as np
import pandas as pd
import streamlit as st
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

LIDAR_PREFIX = "lidar/"


@st.cache_resource(show_spinner=False)
def _client():
    s = st.secrets["r2"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{s['account_id']}.r2.cloudflarestorage.com",
        aws_access_key_id=s["access_key"],
        aws_secret_access_key=s["secret_key"],
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
    )

def _bucket():
    return st.secrets["r2"]["bucket"]

def _get_bytes(key):
    try:
        obj = _client().get_object(Bucket=_bucket(), Key=key)
        return obj["Body"].read()
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NotFound"):
            return None
        raise


@st.cache_data(ttl=300, show_spinner=False)
def list_surveys():
    """List ready bundles under lidar/. A bundle is ready only when meta.json exists."""
    paginator = _client().get_paginator("list_objects_v2")
    out = []
    for page in paginator.paginate(Bucket=_bucket(), Prefix=LIDAR_PREFIX, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            prefix = cp["Prefix"]
            slug = prefix[len(LIDAR_PREFIX):].strip("/")
            if not slug:
                continue
            raw = _get_bytes(prefix + "meta.json")
            if raw is None:
                continue
            try:
                meta = json.loads(raw.decode("utf-8"))
            except Exception:
                continue
            out.append({"slug": slug, "name": meta.get("name", slug), "meta": meta})
    out.sort(key=lambda d: d["name"].lower())
    return out


def refresh_surveys():
    list_surveys.clear()


@st.cache_resource(show_spinner=True)
def load_bundle(slug):
    """Load one survey's display data. Cached as a resource keyed by slug.

    Returns:
        {
          "slug":     str,
          "meta":     dict,            # parsed meta.json
          "layers":   {name: DataFrame},  # point cloud layer(s)
          "dtm":      float32 ndarray or None,
          "dtm_meta": dict or None,
        }
    """
    prefix = f"{LIDAR_PREFIX}{slug}/"

    raw_meta = _get_bytes(prefix + "meta.json")
    if raw_meta is None:
        raise FileNotFoundError(f"{slug}: meta.json not found")
    meta = json.loads(raw_meta.decode("utf-8"))

    # Point cloud layers (points.parquet / ground.parquet / nonground.parquet)
    layers = {}
    for name in meta.get("layers", ["points"]):
        raw = _get_bytes(f"{prefix}{name}.parquet")
        if raw is None:
            raise FileNotFoundError(f"{slug}: {name}.parquet not found")
        layers[name] = pd.read_parquet(io.BytesIO(raw))

    # DTM (optional)
    dtm, dtm_meta = None, None
    if meta.get("dtm_available"):
        raw_dtm = _get_bytes(prefix + "dtm.npy")
        raw_dtm_meta = _get_bytes(prefix + "dtm_meta.json")
        if raw_dtm and raw_dtm_meta:
            dtm = np.load(io.BytesIO(raw_dtm), allow_pickle=False)
            dtm_meta = json.loads(raw_dtm_meta.decode("utf-8"))

    return {"slug": slug, "meta": meta, "layers": layers,
            "dtm": dtm, "dtm_meta": dtm_meta}


def presigned_url(key, ttl=900):
    """Temporary direct-download URL (browser ↔ R2, bypasses app)."""
    return _client().generate_presigned_url(
        "get_object",
        Params={"Bucket": _bucket(), "Key": key},
        ExpiresIn=int(ttl),
    )
