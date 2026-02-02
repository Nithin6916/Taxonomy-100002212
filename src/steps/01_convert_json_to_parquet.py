import os
import pandas as pd
from src.config import load_config

def split_3levels(s):
    if pd.isna(s):
        return (None, None, None)
    parts = [p.strip() for p in str(s).split(">")]
    parts += [None] * (3 - len(parts))
    return parts[0], parts[1], parts[2]

def main():
    cfg = load_config("configs/config.yaml")
    raw = cfg.paths["raw_json"]
    feat_out = cfg.paths["features_parquet"]
    lab_out = cfg.paths["labels_parquet"]

    os.makedirs(os.path.dirname(feat_out), exist_ok=True)
    os.makedirs(os.path.dirname(lab_out), exist_ok=True)

    df = pd.read_json(raw)

    if "pathlist_names" not in df.columns:
        raise RuntimeError("Missing pathlist_names")

    df["L1"], df["L2"], df["L3"] = zip(*df["pathlist_names"].map(split_3levels))
    df = df[df["L3"].notna()].copy()
    df["L3"] = df["L3"].astype(str)

    if df.index.name is None:
        df.index.name = "product_id"

    FEATURE_COLS = [
        "Title",
        "Brand",
        "BrandInfo.BrandName",
        "ProductName",
        "BrandPartCode",
        "GTIN",
        "ProductFamily.Value",
        "ProductSeries.Value",
        "SummaryDescription.ShortSummaryDescription",
        "SummaryDescription.LongSummaryDescription",
        "Description.LongProductName",
        "Description.MiddleDesc",
        "Description.LongDesc",
        "BulletPoints.Values",
        "BulletPoints",
    ]
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]

    features = df[feature_cols].copy()
    features = features.reset_index()

    labels = df[["L1", "L2", "L3"]].copy().reset_index()

    features.to_parquet(feat_out, index=False)
    labels.to_parquet(lab_out, index=False)

    print("features:", feat_out, features.shape)
    print("labels:", lab_out, labels.shape)

if __name__ == "__main__":
    main()