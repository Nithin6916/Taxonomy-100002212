import os, json
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, v_measure_score, adjusted_mutual_info_score
from src.config import load_config
from src.utils.metrics import purity_score

def main():
    cfg = load_config("configs/config.yaml")
    labels_path = cfg.paths["labels_parquet"]
    pred_path = cfg.paths["final_clusters"]
    ids_path = cfg.paths["product_ids_npy"]
    out_json = cfg.paths["metrics_json"]
    out_csv = cfg.paths["cluster_report_csv"]

    labels = pd.read_parquet(labels_path)
    labels = labels.set_index("product_id")

    y_pred = np.load(pred_path).astype(np.int32)
    product_ids = np.load(ids_path).astype(np.int64)

    y_true = labels.loc[product_ids, "L3"].astype(str).to_numpy()

    pur = purity_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    vms = v_measure_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    k = int(len(np.unique(y_pred)))

    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    metrics = {"purity": pur, "nmi": nmi, "ari": ari, "v_measure": vms, "ami": ami, "n_clusters": k}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    df = pd.DataFrame({"cluster": y_pred, "L3": y_true})
    grp = df.groupby("cluster")["L3"]
    report = grp.value_counts().groupby(level=0).head(1).reset_index()
    report = report.rename(columns={"L3": "majority_label", "count": "majority_count"})
    sizes = df.groupby("cluster").size().rename("size").reset_index()
    report = sizes.merge(report, on="cluster", how="left").sort_values("size", ascending=False)
    report.to_csv(out_csv, index=False)

    print(metrics)
    print("metrics:", out_json)
    print("report:", out_csv)

if __name__ == "__main__":
    main()