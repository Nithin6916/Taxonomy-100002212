import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    v_measure_score,
    adjusted_mutual_info_score,
)

from src.utils.metrics import purity_score


def load_truth():
    labels = pd.read_parquet("data/interim/labels.parquet").set_index("product_id")
    product_ids = np.load("data/processed/product_ids.npy").astype(np.int64)
    y_true = labels.loc[product_ids, "L3"].astype(str).to_numpy()
    return y_true, product_ids


def eval_variant(y_true, y_pred):
    return {
        "purity": float(purity_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "v_measure": float(v_measure_score(y_true, y_pred)),
        "ami": float(adjusted_mutual_info_score(y_true, y_pred)),
        "n_clusters": int(len(np.unique(y_pred))),
    }


def top_labels_report(y_true, y_pred, out_csv, topk=3):
    df = pd.DataFrame({"cluster": y_pred, "L3": y_true})
    rows = []
    for c, g in df.groupby("cluster"):
        vc = g["L3"].value_counts()
        total = int(vc.sum())
        for rank, (lab, cnt) in enumerate(vc.head(topk).items(), start=1):
            rows.append(
                {
                    "cluster": int(c),
                    "rank": rank,
                    "label": str(lab),
                    "count": int(cnt),
                    "share": float(cnt / total),
                    "cluster_size": total,
                }
            )
    rep = pd.DataFrame(rows).sort_values(
        ["cluster_size", "cluster", "rank"],
        ascending=[False, True, True]
    )
    rep.to_csv(out_csv, index=False)


def plot_metric_bars(df, out_png):
    metrics = ["purity", "nmi", "ari", "ami", "v_measure"]
    x = np.arange(len(metrics))
    width = 0.35

    r_p = df[df["variant"] == "purity_optimized"].iloc[0]
    r_b = df[df["variant"] == "balanced"].iloc[0]

    y1 = [r_p[m] for m in metrics]
    y2 = [r_b[m] for m in metrics]

    plt.figure()
    plt.bar(x - width / 2, y1, width, label="purity_optimized")
    plt.bar(x + width / 2, y2, width, label="balanced")
    plt.xticks(x, metrics, rotation=20)
    plt.ylim(0, 1.0)
    plt.ylabel("score")
    plt.title("Metric comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_cluster_counts(df, out_png):
    plt.figure()
    plt.bar(df["variant"], df["n_clusters"])
    plt.ylabel("number of clusters")
    plt.title("Cluster count comparison")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def cluster_sizes(y_pred):
    _, counts = np.unique(y_pred, return_counts=True)
    return counts.astype(np.int32)


def per_cluster_majority_share(y_true, y_pred):
    df = pd.DataFrame({"cluster": y_pred, "L3": y_true})
    shares = []
    for _, g in df.groupby("cluster"):
        vc = g["L3"].value_counts()
        shares.append(float(vc.iloc[0] / vc.sum()))
    return np.array(shares, dtype=np.float32)


def plot_hist_compare(a, b, title, xlabel, out_png, bins=60, log=False):
    plt.figure()
    plt.hist(a, bins=bins, alpha=0.6, label="purity_optimized")
    plt.hist(b, bins=bins, alpha=0.6, label="balanced")
    if log:
        plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    out_dir = os.path.join("outputs", "final")
    os.makedirs(out_dir, exist_ok=True)

    purity_src = "data/processed/final_cluster_labels.npy"
    balanced_src = "data/processed/final_cluster_labels_merged.npy"

    if not os.path.exists(purity_src):
        raise RuntimeError(f"Missing {purity_src}. Run step 06 first (fine clustering).")
    if not os.path.exists(balanced_src):
        raise RuntimeError(f"Missing {balanced_src}. Run merge step first (balanced clustering).")

    purity_dst = os.path.join(out_dir, "clusters_purity_optimized.npy")
    balanced_dst = os.path.join(out_dir, "clusters_balanced.npy")

    y_pred_purity = np.load(purity_src).astype(np.int32)
    y_pred_bal = np.load(balanced_src).astype(np.int32)

    np.save(purity_dst, y_pred_purity)
    np.save(balanced_dst, y_pred_bal)

    y_true, _ = load_truth()

    m1 = eval_variant(y_true, y_pred_purity)
    m2 = eval_variant(y_true, y_pred_bal)

    rows = [
        {"variant": "purity_optimized", **m1},
        {"variant": "balanced", **m2},
    ]
    df = pd.DataFrame(rows)

    metrics_json = os.path.join(out_dir, "metrics_comparison.json")
    metrics_csv = os.path.join(out_dir, "metrics_comparison.csv")

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    df.to_csv(metrics_csv, index=False)

    plot_metric_bars(df, os.path.join(out_dir, "metrics_comparison_bar.png"))
    plot_cluster_counts(df, os.path.join(out_dir, "cluster_count_comparison.png"))

    top_labels_report(y_true, y_pred_purity, os.path.join(out_dir, "top_labels_per_cluster_purity.csv"), topk=3)
    top_labels_report(y_true, y_pred_bal, os.path.join(out_dir, "top_labels_per_cluster_balanced.csv"), topk=3)

    sizes_p = cluster_sizes(y_pred_purity)
    sizes_b = cluster_sizes(y_pred_bal)
    maj_p = per_cluster_majority_share(y_true, y_pred_purity)
    maj_b = per_cluster_majority_share(y_true, y_pred_bal)

    plot_hist_compare(
        sizes_p, sizes_b,
        title="Cluster size distribution (counts per cluster)",
        xlabel="cluster size",
        out_png=os.path.join(out_dir, "cluster_size_hist.png"),
        bins=80,
        log=True
    )

    plot_hist_compare(
        np.log10(sizes_p), np.log10(sizes_b),
        title="Cluster size distribution (log10 scale)",
        xlabel="log10(cluster size)",
        out_png=os.path.join(out_dir, "cluster_size_hist_log10.png"),
        bins=80,
        log=False
    )

    plot_hist_compare(
        maj_p, maj_b,
        title="Per-cluster majority share (cluster purity distribution)",
        xlabel="majority share (0..1)",
        out_png=os.path.join(out_dir, "per_cluster_majority_share_hist.png"),
        bins=60,
        log=False
    )

    print("saved:", purity_dst)
    print("saved:", balanced_dst)
    print("saved:", metrics_json)
    print("saved:", metrics_csv)
    print("saved plots + reports in:", out_dir)


if __name__ == "__main__":
    main()