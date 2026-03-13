import os
import json
import numpy as np
import pandas as pd

from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_memmap(path: str, n: int, d: int):
    return np.memmap(path, dtype="float32", mode="r", shape=(n, d))


def compute_pca_artifacts(
    emb_path: str,
    out_dir: str,
    n: int,
    d_in: int = 1024,
    d_out: int = 256,
    batch: int = 4096,
):
    X = load_memmap(emb_path, n, d_in)
    ipca = IncrementalPCA(n_components=d_out, batch_size=batch)

    for i in range(0, n, batch):
        ipca.partial_fit(X[i : i + batch])

    np.save(os.path.join(out_dir, "pca_mean.npy"), ipca.mean_.astype(np.float32))
    np.save(os.path.join(out_dir, "pca_components.npy"), ipca.components_.astype(np.float32))


def compute_cluster_centroids(
    V_pca_path: str,
    labels_path: str,
    out_path: str,
    n: int,
    d: int = 256,
):
    V = load_memmap(V_pca_path, n, d)
    y = np.load(labels_path).astype(np.int32)

    uniq = np.unique(y)  # original cluster IDs
    remap = {c: i for i, c in enumerate(uniq)}
    yi = np.vectorize(remap.get)(y).astype(np.int32)  # 0..K-1

    k = len(uniq)
    C = np.zeros((k, d), dtype=np.float32)
    cnt = np.zeros((k,), dtype=np.int32)

    for i in range(n):
        c = yi[i]
        C[c] += V[i]
        cnt[c] += 1

    C /= cnt[:, None]
    C = normalize(C, norm="l2").astype(np.float32)

    np.save(out_path, C)
    return uniq, yi


def build_cluster_metadata(
    features_parquet: str,
    labels_array: np.ndarray,
    uniq_clusters: np.ndarray,
    out_json: str,
    top_terms: int = 10,
    top_examples: int = 5,
):
    y = labels_array
    meta = {}

    # Always store sizes
    for c in uniq_clusters:
        members = np.where(y == c)[0]
        meta[str(int(c))] = {
            "size": int(members.size),
            "keywords": [],
            "examples": [],
        }

    # Try to enrich with keywords/examples
    try:
        df = pd.read_parquet(features_parquet)

        title_col = "Title" if "Title" in df.columns else ("ProductName" if "ProductName" in df.columns else df.columns[0])
        cols = [c for c in ["Title", "Brand", "ProductName"] if c in df.columns]
        if not cols:
            cols = [title_col]

        text = (
            df[cols]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .tolist()
        )

        n = len(text)
        sample_n = min(120000, n)
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=sample_n, replace=False)

        vect = TfidfVectorizer(
            max_features=80000,
            ngram_range=(1, 2),
            min_df=3,
            stop_words="english",
        )
        X = vect.fit_transform([text[i] for i in idx])
        vocab = np.array(vect.get_feature_names_out())

        pos = {int(g): j for j, g in enumerate(idx.tolist())}

        for c in uniq_clusters:
            members = np.where(y == c)[0]
            if members.size == 0:
                continue

            ex = members[:top_examples]
            examples = [text[int(i)][:200] for i in ex]

            sampled_members = [pos.get(int(i), -1) for i in members]
            sampled_members = [s for s in sampled_members if s != -1]

            if len(sampled_members) >= 5:
                mean_vec = X[sampled_members].mean(axis=0)
                mean_vec = np.asarray(mean_vec).ravel()
                top_idx = np.argsort(-mean_vec)[:top_terms]
                keywords = vocab[top_idx].tolist()
            else:
                keywords = []

            meta[str(int(c))]["keywords"] = keywords
            meta[str(int(c))]["examples"] = examples

        print("Metadata enriched using features.parquet")

    except Exception as e:
        print("WARNING: Could not read features.parquet for metadata. Using minimal metadata only.")
        print("Reason:", repr(e))

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    root = os.getcwd()
    out_dir = os.path.join(root, "artifacts")
    ensure_dir(out_dir)

    emb_path = os.path.join(root, "data", "processed", "emb_bge_m3.f32.mmap")
    pca_path = os.path.join(root, "data", "processed", "emb_pca256.f32.mmap")
    features_parquet = os.path.join(root, "data", "interim", "features.parquet")

    y_purity_path = os.path.join(root, "data", "processed", "final_cluster_labels.npy")
    y_bal_path = os.path.join(root, "data", "processed", "final_cluster_labels_merged.npy")

    prod_ids = np.load(os.path.join(root, "data", "processed", "product_ids.npy"))
    n = int(prod_ids.shape[0])

    # PCA artifacts for inference
    mean_path = os.path.join(out_dir, "pca_mean.npy")
    comp_path = os.path.join(out_dir, "pca_components.npy")
    if not (os.path.exists(mean_path) and os.path.exists(comp_path)):
        compute_pca_artifacts(emb_path, out_dir, n=n)

    # Centroids + mapping arrays
    uniq_p, _ = compute_cluster_centroids(
        pca_path,
        y_purity_path,
        os.path.join(out_dir, "centroids_purity.npy"),
        n=n,
    )
    uniq_b, _ = compute_cluster_centroids(
        pca_path,
        y_bal_path,
        os.path.join(out_dir, "centroids_balanced.npy"),
        n=n,
    )

    # Critical: centroid index -> original cluster_id mapping
    np.save(os.path.join(out_dir, "cluster_ids_purity.npy"), uniq_p.astype(np.int32))
    np.save(os.path.join(out_dir, "cluster_ids_balanced.npy"), uniq_b.astype(np.int32))

    # Metadata
    y_p = np.load(y_purity_path).astype(np.int32)
    y_b = np.load(y_bal_path).astype(np.int32)

    build_cluster_metadata(
        features_parquet,
        y_p,
        uniq_p,
        os.path.join(out_dir, "cluster_meta_purity.json"),
    )
    build_cluster_metadata(
        features_parquet,
        y_b,
        uniq_b,
        os.path.join(out_dir, "cluster_meta_balanced.json"),
    )

    print("Saved artifacts to:", out_dir)


if __name__ == "__main__":
    main()