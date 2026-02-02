import os
import numpy as np
from sklearn.cluster import Birch
from src.config import load_config

def main():
    cfg = load_config("configs/config.yaml")
    V_path = cfg.paths["emb_pca_mmap"]
    out_labels = cfg.paths["micro_labels"]
    out_ids = cfg.paths["micro_ids"]
    out_cent = cfg.paths["micro_centroids"]
    dim = int(cfg.paths["pca_dim"])
    thr = float(cfg.params["birch_threshold"])

    os.makedirs(os.path.dirname(out_labels), exist_ok=True)

    n = os.path.getsize(V_path) // (4 * dim)
    V = np.memmap(V_path, dtype="float32", mode="r", shape=(n, dim))

    bir = Birch(threshold=thr, n_clusters=None)
    micro = bir.fit_predict(V).astype(np.int32)

    micro_ids, inv = np.unique(micro, return_inverse=True)
    m = len(micro_ids)

    centroids = np.zeros((m, dim), dtype=np.float32)
    counts = np.zeros(m, dtype=np.int32)

    for i in range(n):
        c = inv[i]
        centroids[c] += V[i]
        counts[c] += 1

    centroids /= counts[:, None]
    norm = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    centroids = (centroids / norm).astype(np.float32)

    np.save(out_labels, micro)
    np.save(out_ids, micro_ids.astype(np.int32))
    np.save(out_cent, centroids)

    print("micro_labels:", out_labels)
    print("micro_ids:", out_ids)
    print("micro_centroids:", out_cent)
    print("microclusters:", m)

if __name__ == "__main__":
    main()