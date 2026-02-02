import os
import numpy as np
import faiss
import igraph as ig
import leidenalg
from sklearn.metrics import adjusted_mutual_info_score
from src.config import load_config

def build_merge_graph(C, knn=40, thr=0.75):
    k0, d = C.shape
    index = faiss.IndexFlatIP(d)
    index.add(C)
    sims, nbrs = index.search(C, min(knn, k0))

    edges = []
    weights = []
    for i in range(k0):
        js = nbrs[i][1:]
        ss = sims[i][1:]
        for j, s in zip(js, ss):
            if j < 0 or j == i:
                continue
            if s >= thr:
                edges.append((i, int(j)))
                weights.append(float(s))
    if not edges:
        return None

    g = ig.Graph(n=k0, edges=edges, directed=False)
    g.es["weight"] = weights
    return g

def leiden_labels(g, seed=42, res=0.5):
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights=g.es["weight"],
        resolution_parameter=float(res),
        seed=int(seed),
    )
    return np.array(part.membership, dtype=np.int32)

def main():
    cfg = load_config("configs/config.yaml")
    V_path = cfg.paths["emb_pca_mmap"]
    dim = int(cfg.paths["pca_dim"])
    pred_path = cfg.paths["final_clusters"]
    out_path = os.path.join("data", "processed", "final_cluster_labels_merged.npy")

    n = os.path.getsize(V_path) // (4 * dim)
    V = np.memmap(V_path, dtype="float32", mode="r", shape=(n, dim))

    y = np.load(pred_path).astype(np.int32)
    uniq = np.unique(y)
    k0 = len(uniq)
    print("fine clusters:", k0)

    remap = {c:i for i, c in enumerate(uniq)}
    yi = np.vectorize(remap.get)(y).astype(np.int32)

    C = np.zeros((k0, dim), dtype=np.float32)
    cnt = np.zeros((k0,), dtype=np.int32)
    for i in range(n):
        c = yi[i]
        C[c] += V[i]
        cnt[c] += 1
    C /= cnt[:, None]
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)

    thresholds = [0.70, 0.705, 0.71, 0.715, 0.72, 0.725, 0.73]
    knn = 50 if k0 >= 800 else 40
    res = 0.45

    rng = np.random.default_rng(42)
    sample_frac = 0.85

    best = None
    for thr in thresholds:
        g = build_merge_graph(C, knn=knn, thr=thr)
        if g is None:
            print(f"thr={thr:.2f}: no edges")
            continue

        # stability via subsampling nodes
        # run twice on two random subsets, compare on intersection
        nodes1 = rng.choice(k0, size=int(sample_frac * k0), replace=False)
        nodes2 = rng.choice(k0, size=int(sample_frac * k0), replace=False)
        nodes1_set = set(nodes1.tolist())
        nodes2_set = set(nodes2.tolist())
        inter = np.array(sorted(nodes1_set.intersection(nodes2_set)), dtype=np.int32)

        g1 = g.induced_subgraph(nodes1.tolist())
        g2 = g.induced_subgraph(nodes2.tolist())

        lab1 = leiden_labels(g1, seed=1, res=res)
        lab2 = leiden_labels(g2, seed=2, res=res)

        # map labels back to original node ids for intersection
        map1 = {node: lab1[i] for i, node in enumerate(nodes1.tolist())}
        map2 = {node: lab2[i] for i, node in enumerate(nodes2.tolist())}
        a = np.array([map1[i] for i in inter], dtype=np.int32)
        b = np.array([map2[i] for i in inter], dtype=np.int32)

        stab = adjusted_mutual_info_score(a, b)
        # small penalty to avoid extreme merging
        full_lab = leiden_labels(g, seed=42, res=res)
        k1 = len(np.unique(full_lab))
        if not (210 <= k1 <= 240):
            print(f"thr={thr:.2f} skipped (merged_k={k1})")
            continue
        score = stab + 0.05 * np.log(k0 / k1)
        print(f"thr={thr:.2f} -> merged_k={k1} stability(AMI)={stab:.3f} score={score:.3f}")

        if best is None or score > best["score"]:
            best = {"thr": thr, "labels": full_lab, "k": k1, "score": score}

    if best is None:
        raise RuntimeError("No valid merge threshold found.")

    print("chosen thr:", best["thr"], "merged_k:", best["k"], "score:", best["score"])

    merged = best["labels"]
    y_new = merged[yi].astype(np.int32)
    np.save(out_path, y_new)
    print("saved:", out_path)

if __name__ == "__main__":
    main()