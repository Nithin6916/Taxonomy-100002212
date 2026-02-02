import os
import numpy as np
import faiss
import igraph as ig
import leidenalg
from sklearn.metrics import adjusted_mutual_info_score
from src.config import load_config

def build_graph(C, k, sim_thr):
    m, d = C.shape
    index = faiss.IndexFlatIP(d)
    index.add(C)
    sims, nbrs = index.search(C, min(k, m))

    src_list, dst_list, w_list = [], [], []
    for i in range(m):
        js = nbrs[i]
        ss = sims[i]
        mask = (js >= 0) & (js != i) & (ss >= sim_thr)
        js = js[mask].astype(np.int32)
        ss = ss[mask].astype(np.float32)
        if js.size:
            src_list.append(np.full(js.shape, i, dtype=np.int32))
            dst_list.append(js)
            w_list.append(ss)

    if not src_list:
        raise RuntimeError("No edges kept; lower centroid_sim_quantile.")

    src = np.concatenate(src_list)
    dst = np.concatenate(dst_list)
    wts = np.concatenate(w_list)

    g = ig.Graph(n=m, edges=list(zip(src.tolist(), dst.tolist())), directed=False)
    g.es["weight"] = wts.tolist()
    return g

def run_leiden(g, res, seed):
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights=g.es["weight"],
        resolution_parameter=float(res),
        seed=int(seed),
    )
    lab = np.array(part.membership, dtype=np.int32)
    _, cnts = np.unique(lab, return_counts=True)
    singletons = float((cnts == 1).mean())
    return lab, singletons

def main():
    cfg = load_config("configs/config.yaml")
    cent_path = cfg.paths["micro_centroids"]
    micro_labels_path = cfg.paths["micro_labels"]
    micro_ids_path = cfg.paths["micro_ids"]
    out_final = cfg.paths["final_clusters"]

    C = np.load(cent_path).astype(np.float32)
    micro = np.load(micro_labels_path).astype(np.int32)
    micro_ids = np.load(micro_ids_path).astype(np.int32)

    m, d = C.shape
    k = int(cfg.params["centroid_knn"])
    q = float(cfg.params["centroid_sim_quantile"])
    res_grid = cfg.params["leiden_resolution_grid"]

    index = faiss.IndexFlatIP(d)
    index.add(C)
    sims, _ = index.search(C, min(k, m))
    flat = sims[:, 1:].ravel()
    sim_thr = float(np.quantile(flat, q))
    sim_thr = max(0.25, min(sim_thr, 0.45))

    g = build_graph(C, k, sim_thr)

    best = None
    for res in res_grid:
        lab1, s1 = run_leiden(g, res, seed=1)
        lab2, _ = run_leiden(g, res, seed=2)
        stab = adjusted_mutual_info_score(lab1, lab2)
        score = stab - 0.7 * s1
        if best is None or score > best["score"]:
            best = {"res": float(res), "score": float(score)}

    res = best["res"]
    centroid_cluster, _ = run_leiden(g, res, seed=42)

    micro_idx = np.searchsorted(micro_ids, micro)
    final = centroid_cluster[micro_idx].astype(np.int32)

    os.makedirs(os.path.dirname(out_final), exist_ok=True)
    np.save(out_final, final)

    print("resolution:", res)
    print("final_clusters:", out_final, "K:", int(len(np.unique(final))))

if __name__ == "__main__":
    main()