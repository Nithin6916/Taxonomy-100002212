import os
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import normalize
from src.config import load_config

def main():
    cfg = load_config("configs/config.yaml")
    emb_in = cfg.paths["emb_mmap"]
    emb_out = cfg.paths["emb_pca_mmap"]
    dim_in = int(cfg.paths["emb_dim"])
    dim_out = int(cfg.paths["pca_dim"])
    ipca_batch = int(cfg.params["ipca_batch"])

    os.makedirs(os.path.dirname(emb_out), exist_ok=True)

    n = os.path.getsize(emb_in) // (4 * dim_in)
    V = np.memmap(emb_in, dtype="float32", mode="r", shape=(n, dim_in))

    ipca = IncrementalPCA(n_components=dim_out, batch_size=ipca_batch)

    for i in range(0, n, ipca_batch):
        ipca.partial_fit(V[i:i+ipca_batch])

    W = np.memmap(emb_out, dtype="float32", mode="w+", shape=(n, dim_out))
    for i in range(0, n, ipca_batch):
        chunk = ipca.transform(V[i:i+ipca_batch]).astype(np.float32)
        chunk = normalize(chunk, norm="l2")
        W[i:i+len(chunk)] = chunk
        W.flush()

    print("pca:", emb_out)

if __name__ == "__main__":
    main()