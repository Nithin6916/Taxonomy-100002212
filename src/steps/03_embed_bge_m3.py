import os, json, glob
import numpy as np
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from src.config import load_config

def main():
    cfg = load_config("configs/config.yaml")
    emb_path = cfg.paths["emb_mmap"]
    dim = int(cfg.paths["emb_dim"])
    ids_path = cfg.paths["product_ids_npy"]
    params = cfg.params

    docs_dir = os.path.join("data", "interim", "docs_shards")
    files = sorted(glob.glob(os.path.join(docs_dir, "docs_*.parquet")))
    if not files:
        raise RuntimeError(f"No docs shards found in {docs_dir}. Run step 02 first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA not detected.")

    model = SentenceTransformer(params["model_name"], device=device)
    model.max_seq_length = int(params["max_seq_length"])

    total_rows = 0
    for f in files:
        total_rows += pq.ParquetFile(f).metadata.num_rows

    os.makedirs(os.path.dirname(emb_path), exist_ok=True)
    os.makedirs(os.path.dirname(ids_path), exist_ok=True)

    prog_path = emb_path + ".progress.json"
    start_row = 0
    if os.path.exists(prog_path):
        start_row = int(json.load(open(prog_path, "r", encoding="utf-8")).get("next_row", 0))

    if os.path.exists(emb_path):
        emb = np.memmap(emb_path, dtype="float32", mode="r+", shape=(total_rows, dim))
    else:
        emb = np.memmap(emb_path, dtype="float32", mode="w+", shape=(total_rows, dim))

    if not os.path.exists(ids_path):
        np.save(ids_path, np.empty((total_rows,), dtype=np.int64))
    product_ids = np.load(ids_path, mmap_mode="r+")
    product_ids = product_ids.view(np.int64)

    row_cursor = 0
    for f in tqdm(files, desc="embed_files"):
        t = pq.read_table(f, columns=["product_id", "doc"])
        pdf = t.to_pandas()
        bsz = len(pdf)

        if row_cursor + bsz <= start_row:
            row_cursor += bsz
            continue

        docs = pdf["doc"].astype(str).tolist()
        ids = pdf["product_id"].to_numpy(dtype=np.int64)

        vecs = model.encode(
            docs,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype(np.float32)

        w0, w1 = row_cursor, row_cursor + bsz
        emb[w0:w1] = vecs
        emb.flush()

        product_ids[w0:w1] = ids
        product_ids.flush()

        row_cursor += bsz
        json.dump({"next_row": row_cursor}, open(prog_path, "w", encoding="utf-8"))

    print("embeddings:", emb_path)
    print("product_ids:", ids_path)

if __name__ == "__main__":
    main()