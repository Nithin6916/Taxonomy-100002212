import os
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from src.config import load_config
from src.utils.text import row_to_doc

def main():
    cfg = load_config("configs/config.yaml")
    feat_in = cfg.paths["features_parquet"]

    docs_dir = os.path.join("data", "interim", "docs_shards")
    os.makedirs(docs_dir, exist_ok=True)

    pf = pq.ParquetFile(feat_in)
    total = pf.metadata.num_rows

    shard_id = 0
    batch_size = 4096

    for batch in tqdm(pf.iter_batches(batch_size=batch_size), total=(total // batch_size + 1), desc="docs"):
        tbl = pa.Table.from_batches([batch])
        pdf = tbl.to_pandas()

        product_ids = pdf["product_id"].tolist()
        cols = [c for c in pdf.columns if c != "product_id"]

        docs = []
        for i in range(len(pdf)):
            row = {c: pdf.at[i, c] for c in cols}
            docs.append(row_to_doc(row))

        out_tbl = pa.Table.from_pydict({"product_id": product_ids, "doc": docs})
        out_path = os.path.join(docs_dir, f"docs_{shard_id:05d}.parquet")
        pq.write_table(out_tbl, out_path, compression="snappy")

        shard_id += 1

    print("docs_dir:", docs_dir)

if __name__ == "__main__":
    main()