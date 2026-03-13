import json
import numpy as np
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

st.set_page_config(page_title="Icecat Taxonomy Clustering Demo", layout="wide")


@st.cache_resource
def load_embedder():
    model = SentenceTransformer("BAAI/bge-m3")
    model.max_seq_length = 256
    return model


@st.cache_resource
def load_artifacts():
    mean = np.load("artifacts/pca_mean.npy").astype(np.float32)
    comps = np.load("artifacts/pca_components.npy").astype(np.float32)

    cent_bal = np.load("artifacts/centroids_balanced.npy").astype(np.float32)
    cent_pur = np.load("artifacts/centroids_purity.npy").astype(np.float32)

    ids_bal = np.load("artifacts/cluster_ids_balanced.npy").astype(np.int32)
    ids_pur = np.load("artifacts/cluster_ids_purity.npy").astype(np.int32)

    with open("artifacts/cluster_meta_balanced.json", "r", encoding="utf-8") as f:
        meta_bal = json.load(f)
    with open("artifacts/cluster_meta_purity.json", "r", encoding="utf-8") as f:
        meta_pur = json.load(f)

    return mean, comps, cent_bal, cent_pur, ids_bal, ids_pur, meta_bal, meta_pur


def clean_text(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).replace("\u00ae", " ").replace("\u2122", " ")
    s = " ".join(s.split())
    if len(s) > 1200:
        s = s[:1200]
    return s


def extract_spec_tokens(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    out = []
    import re

    out += [f"gb:{m}" for m in re.findall(r"\b(\d{2,4})\s*gb\b", t)]
    out += [f"mah:{m}" for m in re.findall(r"\b(\d{3,5})\s*mah\b", t)]
    out += [f"hz:{m}" for m in re.findall(r"\b(\d{2,4})\s*hz\b", t)]
    out += [f"w:{m}" for m in re.findall(r"\b(\d{2,4})\s*w\b", t)]
    out += [f"in:{m}" for m in re.findall(r"\b(\d{1,2}\.?\d?)\s*(?:inch|inches|in|\" )\b", t)]
    out += [f"res:{a}x{b}" for a, b in re.findall(r"\b(\d{3,4})\s*x\s*(\d{3,4})\b", t)]

    out = out[:50]
    return " ".join(out)


def build_doc(row: pd.Series, title_col: str, brand_col: str, desc_col: str) -> str:
    parts = []
    if title_col:
        v = clean_text(row.get(title_col, ""))
        if v:
            parts.append(f"title: {v}")
    if brand_col:
        v = clean_text(row.get(brand_col, ""))
        if v:
            parts.append(f"brand: {v}")
    if desc_col:
        v = clean_text(row.get(desc_col, ""))
        if v:
            parts.append(f"description: {v}")

    doc = "\n".join(parts)
    spec = extract_spec_tokens(doc)
    if spec:
        doc = doc + "\n" + "spec_tokens: " + spec
    return doc


def pca_project(X: np.ndarray, mean: np.ndarray, comps: np.ndarray) -> np.ndarray:
    Z = (X - mean) @ comps.T
    Z = normalize(Z, norm="l2").astype(np.float32)
    return Z


def cosine_sim_matrix(Z: np.ndarray, C: np.ndarray) -> np.ndarray:
    # Z and C are L2-normalized => cosine sim = dot product
    return Z @ C.T


def cluster_display_name(meta: dict, cluster_id: int) -> str:
    m = meta.get(str(int(cluster_id)), {})
    kws = m.get("keywords", [])
    if kws:
        return ", ".join(kws[:6])
    return f"Cluster {cluster_id}"


def topk_assignments(sim_row: np.ndarray, k: int = 3):
    # returns indices and scores for top-k
    idx = np.argpartition(-sim_row, kth=min(k, len(sim_row)-1))[:k]
    idx = idx[np.argsort(-sim_row[idx])]
    return idx, sim_row[idx]


def main():
    st.title("Icecat Taxonomy Clustering Demo")
    st.write("Upload 100–300 products. The app assigns each product to learned clusters (balanced or purity-optimized).")

    variant = st.radio("Choose clustering variant", ["balanced", "purity_optimized"], horizontal=True)
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    if uploaded is None:
        st.stop()

    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    st.write("Preview:")
    st.dataframe(df.head(10), use_container_width=True)

    if len(df) < 1:
        st.error("File has no rows.")
        st.stop()
    if len(df) > 300:
        st.warning(f"You uploaded {len(df)} rows. The demo is designed for 100–300. We'll use the first 300 rows.")
        df = df.head(300).copy()

    cols = list(df.columns)

    def pick(defaults):
        for d in defaults:
            if d in cols:
                return d
        return ""

    with st.expander("Column mapping (adjust if needed)", expanded=True):
        title_col = st.selectbox(
            "Title column",
            [""] + cols,
            index=([""] + cols).index(pick(["Title", "title", "ProductName", "product_name"]))
        )
        brand_col = st.selectbox(
            "Brand column",
            [""] + cols,
            index=([""] + cols).index(pick(["Brand", "brand"]))
        )
        desc_col = st.selectbox(
            "Description column",
            [""] + cols,
            index=([""] + cols).index(pick(["Description", "description", "Summary", "summary", "LongDesc", "long_description"]))
        )
        st.caption("At least Title is recommended. Brand/Description improve assignment quality.")

    if not title_col:
        st.error("Please select a Title column.")
        st.stop()

    mean, comps, cent_bal, cent_pur, ids_bal, ids_pur, meta_bal, meta_pur = load_artifacts()
    model = load_embedder()

    if variant == "balanced":
        C = cent_bal
        ids = ids_bal
        meta = meta_bal
    else:
        C = cent_pur
        ids = ids_pur
        meta = meta_pur

    show_topk = st.checkbox("Show top-3 suggested clusters (for ambiguity explanation)", value=True)

    if st.button("Run clustering"):
        docs = [build_doc(df.iloc[i], title_col, brand_col, desc_col) for i in range(len(df))]

        with st.spinner("Embedding..."):
            X = model.encode(
                docs,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False
            ).astype(np.float32)

        with st.spinner("Projecting (PCA) + assigning to clusters..."):
            Z = pca_project(X, mean, comps)
            sims = cosine_sim_matrix(Z, C)

            # centroid index -> original cluster id
            best_idx = np.argmax(sims, axis=1).astype(np.int32)
            cluster_ids = ids[best_idx].astype(np.int32)

            # confidence = top1 - top2 similarity
            part = np.partition(sims, kth=-2, axis=1)
            top2 = part[:, -2:]
            top1 = np.max(top2, axis=1)
            top2v = np.min(top2, axis=1)
            confidence = (top1 - top2v).astype(np.float32)

        out = df.copy()
        out["cluster_id"] = cluster_ids
        out["cluster_name"] = [cluster_display_name(meta, int(c)) for c in cluster_ids]
        out["confidence"] = confidence

        if show_topk:
            topk_names = []
            topk_scores = []
            for i in range(sims.shape[0]):
                idxs, scs = topk_assignments(sims[i], k=3)
                cids = ids[idxs]
                names = [cluster_display_name(meta, int(c)) for c in cids]
                topk_names.append(" | ".join(names))
                topk_scores.append(" | ".join([f"{float(s):.3f}" for s in scs]))
            out["top3_cluster_names"] = topk_names
            out["top3_similarities"] = topk_scores

        st.subheader("Cluster summary")
        summary = (
            out.groupby(["cluster_id", "cluster_name"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        st.dataframe(summary, use_container_width=True)

        st.subheader("Grouped results")
        for _, row in summary.head(20).iterrows():
            cid = int(row["cluster_id"])
            cname = row["cluster_name"]
            cnt = int(row["count"])

            with st.expander(f"[{cid}] {cname} — {cnt} items", expanded=False):
                m = meta.get(str(cid), {})
                if m.get("keywords"):
                    st.caption("Keywords: " + ", ".join(m["keywords"][:12]))
                if m.get("examples"):
                    st.caption("Examples: " + " | ".join(m["examples"][:3]))

                st.dataframe(out[out["cluster_id"] == cid], use_container_width=True)

        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download annotated CSV",
            data=csv_bytes,
            file_name=f"clustered_{variant}.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()