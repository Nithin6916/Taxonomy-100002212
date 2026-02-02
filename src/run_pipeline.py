import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run(cmd: str):
    print("\n>>", cmd)
    r = subprocess.run(cmd, shell=True, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(r.returncode)

def exists(p: str) -> bool:
    return (ROOT / p).exists()

def main():
    # Minimal required outputs for "fast run" using existing embeddings
    need_pca = "data/processed/emb_pca256.f32.mmap"
    need_micro = "data/processed/micro_labels.npy"
    need_final = "data/processed/final_cluster_labels.npy"
    need_merged = "data/processed/final_cluster_labels_merged.npy"

    # If embeddings exist but PCA doesn't, run PCA onward
    if not exists(need_pca):
        run("python -m src.steps.04_pca")

    if not exists(need_micro):
        run("python -m src.steps.05_microcluster_birch")

    if not exists(need_final):
        run("python -m src.steps.06_leiden_on_centroids")

    # Evaluate fine clustering (purity optimized)
    run("python -m src.steps.07_evaluate")

    # If merged labels already exist, keep them; otherwise run your merge step
    # (Assumes you already have src.steps.08_merge_clusters implemented)
    if not exists(need_merged):
        run("python -m src.steps.08_merge_clusters")

    # Finalize outputs + plots (your 09 file generates everything)
    run("python -m src.steps.09_finalize_variants")

    print("\nDONE")
    print("Check outputs/final/ for:")
    print("- clusters_purity_optimized.npy / clusters_balanced.npy")
    print("- metrics_comparison.csv")
    print("- plots (png) + top labels CSVs")

if __name__ == "__main__":
    main()