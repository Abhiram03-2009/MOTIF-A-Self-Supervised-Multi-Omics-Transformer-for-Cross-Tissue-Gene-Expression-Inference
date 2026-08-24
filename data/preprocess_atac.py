"""
data/preprocess_atac.py

Condenses real ATAC-seq chromatin accessibility data (ENCODE narrowPeak,
IDR-thresholded) for the liver/brain/lung panel into a per-gene profile,
replacing the fully synthetic ATAC values used elsewhere in the pipeline.

Source: ENCODE Project (encodeproject.org), ATAC-seq, GRCh38. See
data/download_data.py::ENCODE_URLS for the exact accessions. Unlike WGBS
methylation, ENCODE has released adult human brain-tissue ATAC-seq
(cerebellum), so no cell-line proxy is needed here.

Usage:
    python data/preprocess_atac.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.preprocess_methylation import GENE_PANEL, PROMOTER_WINDOW

RAW_ATAC_DIR = Path("data/raw/encode/atac")
OUT_PATH = Path("data/processed/real_atac_profile.csv")

TISSUE_FILES = {
    "liver": RAW_ATAC_DIR / "liver_atac.bed.gz",
    "brain": RAW_ATAC_DIR / "brain_atac.bed.gz",   # cerebellum tissue
    "lung":  RAW_ATAC_DIR / "lung_atac.bed.gz",
}

NARROWPEAK_COLS = ["chrom", "start", "end", "name", "score", "strand",
                   "signal_value", "p_value", "q_value", "summit"]


def promoter_windows() -> dict:
    """chrom -> list of (gene, window_start, window_end)."""
    windows = {}
    for gene, (_, chrom, tss, _, _) in GENE_PANEL.items():
        windows.setdefault(chrom, []).append(
            (gene, tss - PROMOTER_WINDOW, tss + PROMOTER_WINDOW)
        )
    return windows


def peaks_overlapping_promoters(bed_path: Path) -> pd.DataFrame:
    """Load a narrowPeak file and keep only peaks overlapping a promoter
    window from GENE_PANEL. Returns one row per matching peak, tagged with
    its gene. narrowPeak files are small enough (~1-4MB gzipped, ~150K rows)
    to load in one shot, unlike the genome-wide WGBS bedMethyl files."""
    df = pd.read_csv(bed_path, sep="\t", header=None, names=NARROWPEAK_COLS,
                      usecols=["chrom", "start", "end", "signal_value"],
                      compression="gzip")

    windows = promoter_windows()
    hits = []
    for chrom, group in df.groupby("chrom"):
        if chrom not in windows:
            continue
        for gene, w_start, w_end in windows[chrom]:
            # Peak overlaps the promoter window if it starts before the
            # window ends and ends after the window starts.
            match = group[(group["start"] < w_end) & (group["end"] > w_start)]
            if len(match):
                m = match.copy()
                m["gene"] = gene
                hits.append(m)

    if not hits:
        return pd.DataFrame(columns=["gene", "signal_value"])
    return pd.concat(hits, ignore_index=True)


def build_real_atac_profile():
    print("\n" + "=" * 60)
    print(" Preprocessing Real ATAC-seq Accessibility (ENCODE narrowPeak)")
    print("=" * 60)

    rows = []
    for tissue, bed_path in TISSUE_FILES.items():
        if not bed_path.exists():
            print(f"\n[{tissue}] SKIP: {bed_path} not found. "
                  f"Run data/download_data.py first.")
            continue

        print(f"\n[{tissue}] Scanning {bed_path.name} for "
              f"{len(GENE_PANEL)} gene promoters (TSS +/- {PROMOTER_WINDOW}bp)...")
        peaks = peaks_overlapping_promoters(bed_path)

        for gene, (ensembl_id, chrom, tss, strand, relevance) in GENE_PANEL.items():
            gene_peaks = peaks[peaks["gene"] == gene]
            if gene_peaks.empty:
                print(f"    {gene:10s} ({relevance:12s}) no peaks overlap this promoter")
                continue
            mean_signal = float(gene_peaks["signal_value"].mean())
            rows.append({
                "tissue": tissue,
                "gene": gene,
                "ensembl_id": ensembl_id,
                "tissue_relevance": relevance,
                "n_peaks": len(gene_peaks),
                "mean_signal_value": round(mean_signal, 3),
            })
            print(f"    {gene:10s} ({relevance:12s}) "
                  f"{len(gene_peaks):3d} peak(s) | mean signal = {mean_signal:7.3f}")

    if not rows:
        print("\nNo ATAC data processed. Run data/download_data.py first.")
        return

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved real per-gene promoter accessibility profile -> {OUT_PATH}")
    print(f"{len(df)} tissue/gene rows from {len(TISSUE_FILES)} real ENCODE ATAC-seq experiments.")


if __name__ == "__main__":
    build_real_atac_profile()
