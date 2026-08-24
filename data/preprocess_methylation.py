"""
data/preprocess_methylation.py

Condenses real DNA methylation data (ENCODE WGBS bedMethyl, CpG calls) for
the liver/brain/lung panel into a per-gene profile, replacing the fully
synthetic methylation values used elsewhere in the pipeline.

Source: ENCODE Project (encodeproject.org), whole-genome bisulfite
sequencing, GRCh38. See data/download_data.py::METHYLATION_URLS for the
exact accessions.

Note on brain: ENCODE has no *released* adult human brain-tissue WGBS with
processed CpG calls (only archived raw reads), so SK-N-SH (neuroblastoma,
ENCODE's standard neuronal-lineage reference line) stands in as the
brain-lineage sample. Flagged here and wherever this data is consumed.

Usage:
    python data/preprocess_methylation.py
"""

from pathlib import Path

import pandas as pd

RAW_METH_DIR = Path("data/raw/encode/methylation")
OUT_PATH = Path("data/processed/real_methylation_profile.csv")

PROMOTER_WINDOW = 2000  # +/- bp around TSS

# Real GRCh38 coordinates (Ensembl REST API, homo_sapiens, GRCh38, fetched
# 2026-06-29). tss = start if strand == "+" else end.
GENE_PANEL = {
    # gene: (ensembl_id, chrom, tss, strand, tissue_relevance)
    "ALB":      ("ENSG00000163631", "chr4",  73397114, "+", "liver"),
    "APOB":     ("ENSG00000084674", "chr2",  21044075, "-", "liver"),
    "CYP3A4":   ("ENSG00000160868", "chr7",  99784327, "-", "liver"),
    "SERPINA1": ("ENSG00000197249", "chr14", 94391024, "-", "liver"),
    "TTR":      ("ENSG00000118271", "chr18", 31557008, "+", "liver"),
    "GFAP":     ("ENSG00000131095", "chr17", 44916937, "-", "brain"),
    "SYP":      ("ENSG00000102003", "chrX",  49200218, "-", "brain"),
    "RBFOX3":   ("ENSG00000167281", "chr17", 79611317, "-", "brain"),
    "MBP":      ("ENSG00000197971", "chr18", 77133683, "-", "brain"),
    "SNAP25":   ("ENSG00000132639", "chr20", 10172395, "+", "brain"),
    "SFTPC":    ("ENSG00000168484", "chr8",  22156913, "+", "lung"),
    "SFTPB":    ("ENSG00000168878", "chr2",  85668741, "-", "lung"),
    "SCGB1A1":  ("ENSG00000149021", "chr11", 62405103, "+", "lung"),
    "NAPSA":    ("ENSG00000131400", "chr19", 50369342, "-", "lung"),
    "SFTPA1":   ("ENSG00000122852", "chr10", 79559387, "+", "lung"),
    "GAPDH":    ("ENSG00000111640", "chr12", 6534012,  "+", "housekeeping"),
    "ACTB":     ("ENSG00000075624", "chr7",  5563902,  "-", "housekeeping"),
    "TBP":      ("ENSG00000112592", "chr6",  170554266, "+", "housekeeping"),
}

# SK-N-SH stands in for "brain" — see module docstring.
TISSUE_FILES = {
    "liver": RAW_METH_DIR / "liver_wgbs_cpg.bed.gz",
    "brain": RAW_METH_DIR / "brain_wgbs_cpg.bed.gz",
    "lung":  RAW_METH_DIR / "lung_wgbs_cpg.bed.gz",
}

BED_COLS = ["chrom", "start", "end", "name", "score", "strand",
            "thick_start", "thick_end", "rgb", "coverage", "pct_meth"]


def promoter_windows() -> dict:
    """chrom -> list of (gene, window_start, window_end)."""
    windows = {}
    for gene, (_, chrom, tss, _, _) in GENE_PANEL.items():
        windows.setdefault(chrom, []).append(
            (gene, tss - PROMOTER_WINDOW, tss + PROMOTER_WINDOW)
        )
    return windows


def extract_promoter_cpgs(bed_path: Path, chunksize: int = 1_000_000) -> pd.DataFrame:
    """
    Stream a genome-wide bedMethyl file and keep only CpGs that fall inside
    a promoter window from GENE_PANEL. Returns one row per matching CpG,
    tagged with its gene.
    """
    windows = promoter_windows()
    relevant_chroms = set(windows.keys())
    hits = []

    reader = pd.read_csv(
        bed_path, sep="\t", header=None, names=BED_COLS,
        usecols=["chrom", "start", "coverage", "pct_meth"],
        chunksize=chunksize, compression="gzip",
    )
    for chunk in reader:
        chunk = chunk[chunk["chrom"].isin(relevant_chroms)]
        if chunk.empty:
            continue
        for chrom, group in chunk.groupby("chrom"):
            for gene, w_start, w_end in windows[chrom]:
                match = group[(group["start"] >= w_start) & (group["start"] < w_end)]
                if len(match):
                    m = match.copy()
                    m["gene"] = gene
                    hits.append(m)

    if not hits:
        return pd.DataFrame(columns=["gene", "start", "coverage", "pct_meth"])
    return pd.concat(hits, ignore_index=True)


def build_real_methylation_profile():
    print("\n" + "=" * 60)
    print(" Preprocessing Real DNA Methylation (ENCODE WGBS)")
    print("=" * 60)

    rows = []
    for tissue, bed_path in TISSUE_FILES.items():
        if not bed_path.exists():
            print(f"\n[{tissue}] SKIP: {bed_path} not found. "
                  f"Run data/download_data.py first.")
            continue

        print(f"\n[{tissue}] Scanning {bed_path.name} for "
              f"{len(GENE_PANEL)} gene promoters (TSS +/- {PROMOTER_WINDOW}bp)...")
        cpgs = extract_promoter_cpgs(bed_path)

        for gene, (ensembl_id, chrom, tss, strand, relevance) in GENE_PANEL.items():
            gene_cpgs = cpgs[cpgs["gene"] == gene]
            if gene_cpgs.empty:
                print(f"    {gene:10s} ({relevance:12s}) no CpGs covered in this file")
                continue
            weights = gene_cpgs["coverage"].clip(lower=1)
            weighted_mean = (gene_cpgs["pct_meth"] * weights).sum() / weights.sum()
            rows.append({
                "tissue": tissue,
                "gene": gene,
                "ensembl_id": ensembl_id,
                "tissue_relevance": relevance,
                "n_cpgs": len(gene_cpgs),
                "mean_coverage": round(float(weights.mean()), 2),
                "mean_percent_methylation": round(float(weighted_mean), 2),
            })
            print(f"    {gene:10s} ({relevance:12s}) "
                  f"{len(gene_cpgs):4d} CpGs | mean methylation = {weighted_mean:5.1f}%")

    if not rows:
        print("\nNo methylation data processed. Run data/download_data.py first.")
        return

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved real per-gene promoter methylation profile -> {OUT_PATH}")
    print(f"{len(df)} tissue/gene rows from {len(TISSUE_FILES)} real ENCODE WGBS experiments.")


if __name__ == "__main__":
    build_real_methylation_profile()
