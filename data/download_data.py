"""
data/download_data.py

Downloads public multi-omics datasets from GTEx and ENCODE.
Run this FIRST before any preprocessing.

Usage:
    python data/download_data.py

What this downloads:
    - GTEx v8 gene expression (RNA-seq) for liver, brain, lung
    - ENCODE ATAC-seq peaks for matched tissues
    - ENCODE methylation (optional)

If downloads fail, instructions are given for manual download.
"""

import os
import requests
import gzip
import shutil
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


# GTEx v8 median gene expression (TPM) per tissue
# Full portal: https://gtexportal.org/home/datasets
GTEX_URLS = {
    "gene_tpm": (
        "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/"
        "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
    ),
    "sample_attributes": (
        "https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/"
        "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
    ),
}

# Real, verified ENCODE accessions (bed narrowPeak, IDR-thresholded peaks,
# GRCh38) — checked against the ENCODE Portal REST API before being added
# here. Unlike methylation, ENCODE does have released adult human
# brain-tissue ATAC-seq (cerebellum), so no cell-line proxy is needed.
ENCODE_URLS = {
    "liver_atac": (
        # ENCSR516CPW - right lobe of liver, female adult (47y), GRCh38
        "https://www.encodeproject.org/files/ENCFF912MCC/@@download/ENCFF912MCC.bed.gz"
    ),
    "brain_atac": (
        # ENCSR802GEV - cerebellum tissue, male adult (20y), GRCh38
        "https://www.encodeproject.org/files/ENCFF858CCP/@@download/ENCFF858CCP.bed.gz"
    ),
    "lung_atac": (
        # ENCSR647AOY - lung tissue, female adult (47y), GRCh38
        "https://www.encodeproject.org/files/ENCFF068ZZZ/@@download/ENCFF068ZZZ.bed.gz"
    ),
}


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> bool:
    """Download a file with a progress bar. Returns True on success."""
    print(f"\n  Downloading: {dest.name}")
    print(f"  URL: {url[:80]}...")
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  Progress: {pct:.1f}%", end="", flush=True)
        print(f"\r  Done: {dest.name}              ")
        return True
    except Exception as e:
        print(f"\r  FAILED: {e}")
        return False


def decompress_gz(src: Path):
    """Decompress a .gz file and remove the archive."""
    dest = src.with_suffix("")
    if dest.exists():
        print(f"  Already decompressed: {dest.name}")
        return dest
    print(f"  Decompressing: {src.name}")
    with gzip.open(src, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    src.unlink()
    return dest


def download_gtex():
    print("\n=== Downloading GTEx RNA-seq ===")
    gtex_dir = RAW_DIR / "gtex"
    gtex_dir.mkdir(exist_ok=True)

    for name, url in GTEX_URLS.items():
        filename = url.split("/")[-1]
        dest = gtex_dir / filename
        if dest.exists() or dest.with_suffix("").exists():
            print(f"  Already exists: {filename}")
            continue
        ok = download_file(url, dest)
        if ok and dest.suffix == ".gz":
            decompress_gz(dest)

    print("\nGTEx download complete.")
    print("Files saved to:", gtex_dir)
    print()
    print("NOTE: The full GTEx TPM matrix is ~2GB.")
    print("If download fails, get it manually from:")
    print("  https://gtexportal.org/home/datasets")
    print("  → GTEx Analysis V8 → 'Gene TPM' file")


def download_encode():
    print("\n=== Downloading ENCODE ATAC-seq (narrowPeak, IDR-thresholded) ===")
    atac_dir = RAW_DIR / "encode" / "atac"
    atac_dir.mkdir(parents=True, exist_ok=True)

    for name, url in ENCODE_URLS.items():
        dest = atac_dir / f"{name}.bed.gz"
        if dest.exists():
            print(f"  Already exists: {dest.name}")
            continue
        # Left gzipped — data/preprocess_atac.py reads the .gz directly.
        download_file(url, dest)

    print("\nENCODE download complete.")
    print("Files saved to:", atac_dir)
    print("Next: python data/preprocess_atac.py")


# Real, verified ENCODE accessions (bed bedMethyl, "methylation state at CpG",
# GRCh38) — checked against the ENCODE Portal REST API before being added here.
# Liver and lung are true adult tissue WGBS experiments. ENCODE currently has
# no *released* adult human brain-tissue WGBS with processed CpG calls (the
# tissue experiments that exist, e.g. cerebellum/hippocampus, are archived and
# only expose raw reads) — SK-N-SH (neuroblastoma, ENCODE's standard
# neuronal-lineage reference line) is used as the brain-lineage proxy instead.
# This limitation is called out again in preprocess_methylation.py.
METHYLATION_URLS = {
    "liver_wgbs_cpg": (
        # ENCSR108ESU - liver tissue, female adult (53y), GRCh38
        "https://www.encodeproject.org/files/ENCFF577VGR/@@download/ENCFF577VGR.bed.gz"
    ),
    "brain_wgbs_cpg": (
        # ENCSR145HNT - SK-N-SH neuroblastoma (brain-lineage proxy), GRCh38
        "https://www.encodeproject.org/files/ENCFF940XWW/@@download/ENCFF940XWW.bed.gz"
    ),
    "lung_wgbs_cpg": (
        # ENCSR556KEJ - lung tissue, female adult (30y), GRCh38
        "https://www.encodeproject.org/files/ENCFF039JFT/@@download/ENCFF039JFT.bed.gz"
    ),
}


def download_methylation():
    print("\n=== Downloading ENCODE WGBS DNA methylation (bedMethyl, CpG) ===")
    meth_dir = RAW_DIR / "encode" / "methylation"
    meth_dir.mkdir(parents=True, exist_ok=True)

    for name, url in METHYLATION_URLS.items():
        dest = meth_dir / f"{name}.bed.gz"
        if dest.exists():
            print(f"  Already exists: {dest.name}")
            continue
        # Left gzipped — data/preprocess_methylation.py streams the .gz
        # directly instead of decompressing (~18M CpG rows per file).
        download_file(url, dest)

    print("\nMethylation download complete.")
    print("Files saved to:", meth_dir)
    print("Next: python data/preprocess_methylation.py")


def print_manual_instructions():
    print("""
=================================================================
 MANUAL DOWNLOAD INSTRUCTIONS (if automated download fails)
=================================================================

1. GTEx RNA-seq (gene expression matrix):
   → Go to: https://gtexportal.org/home/datasets
   → Download: GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz
   → Download: GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt
   → Place in: data/raw/gtex/

2. ENCODE ATAC-seq peaks (narrowPeak, IDR-thresholded) — see ENCODE_URLS above:
   → ENCFF912MCC (liver), ENCFF858CCP (brain/cerebellum), ENCFF068ZZZ (lung)
   → https://www.encodeproject.org/files/<accession>/@@download/<accession>.bed.gz
   → Place in: data/raw/encode/atac/ (name them liver_atac.bed.gz,
     brain_atac.bed.gz, lung_atac.bed.gz)

3. DNA Methylation (WGBS bedMethyl, CpG) — see METHYLATION_URLS above for the
   exact accessions used. If automated download fails:
   → ENCFF577VGR (liver), ENCFF940XWW (brain-lineage/SK-N-SH), ENCFF039JFT (lung)
   → https://www.encodeproject.org/files/<accession>/@@download/<accession>.bed.gz
   → Place in: data/raw/encode/methylation/ (name them liver_wgbs_cpg.bed.gz,
     brain_wgbs_cpg.bed.gz, lung_wgbs_cpg.bed.gz)

After manual download, run:
   python data/preprocess_atac.py
   python data/preprocess_methylation.py
   python data/preprocess.py

=================================================================
""")


if __name__ == "__main__":
    print("=" * 60)
    print(" MOTIF — Data Downloader")
    print("=" * 60)

    download_gtex()
    download_encode()
    download_methylation()
    print_manual_instructions()

    print("\nAll done. Next step:")
    print("  python data/preprocess_atac.py")
    print("  python data/preprocess_methylation.py")
    print("  python data/preprocess.py")
