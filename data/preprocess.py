"""
data/preprocess.py

Preprocesses raw GTEx + ENCODE data into clean tensors ready for training.

Usage:
    python data/preprocess.py
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.configs.config import Config

cfg = Config()

RAW_DIR       = Path(cfg.data.RAW_DIR)
PROCESSED_DIR = Path(cfg.data.PROCESSED_DIR)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TISSUES    = cfg.data.TISSUES
N_GENES    = cfg.data.N_GENES
N_PEAKS    = cfg.data.N_ATAC_PEAKS
SEED       = cfg.data.RANDOM_SEED

# Real, ENCODE-derived per-tissue profiles (see preprocess_atac.py and
# preprocess_methylation.py). Optional — code falls back to purely
# synthetic values if these haven't been generated yet.
REAL_ATAC_PROFILE        = Path("data/processed/real_atac_profile.csv")
REAL_METHYLATION_PROFILE = Path("data/processed/real_methylation_profile.csv")


def gtex_header_columns(gct_path: Path) -> list:
    """Read just the GCT column-name row (cheap) without loading any data,
    so callers can figure out which sample columns they actually need
    before touching the multi-gigabyte body of the file."""
    header = pd.read_csv(gct_path, sep="\t", skiprows=2, index_col=0, nrows=0)
    return list(header.columns)


def parse_gtex_tpm(gct_path: Path, sample_ids: list = None) -> pd.DataFrame:
    """
    Parse GTEx .gct file into a DataFrame:
        rows = genes (ENSEMBL IDs)
        cols = sample IDs

    The full matrix is ~56,200 genes x ~17,400 samples — loading every
    column needs ~7GB+ as float64 and can exhaust memory on a laptop. Since
    we only ever use 3 tissues, pass sample_ids (from the metadata file, see
    filter_gtex_by_tissue) to load only the relevant columns via usecols.
    """
    print("  Parsing GTEx TPM matrix (this may take a minute)...")
    usecols = None
    dtype = None
    if sample_ids is not None:
        available = set(gtex_header_columns(gct_path))
        sample_cols = [c for c in sample_ids if c in available]
        gene_id_col = pd.read_csv(gct_path, sep="\t", skiprows=2, nrows=0).columns[0]
        usecols = [gene_id_col] + sample_cols
        dtype = {c: np.float32 for c in sample_cols}
        print(f"  Restricting to {len(sample_cols)} relevant sample columns "
              f"(out of ~17,400 total) to keep memory usage manageable.")

    df = pd.read_csv(
        gct_path,
        sep="\t",
        skiprows=2,
        index_col=0,
        usecols=usecols,
        dtype=dtype,
    )
    if "Description" in df.columns:
        df = df.drop(columns=["Description"])
    print(f"  GTEx shape: {df.shape} (genes x samples)")
    return df


def parse_gtex_metadata(meta_path: Path) -> pd.DataFrame:
    """
    Parse GTEx sample attributes to get tissue labels.
    Returns: DataFrame with columns [SAMPID, SMTSD (tissue name)]
    """
    meta = pd.read_csv(meta_path, sep="\t", usecols=["SAMPID", "SMTSD"])
    return meta


GTEX_TISSUE_MAP = {
    "liver": "Liver",
    "brain": "Brain - Cortex",
    "lung":  "Lung",
}


def gtex_sample_ids_for_tissues(meta: pd.DataFrame, tissues: list) -> list:
    """Flat list of GTEx sample IDs belonging to any of `tissues` — used to
    restrict parse_gtex_tpm() to only the columns we'll actually use."""
    labels = {GTEX_TISSUE_MAP.get(t, t) for t in tissues}
    return meta.loc[meta["SMTSD"].isin(labels), "SAMPID"].tolist()


def filter_gtex_by_tissue(tpm: pd.DataFrame, meta: pd.DataFrame,
                           tissues: list) -> dict:
    """
    Filter GTEx samples by tissue.
    Returns: dict { tissue_name: expression_DataFrame }
    """
    tissue_map = GTEX_TISSUE_MAP
    result = {}
    meta_indexed = meta.set_index("SAMPID")

    for tissue_key in tissues:
        tissue_label = tissue_map.get(tissue_key, tissue_key)
        samples = meta_indexed[meta_indexed["SMTSD"] == tissue_label].index
        available = [s for s in samples if s in tpm.columns]
        if len(available) == 0:
            print(f"  WARNING: No samples found for tissue '{tissue_label}'")
            continue
        sub = tpm[available]
        result[tissue_key] = sub
        print(f"  {tissue_key}: {sub.shape[1]} samples, {sub.shape[0]} genes")

    return result


def select_highly_variable_genes(tissue_dict: dict, n_genes: int) -> list:
    """
    Select top N most variable genes across all tissues combined.
    """
    print(f"  Selecting top {n_genes} most variable genes...")
    all_data = pd.concat(list(tissue_dict.values()), axis=1)
    variance = all_data.var(axis=1)
    top_genes = variance.nlargest(n_genes).index.tolist()
    return top_genes


def build_atac_proxy(rna_matrix: np.ndarray, n_peaks: int,
                     tissue_label: int, seed: int = 42) -> np.ndarray:
    """
    Build a synthetic ATAC proxy correlated with expression but with noise.
    Real per-sample ATAC quantification isn't available for these donors —
    see real_tissue_effect() for how real ENCODE signal still gets folded in.

    Args:
        rna_matrix: (n_samples, n_genes) normalized RNA
        n_peaks:    number of ATAC peaks
        tissue_label: int (0=liver, 1=brain, 2=lung)
        seed:       random seed

    Returns:
        atac: (n_samples, n_peaks) ATAC-like matrix
    """
    rng = np.random.RandomState(seed + tissue_label)
    n_samples = rna_matrix.shape[0]

    n_gene_linked = min(n_peaks, rna_matrix.shape[1])
    # Peak i maps to gene i in every tissue, not a random gene per tissue, so a
    # peak stays near the same gene across tissues like a real one would.
    rng.choice(rna_matrix.shape[1], size=n_gene_linked, replace=False)  # advances rng to match prior noise draws
    gene_idx = np.arange(n_gene_linked)
    gene_signal = rna_matrix[:, gene_idx]

    tissue_bias = rng.randn(n_peaks) * 0.3
    noise = rng.randn(n_samples, n_peaks) * 0.5

    if n_gene_linked < n_peaks:
        extra = rng.randn(n_samples, n_peaks - n_gene_linked) * 0.8
        atac = np.concatenate([gene_signal + noise[:, :n_gene_linked], extra], axis=1)
    else:
        atac = gene_signal + noise

    atac = atac + tissue_bias
    return atac.astype(np.float32)


def build_methylation_proxy(rna_matrix: np.ndarray, n_sites: int,
                             tissue_label: int, seed: int = 42) -> np.ndarray:
    """
    Build a synthetic methylation proxy (anti-correlated with expression).
    High methylation → gene silenced → low expression (known biology).

    True per-sample WGBS matched to these GTEx donors isn't publicly
    available, so per-sample variation here is still modeled. Real,
    ENCODE-WGBS-derived cross-tissue differences are added back in
    separately by real_tissue_effect() *after* per-tissue normalization —
    see the note there for why it can't be done inside this function.
    """
    rng = np.random.RandomState(seed + 100 + tissue_label)
    n_samples = rna_matrix.shape[0]

    n_gene_linked = min(n_sites, rna_matrix.shape[1])
    # Site i maps to gene i, matching the ATAC peak at the same position (see build_atac_proxy).
    rng.choice(rna_matrix.shape[1], size=n_gene_linked, replace=False)  # advances rng to match prior noise draws
    gene_idx = np.arange(n_gene_linked)
    gene_signal = rna_matrix[:, gene_idx]

    meth_signal = -gene_signal + rng.randn(n_samples, n_gene_linked) * 0.4

    if n_gene_linked < n_sites:
        extra = rng.randn(n_samples, n_sites - n_gene_linked) * 0.5
        meth = np.concatenate([meth_signal, extra], axis=1)
    else:
        meth = meth_signal

    meth = (meth - meth.min()) / (meth.max() - meth.min() + 1e-8)
    return meth.astype(np.float32)


def real_tissue_effect(tissue_name: str, profile_path: Path, value_col: str) -> float:
    """
    Real, standardized cross-tissue effect size derived from measured ENCODE
    data (WGBS methylation % or ATAC signal), for one tissue.

    Why this has to be added *after* normalize_features() rather than mixed
    into build_methylation_proxy/build_atac_proxy: those matrices get
    z-scored per tissue block (a fresh StandardScaler fit on each tissue's
    samples alone, before the tissue blocks are concatenated). Any constant
    shift added before that z-scoring is mean-subtracted straight back out,
    so it never reaches the saved .npy files — it would look like it's
    doing something without actually doing anything. What DOES survive
    concatenation is a per-tissue offset applied after normalization, so
    that's what this computes: each tissue's real measured value expressed
    relative to the cross-tissue mean/spread (a standardized effect size),
    added back onto the already-normalized block.
    """
    if not profile_path.exists():
        return 0.0
    df = pd.read_csv(profile_path)
    means = df.groupby("tissue")[value_col].mean()
    if tissue_name not in means.index or len(means) < 2 or means.std() == 0:
        return 0.0
    return float((means[tissue_name] - means.mean()) / means.std())


def normalize_rna(matrix: np.ndarray) -> np.ndarray:
    """log1p transform then z-score."""
    log_transformed = np.log1p(matrix)
    scaler = StandardScaler()
    return scaler.fit_transform(log_transformed).astype(np.float32)


def normalize_features(matrix: np.ndarray) -> np.ndarray:
    """Z-score normalize."""
    scaler = StandardScaler()
    return scaler.fit_transform(matrix).astype(np.float32)


def build_dataset():
    """
    Full pipeline: parse → filter → select genes → normalize → save.
    """
    print("\n" + "=" * 60)
    print(" Building Multi-Omics Dataset")
    print("=" * 60)

    gtex_dir = RAW_DIR / "gtex"
    gct_files = list(gtex_dir.glob("*.gct"))
    meta_files = list(gtex_dir.glob("*SampleAttributes*"))

    if not gct_files or not meta_files:
        print("\nWARNING: GTEx files not found in data/raw/gtex/")
        print("Creating synthetic demo dataset for development...")
        build_synthetic_demo()
        return

    gct_path  = gct_files[0]
    meta_path = meta_files[0]

    print("\n[1/5] Parsing GTEx expression matrix...")
    meta = parse_gtex_metadata(meta_path)
    relevant_samples = gtex_sample_ids_for_tissues(meta, TISSUES)
    tpm = parse_gtex_tpm(gct_path, sample_ids=relevant_samples)

    print("\n[2/5] Filtering by tissue...")
    tissue_dict = filter_gtex_by_tissue(tpm, meta, TISSUES)

    print("\n[3/5] Selecting highly variable genes...")
    top_genes = select_highly_variable_genes(tissue_dict, N_GENES)

    print("\n[4/5] Building multi-omics matrices per tissue...")
    tissue_label_map = {t: i for i, t in enumerate(TISSUES)}

    all_atac, all_rna, all_meth, all_labels = [], [], [], []

    for tissue, expr_df in tissue_dict.items():
        label = tissue_label_map[tissue]
        sub = expr_df.loc[top_genes].T  # (samples, genes)

        rna_norm  = normalize_rna(sub.values)
        atac_norm = normalize_features(
            build_atac_proxy(rna_norm, N_PEAKS, label, SEED)
        )
        meth_norm = normalize_features(
            build_methylation_proxy(rna_norm, N_PEAKS, label, SEED)
        )

        atac_effect = real_tissue_effect(tissue, REAL_ATAC_PROFILE, "mean_signal_value")
        meth_effect = real_tissue_effect(tissue, REAL_METHYLATION_PROFILE, "mean_percent_methylation")
        atac_norm = atac_norm + atac_effect
        meth_norm = meth_norm + meth_effect
        if atac_effect or meth_effect:
            print(f"    [{tissue}] real cross-tissue effect applied — "
                  f"ATAC: {atac_effect:+.3f}, methylation: {meth_effect:+.3f}")

        labels = np.full(rna_norm.shape[0], label, dtype=np.int64)

        all_rna.append(rna_norm)
        all_atac.append(atac_norm)
        all_meth.append(meth_norm)
        all_labels.append(labels)

        print(f"  {tissue}: {rna_norm.shape[0]} samples")

    rna    = np.concatenate(all_rna)
    atac   = np.concatenate(all_atac)
    meth   = np.concatenate(all_meth)
    labels = np.concatenate(all_labels)

    print("\n[5/5] Splitting into train/val/test...")
    _save_splits(rna, atac, meth, labels)
    print("\nDataset built successfully.")


def _save_splits(rna, atac, meth, labels):
    """Split and save numpy arrays to data/processed/"""
    idx = np.arange(len(labels))

    idx_train, idx_test = train_test_split(
        idx, test_size=cfg.data.TEST_SPLIT, random_state=SEED, stratify=labels
    )
    idx_train, idx_val = train_test_split(
        idx_train,
        test_size=cfg.data.VAL_SPLIT / (1 - cfg.data.TEST_SPLIT),
        random_state=SEED,
        stratify=labels[idx_train]
    )

    splits = {
        "train": idx_train,
        "val":   idx_val,
        "test":  idx_test,
    }

    for split_name, split_idx in splits.items():
        out_dir = PROCESSED_DIR / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "rna.npy",    rna[split_idx])
        np.save(out_dir / "atac.npy",   atac[split_idx])
        np.save(out_dir / "meth.npy",   meth[split_idx])
        np.save(out_dir / "labels.npy", labels[split_idx])

        print(f"  {split_name}: {len(split_idx)} samples -> {out_dir}")


def build_synthetic_demo():
    """
    Build a small synthetic dataset for code testing.
    Simulates 3 tissues × 200 samples.
    Replace with real data when available.
    """
    print("Creating synthetic demo dataset...")
    rng = np.random.RandomState(SEED)

    n_per_tissue = 200
    n_tissues    = len(TISSUES)
    n_samples    = n_per_tissue * n_tissues

    tissue_centers = rng.randn(n_tissues, N_GENES) * 2.0

    all_rna, all_atac, all_meth, all_labels = [], [], [], []

    for t_idx, tissue in enumerate(TISSUES):
        rna = tissue_centers[t_idx] + rng.randn(n_per_tissue, N_GENES) * 0.8

        atac_signal = rna[:, :N_PEAKS] if N_PEAKS <= N_GENES else np.concatenate(
            [rna, rng.randn(n_per_tissue, N_PEAKS - N_GENES)], axis=1
        )
        atac = atac_signal + rng.randn(n_per_tissue, N_PEAKS) * 0.5

        meth = -atac_signal * 0.7 + rng.randn(n_per_tissue, N_PEAKS) * 0.4
        meth = (meth - meth.min()) / (meth.max() - meth.min() + 1e-8)

        atac_effect = real_tissue_effect(tissue, REAL_ATAC_PROFILE, "mean_signal_value")
        meth_effect = real_tissue_effect(tissue, REAL_METHYLATION_PROFILE, "mean_percent_methylation")
        atac = atac + atac_effect
        meth = np.clip(meth + meth_effect, 0.0, 1.0)
        if atac_effect or meth_effect:
            print(f"  {tissue}: real cross-tissue effect applied — "
                  f"ATAC: {atac_effect:+.3f}, methylation: {meth_effect:+.3f}")

        labels = np.full(n_per_tissue, t_idx, dtype=np.int64)

        all_rna.append(rna.astype(np.float32))
        all_atac.append(atac.astype(np.float32))
        all_meth.append(meth.astype(np.float32))
        all_labels.append(labels)

    rna    = np.concatenate(all_rna)
    atac   = np.concatenate(all_atac)
    meth   = np.concatenate(all_meth)
    labels = np.concatenate(all_labels)

    _save_splits(rna, atac, meth, labels)
    print(f"Synthetic demo: {n_samples} samples | {N_GENES} genes | {N_PEAKS} peaks")
    print("NOTE: Replace with real GTEx + ENCODE data for publication results.")


if __name__ == "__main__":
    build_dataset()
