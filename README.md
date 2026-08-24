# MOTIF

Self-supervised transformer for predicting gene expression from chromatin
accessibility and DNA methylation, paired with a conditional diffusion model
(DDPM) for generating synthetic multi-omics profiles. Built and evaluated on
real GTEx RNA-seq and ENCODE ATAC-seq/WGBS data across liver, brain, and lung
tissue, with cross-tissue transfer evaluated explicitly rather than assumed.

MOTIF is pretrained with a masked-reconstruction objective over accessibility
and methylation tokens, then fine-tuned to predict expression. A linear
regression and an MLP are trained under the same objective as baselines. All
four models are compared on held-out same-tissue prediction and on a
liver-to-brain/liver-to-lung transfer benchmark.

## Layout

- `src/models` — the MOTIF transformer encoder, the linear/MLP baselines, and the DDPM
- `src/training` — pretraining, fine-tuning, and diffusion training loops
- `src/evaluation` — metrics, biological validation checks, figures, manuscript generation
- `src/webapp` — a Streamlit platform for interactive prediction and profile generation
- `data` — download and preprocessing scripts for GTEx/ENCODE
- `notebooks/MOTIF_Pipeline.ipynb` — the pipeline end to end in one notebook

Raw/processed data and trained checkpoints aren't tracked in this repo (the
raw download alone is several gigabytes) — running the pipeline below
regenerates all of it locally.

## Setup

```bash
pip install -r requirements.txt
python data/download_data.py
python data/preprocess_atac.py
python data/preprocess_methylation.py
python data/preprocess.py
```

## Training and evaluation

```bash
python main.py --mode pretrain    # self-supervised masked-reconstruction pretraining
python main.py --mode finetune    # baselines + fine-tuning
python main.py --mode diffusion   # conditional diffusion model
python main.py --mode evaluate    # metrics + figures
```

or `python main.py --mode all` for the full pipeline in order.

## Web platform

```bash
python main.py --mode platform
# or: streamlit run src/webapp/app.py
```

Six pages: a dashboard with model status and headline results, an expression
predictor, a synthetic profile generator, a tissue explorer, cross-tissue
transfer results, and a research summary.

## Results

Current results, methodology, and an honest accounting of this pilot's
limitations — including a real construction issue in the same-tissue
benchmark that favors the linear baseline — are in
[`outputs/results/manuscript_draft.md`](outputs/results/manuscript_draft.md),
regenerated automatically each time `python main.py --mode evaluate` runs.

## Data sources

GTEx v8 (RNA-seq) and ENCODE (ATAC-seq, WGBS methylation), liver/brain/lung.
Exact file accessions are in `data/download_data.py`.

## License

MIT — see [LICENSE](LICENSE).
