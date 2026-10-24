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

## Quick start (no training needed)

The trained checkpoints (`outputs/checkpoints/*_best.pt`) and the processed
dataset (`data/processed/`, about 6 MB) are included, so the web platform runs
right after installing the requirements:

```bash
pip install -r requirements.txt
python main.py --mode platform
```

The included checkpoints come from a full CPU re-run of `python main.py --mode all`
(PyTorch 2.14, October 2026). They reproduce the reported results: fine-tuning
Pearson/R² are within 0.01 of the values in `outputs/results/`, and the
classification and cross-tissue transfer metrics are identical.

The raw GTEx/ENCODE download (several gigabytes) is not tracked; the steps
below regenerate everything from scratch if you want to retrain.

## Setup (to retrain from raw data)

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

For a non-technical walkthrough of what this project does and why, see
[`docs/MOTIF_Overview.pptx`](docs/MOTIF_Overview.pptx) — a short slide deck
covering the problem, the pipeline, and the headline results.

## Data sources

GTEx v8 (RNA-seq) and ENCODE (ATAC-seq, WGBS methylation), liver/brain/lung.
Exact file accessions are in `data/download_data.py`.

## License

MIT — see [LICENSE](LICENSE).
