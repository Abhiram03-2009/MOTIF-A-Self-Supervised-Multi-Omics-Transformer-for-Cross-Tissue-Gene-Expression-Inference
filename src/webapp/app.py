"""
MOTIF Research Platform: Streamlit web app for exploring the model.

Launch:
    streamlit run src/webapp/app.py
    python main.py --mode platform
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from scipy import stats

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.configs.config import Config
from src.webapp.inference import get_platform

cfg = Config()
TISSUE_COLORS = {"liver": "#2E86AB", "brain": "#A23B72", "lung": "#F18F01"}

# Fill in once the repo is public. Kept as one constant so it's a single
# edit to point the platform at the real research writeup.
GITHUB_REPO_URL = "https://github.com/your-org/motif"

st.set_page_config(page_title="MOTIF", layout="wide")

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}

    html, body, [class*="css"] { font-family: "Segoe UI", -apple-system, Roboto, Helvetica, Arial, sans-serif; }

    :root {
        --motif-ink: #16212B;
        --motif-primary: #16344A;
        --motif-accent: #2E6E8E;
        --motif-bg-soft: #F5F7F8;
        --motif-border: #DDE3E8;
        --motif-ok: #276B45;
        --motif-warn: #8A4B14;
    }

    [data-testid="stSidebarCollapsedControl"] { display: none; }

    .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1150px; }

    .motif-topbar { display: flex; align-items: baseline; gap: 0.6rem; margin-bottom: 0.9rem; }
    .motif-topbar-name { font-family: Georgia, "Times New Roman", serif; font-size: 1.4rem;
                          font-weight: 700; color: var(--motif-primary); letter-spacing: 0.01em; }
    .motif-topbar-sub { font-size: 0.82rem; color: #5C6B76; }

    div[data-testid="stButtonGroup"] {
        border-bottom: 1px solid var(--motif-border);
        padding-bottom: 0.9rem;
        margin-bottom: 0.3rem;
        gap: 0.4rem;
    }
    div[data-testid="stButtonGroup"] button {
        font-size: 0.98rem;
        font-weight: 600;
        padding: 0.55rem 1.1rem;
        border-radius: 6px;
        transition: color 0.18s ease, background-color 0.18s ease, border-color 0.18s ease, transform 0.1s ease;
    }
    div[data-testid="stButtonGroup"] button:hover { transform: translateY(-1px); }
    button[data-testid="stBaseButton-segmented_controlActive"] {
        background-color: var(--motif-primary) !important;
        border-color: var(--motif-primary) !important;
        color: white !important;
    }

    [class*="st-key-page-"] { animation: motifFadeIn 0.4s cubic-bezier(0.16, 1, 0.3, 1); }
    @keyframes motifFadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    div[data-testid="stMetric"], .motif-card, .motif-card-soft, div[data-testid="stDataFrame"] {
        transition: box-shadow 0.18s ease, border-color 0.18s ease;
    }
    div[data-testid="stButton"] button, div[data-testid="stLinkButton"] a {
        transition: background-color 0.15s ease, border-color 0.15s ease, transform 0.1s ease;
    }
    div[data-testid="stButton"] button:active, div[data-testid="stLinkButton"] a:active { transform: scale(0.98); }

    .motif-brand { font-family: Georgia, "Times New Roman", serif; font-size: 1.9rem; font-weight: 700;
                   color: var(--motif-primary); letter-spacing: 0.01em; margin-bottom: 0; line-height: 1.1; }
    .motif-tagline { color: #5C6B76; font-size: 0.95rem; margin-top: 0.3rem; margin-bottom: 1.6rem;
                      max-width: 640px; line-height: 1.5; }
    .motif-eyebrow { text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.7rem;
                      font-weight: 600; color: var(--motif-accent); margin-bottom: 0.25rem; }

    .motif-card { background: white; border: 1px solid var(--motif-border); border-radius: 4px;
                  padding: 1.25rem 1.4rem; margin-bottom: 1rem; }
    .motif-card-soft { background: var(--motif-bg-soft); border: 1px solid var(--motif-border);
                        border-radius: 4px; padding: 1.25rem 1.4rem; margin-bottom: 1rem; }

    .motif-status { display: flex; align-items: center; gap: 0.45rem; font-size: 0.82rem; font-weight: 600; }
    .motif-status-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
    .motif-status-ok .motif-status-dot { background: var(--motif-ok); }
    .motif-status-ok { color: var(--motif-ok); }
    .motif-status-warn .motif-status-dot { background: var(--motif-warn); }
    .motif-status-warn { color: var(--motif-warn); }

    .motif-step { border-left: 2px solid var(--motif-border); padding: 0.1rem 0 0.1rem 0.9rem; margin: 0.75rem 0; }
    .motif-step-num { font-family: Georgia, serif; color: var(--motif-accent); font-weight: 700;
                       font-size: 0.85rem; margin-right: 0.35rem; }

    .motif-tag { color: #3D4B55; font-size: 0.85rem; margin-right: 1.1rem; border-bottom: 2px solid; padding-bottom: 1px; }

    div[data-testid="stMetric"] { background: white; border: 1px solid var(--motif-border);
                                    border-radius: 4px; padding: 0.9rem 1rem; }
    div[data-testid="stMetricLabel"] { font-weight: 600; color: #5C6B76; }

    h1, h2, h3 { color: var(--motif-primary); }
    div[data-testid="stButton"] button, div[data-testid="stLinkButton"] a { border-radius: 4px; }
    hr { margin: 1.6rem 0; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_platform():
    return get_platform()


def status_badge(ok: bool) -> str:
    cls, label = ("motif-status-ok", "Ready") if ok else ("motif-status-warn", "Missing")
    return (f'<div class="motif-status {cls}">'
            f'<span class="motif-status-dot"></span><span>{label}</span></div>')


def page_header(eyebrow: str, title: str, subtitle: str = ""):
    st.markdown(f'<p class="motif-eyebrow">{eyebrow}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="motif-brand" style="font-size:1.6rem;">{title}</p>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="motif-tagline">{subtitle}</p>', unsafe_allow_html=True)


def page_dashboard(platform):
    page_header("Home", "Dashboard",
                "Self-supervised transformer and diffusion platform for cross-tissue gene "
                "expression prediction and synthetic profile generation.")

    status = platform.model_status()
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, (label, key) in zip([c1, c2, c3, c4, c5],
                                  [("MOTIF Transformer", "transformer"), ("Linear Baseline", "linear"),
                                   ("MLP Baseline", "mlp"), ("Diffusion Model", "diffusion"), ("Dataset", "data")]):
        with col:
            st.markdown(
                f'<div class="motif-card" style="padding:0.9rem 1rem;">'
                f'<div style="font-size:0.78rem; color:#5C6B76; font-weight:600; margin-bottom:0.5rem;">{label}</div>'
                f'{status_badge(status[key])}</div>',
                unsafe_allow_html=True,
            )

    st.write("")
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown(
            '<div class="motif-card"><div class="motif-eyebrow">Overview</div>'
            '<div style="font-size:0.95rem; line-height:1.6; color:#33404A;">'
            'MOTIF predicts gene expression directly from chromatin accessibility (ATAC-seq) '
            'and DNA methylation, and carries much of what it learns across tissues. A diffusion '
            'model for synthetic profiles is included as a prototype.</div></div>',
            unsafe_allow_html=True,
        )
        steps = [
            ("01", "Predict", "Infer expression from epigenomic input using the trained transformer."),
            ("02", "Generate", "Sample synthetic profiles from the diffusion model (prototype)."),
            ("03", "Explore", "Visualize tissue structure and cross-tissue transfer performance."),
        ]
        for num, title, desc in steps:
            st.markdown(
                f'<div class="motif-step">'
                f'<span class="motif-step-num">{num}</span>'
                f'<b style="color:var(--motif-primary); font-size:0.92rem;">{title}</b>'
                f'<div style="font-size:0.86rem; color:#5C6B76; margin-top:0.1rem;">{desc}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown('<div class="motif-eyebrow" style="margin-top:1.2rem;">Tissues Covered</div>', unsafe_allow_html=True)
        tags = "".join(
            f'<span class="motif-tag" style="border-color:{TISSUE_COLORS[t]};">{t.capitalize()}</span>'
            for t in cfg.data.TISSUES
        )
        st.markdown(tags, unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="motif-eyebrow">Model Performance — Expression Prediction</div>', unsafe_allow_html=True)
        results = platform.results.get("expression", {})
        if results:
            df = pd.DataFrame([
                {"Model": k, "Pearson r": f"{v['pearson']:.3f}", "R²": f"{v['r2']:.3f}"}
                for k, v in results.items()
            ])
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.info("No evaluation results yet — run `python main.py --mode evaluate`.")

        ct = platform.cross_tissue_results()
        if ct:
            st.markdown('<div class="motif-eyebrow" style="margin-top:1rem;">Cross-Tissue Transfer</div>', unsafe_allow_html=True)
            df_ct = pd.DataFrame([
                {"Train → Test": k.replace("_to_", " → "), "Pearson r": f"{v['pearson']:.3f}", "R²": f"{v['r2']:.3f}"}
                for k, v in ct.items()
            ])
            st.dataframe(df_ct, hide_index=True, use_container_width=True)


def page_expression_predictor(platform):
    page_header("Research Tool", "Gene Expression Predictor",
                "Infer RNA expression from chromatin accessibility and methylation input.")
    status = platform.model_status()

    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        with st.container(border=True):
            tissue = st.selectbox("Tissue", cfg.data.TISSUES, key="pred_tissue")
            model_name = st.selectbox("Model", ["transformer", "linear", "mlp"], key="pred_model")
            model_key = "transformer" if model_name == "transformer" else model_name
            if not status.get(model_key, False):
                st.warning(f"No trained **{model_name}** checkpoint found yet. Train it first "
                           f"with `python main.py --mode finetune` before predicting.")
            n_show = st.slider("Genes shown", 20, 100, 50, key="pred_n_show")
            if st.button("Run Prediction", type="primary", key="pred_run",
                         disabled=not status.get(model_key, False), use_container_width=True):
                try:
                    atac, meth, rna_true = platform.get_demo_profile(tissue)
                    pred = platform.predict_expression(atac, meth, model_name)
                    st.session_state["pred"] = pred[0]
                    st.session_state["rna_true"] = rna_true
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
    with c2:
        pred = st.session_state.get("pred")
        rna_true = st.session_state.get("rna_true")
        if pred is not None and rna_true is not None:
            n = st.session_state.get("pred_n_show", 50)
            r, _ = stats.pearsonr(pred, rna_true)
            mc1, mc2 = st.columns(2)
            mc1.metric("Pearson r", f"{r:.3f}")
            mc2.metric("Genes evaluated", f"{len(pred)}")

            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.arange(min(n, len(pred)))
            ax.bar(x - 0.2, pred[:n], 0.4, label="Predicted", color="#2E86AB")
            ax.bar(x + 0.2, rna_true[:n], 0.4, label="Actual", color="#A23B72")
            ax.legend()
            ax.set_xlabel("Gene index")
            ax.set_ylabel("Expression")
            ax.set_title(f"Predicted vs. Actual Expression — {tissue.capitalize()}")
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Select a tissue and model, then click **Run Prediction** to run inference on a demo profile.")


def page_synthetic_generator(platform):
    page_header("Research Tool", "Synthetic Profile Generator",
                "Sample multi-omics profiles from the diffusion model.")
    st.warning("Research prototype: in the current model the output barely changes with the selected tissue, "
               "so these profiles should not be treated as tissue-specific.")
    if not platform.model_status()["diffusion"]:
        st.error("No trained diffusion checkpoint found. Train it first with `python main.py --mode diffusion`.")
        return

    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        with st.container(border=True):
            tissue = st.selectbox("Tissue", cfg.data.TISSUES, key="syn_tissue_select")
            n = st.slider("Profiles to generate", 5, 50, 20, key="syn_n")
            if st.button("Generate", type="primary", key="syn_generate", use_container_width=True):
                try:
                    with st.spinner("Running diffusion sampling..."):
                        synthetic = platform.generate_synthetic(tissue, n, fast=True)
                        st.session_state["syn"] = synthetic
                        st.session_state["syn_tissue"] = tissue
                except Exception as e:
                    st.error(f"Diffusion generation failed: {e}")
                    st.session_state.pop("syn", None)
                    st.session_state.pop("syn_tissue", None)

    with c2:
        syn = st.session_state.get("syn")
        syn_tissue = st.session_state.get("syn_tissue")
        if syn is not None and syn_tissue is not None:
            n = st.session_state.get("syn_n", len(syn))
            ds = platform.load_dataset("test")
            t_idx = cfg.data.TISSUES.index(syn_tissue)
            real = ds.atac.numpy()[ds.labels.numpy() == t_idx]
            n_real = min(n, len(real))

            fig, ax = plt.subplots(figsize=(8, 5))
            nf = min(200, syn.shape[1])
            if n_real > 0:
                combined = np.vstack([real[:n_real, :nf], syn[:, :nf]])
                pca = PCA(2)
                coords = pca.fit_transform(combined)
                ax.scatter(coords[:n_real, 0], coords[:n_real, 1],
                           c=TISSUE_COLORS[syn_tissue], label="Real", alpha=0.6, s=30)
                ax.scatter(coords[n_real:, 0], coords[n_real:, 1],
                           c="gray", label="Synthetic (diffusion)", marker="x", alpha=0.6, s=30)
            else:
                pca = PCA(2)
                coords = pca.fit_transform(syn[:, :nf])
                ax.scatter(coords[:, 0], coords[:, 1], c="gray",
                           label="Synthetic (diffusion)", marker="x", alpha=0.6, s=30)
            ax.legend()
            ax.set_title(f"{syn_tissue.capitalize()} — Real vs. Diffusion-Generated Profiles")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            st.pyplot(fig)
            plt.close()

            st.success(f"Generated {len(syn)} synthetic profiles ({syn.shape[1]} features each).")
            st.download_button(
                "Download CSV",
                pd.DataFrame(syn).to_csv(index=False),
                file_name=f"synthetic_{syn_tissue}.csv",
                mime="text/csv",
                key="syn_download",
            )
        else:
            st.info("Select a tissue and click **Generate** to create synthetic profiles.")


def page_tissue_explorer(platform):
    page_header("Analysis", "Tissue Explorer", "Visualize expression structure across tissues via PCA.")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        split = c1.selectbox("Data split", ["train", "val", "test"])
        nf = c2.slider("PCA features", 50, 300, 100)

    ds = platform.load_dataset(split)
    rna, labels = ds.rna.numpy(), ds.labels.numpy()
    coords = PCA(2).fit_transform(rna[:, :nf])
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, t in enumerate(cfg.data.TISSUES):
        m = labels == i
        ax.scatter(coords[m, 0], coords[m, 1], c=TISSUE_COLORS[t], label=t.capitalize(), alpha=0.6, s=25)
    ax.legend()
    ax.set_title(f"RNA Expression PCA — {split.capitalize()} Split")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    st.pyplot(fig)
    plt.close()


def page_cross_tissue(platform):
    page_header("Analysis", "Cross-Tissue Transfer", "Model trained on one tissue, evaluated cold on another.")
    t = platform.cross_tissue_results()
    if t:
        st.dataframe(
            pd.DataFrame([
                {"Experiment": k.replace("_to_", " → "), "Pearson r": f"{v['pearson']:.4f}", "R²": f"{v['r2']:.4f}"}
                for k, v in t.items()
            ]),
            hide_index=True, use_container_width=True,
        )
    else:
        st.info("No cross-tissue results yet — run `python main.py --mode evaluate`.")
    fig = ROOT / "outputs" / "figures" / "2_cross_tissue_heatmap.png"
    if fig.exists():
        st.image(str(fig), use_container_width=True)


def page_research_results(platform):
    page_header("Documentation", "Research & Methodology",
                "Full technical writeup, figures, and reproducibility details.")

    st.markdown(
        f'''<div class="motif-card-soft">
        <div style="font-size:0.98rem; line-height:1.7; color:#33404A;">
        This platform runs a self-supervised multi-omics transformer paired with a conditional
        diffusion model, trained and evaluated on real GTEx and ENCODE data. The full methodology —
        architecture details, training curves, benchmark comparisons, cross-tissue evaluation, and
        the manuscript draft — is maintained in the project repository rather than duplicated here.
        </div></div>''',
        unsafe_allow_html=True,
    )

    st.link_button("View Full Research on GitHub", GITHUB_REPO_URL, type="primary", use_container_width=False)

    figs = sorted((ROOT / "outputs" / "figures").glob("*.png"))
    if figs:
        with st.expander(f"Preview figures generated locally ({len(figs)})"):
            for f in figs:
                st.image(str(f), caption=f.stem.replace("_", " ").title(), use_container_width=True)


def main():
    platform = load_platform()
    st.markdown(
        '<div class="motif-topbar">'
        '<span class="motif-topbar-name">MOTIF</span>'
        '<span class="motif-topbar-sub">Research Platform</span></div>',
        unsafe_allow_html=True,
    )

    tab_names = ["Dashboard", "Expression Predictor", "Synthetic Generator",
                 "Tissue Explorer", "Cross-Tissue Analysis", "Research Results"]
    page_funcs = dict(zip(tab_names, [page_dashboard, page_expression_predictor, page_synthetic_generator,
                                       page_tissue_explorer, page_cross_tissue, page_research_results]))

    selected = st.segmented_control("Navigation", tab_names, default=tab_names[0],
                                     key="motif_nav", label_visibility="collapsed")
    selected = selected or tab_names[0]

    with st.container(key=f"page-{selected.lower().replace(' ', '-')}"):
        page_funcs[selected](platform)

if __name__ == "__main__":
    main()
