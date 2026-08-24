"""
main.py

Single entry point to run any phase of the project.

Usage:
    python main.py --mode all          # run everything in order
    python main.py --mode data         # preprocess data only
    python main.py --mode baseline     # train baselines only
    python main.py --mode pretrain     # self-supervised pretraining
    python main.py --mode finetune     # fine-tune + baseline comparison
    python main.py --mode diffusion    # train conditional diffusion model
    python main.py --mode evaluate     # run full evaluation + figures
    python main.py --mode platform     # launch clinical web platform
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def run_download():
    print("\n" + "=" * 60)
    print(" Step 0: Download Data (GTEx + ENCODE)")
    print("=" * 60)
    from data.download_data import download_gtex, download_encode, download_methylation
    download_gtex()
    download_encode()
    download_methylation()


def run_data():
    print("\n" + "=" * 60)
    print(" Step 1: Data Preprocessing")
    print("=" * 60)
    from data.preprocess_atac import build_real_atac_profile
    from data.preprocess_methylation import build_real_methylation_profile
    from data.preprocess import build_dataset
    build_real_atac_profile()
    build_real_methylation_profile()
    build_dataset()


def run_pretrain():
    print("\n" + "=" * 60)
    print(" Step 2: Self-Supervised Pretraining")
    print("=" * 60)
    from src.training.pretrain import pretrain
    pretrain()


def run_finetune():
    print("\n" + "=" * 60)
    print(" Step 3: Fine-Tuning + Baseline Comparison")
    print("=" * 60)
    from src.training.finetune import finetune
    finetune()


def run_diffusion():
    print("\n" + "=" * 60)
    print(" Step 4: Train Conditional Diffusion Model")
    print("=" * 60)
    from src.training.train_diffusion import train_diffusion
    train_diffusion()


def run_evaluate():
    print("\n" + "=" * 60)
    print(" Step 5: Full Evaluation")
    print("=" * 60)
    from src.evaluation.evaluate import run_evaluation
    run_evaluation()


def run_platform():
    print("\n" + "=" * 60)
    print(" Launching MOTIF Clinical Platform")
    print("=" * 60)
    app_path = Path(__file__).parent / "src" / "webapp" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


def main():
    parser = argparse.ArgumentParser(description="MOTIF (Multi-Omics Transformer for Inferring Function)")
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["all", "download", "data", "pretrain", "finetune",
                 "diffusion", "evaluate", "baseline", "platform"],
        help="Which phase to run",
    )
    args = parser.parse_args()

    print("\n" + "=" * 62)
    print("   MOTIF -- Multi-Omics Transformer for Inferring Function")
    print("   Transformer + Conditional Diffusion")
    print("=" * 62)

    if args.mode == "download":
        run_download()
    elif args.mode == "data":
        run_data()
    elif args.mode == "pretrain":
        run_pretrain()
    elif args.mode in ("finetune", "baseline"):
        run_finetune()
    elif args.mode == "diffusion":
        run_diffusion()
    elif args.mode == "evaluate":
        run_evaluate()
    elif args.mode == "platform":
        run_platform()
    elif args.mode == "all":
        print("\nRunning complete pipeline...")
        run_download()
        run_data()
        run_pretrain()
        run_finetune()
        run_diffusion()
        run_evaluate()

    print("\nDone.")


if __name__ == "__main__":
    main()
