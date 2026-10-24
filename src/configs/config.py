class DataConfig:
    
    RAW_DIR        = "data/raw"
    PROCESSED_DIR  = "data/processed"
    TRAIN_DIR      = "data/train"
    VAL_DIR        = "data/val"
    TEST_DIR       = "data/test"

    
    TISSUES        = ["liver", "brain", "lung"]
    TRAIN_TISSUES  = ["liver", "brain"]   # train on these
    TEST_TISSUES   = ["lung"]             # test generalization on unseen tissue

    N_GENES        = 512    # increase to 3000 for publication runs
    N_ATAC_PEAKS   = 512    # increase to 3000 for publication runs
    USE_METHYLATION = True  # set False to skip methylation modality

    VAL_SPLIT      = 0.15
    TEST_SPLIT     = 0.10
    RANDOM_SEED    = 42


class ModelConfig:
    ATAC_DIM       = 512    # number of ATAC peaks
    METH_DIM       = 512    # number of methylation sites (0 if not used)
    RNA_DIM        = 512    # number of genes (output)

    INPUT_DIM      = ATAC_DIM + METH_DIM   # combined input

    HIDDEN_DIM     = 160    # embedding dimension (bumped from 128)
    NUM_HEADS      = 4      # attention heads (must divide HIDDEN_DIM evenly)
    NUM_LAYERS     = 3      # transformer encoder layers (bumped from 2)
    FF_DIM         = 640    # feedforward dimension inside transformer (bumped from 512)
    DROPOUT        = 0.1

    MASK_RATIO     = 0.15   # fraction of input to mask (BERT-style)


class PretrainConfig:
    EPOCHS         = 10   # validation loss plateaus by epoch ~10 on the corrected data
    # Attention is O(seq_len^2); long ATAC token sequences need smaller batches on CPU.
    BATCH_SIZE     = 8
    LEARNING_RATE  = 1e-4
    WEIGHT_DECAY   = 1e-5
    WARMUP_STEPS   = 500   # linear LR warmup (optimizer steps), then cosine decay
    GRAD_CLIP      = 1.0
    LOG_EVERY      = 10     # log every N batches
    EARLY_STOP_PATIENCE = 4


class FinetuneConfig:
    # Values used for the reported results (manuscript Table VII).
    EPOCHS         = 10   # transformer models
    BATCH_SIZE     = 8
    LEARNING_RATE  = 5e-4   # transformer models
    WEIGHT_DECAY   = 1e-5
    GRAD_CLIP      = None   # no gradient clipping during fine-tuning
    USE_COSINE_SCHEDULE = False
    BASELINE_EPOCHS = 40
    LINEAR_LR      = 5e-4
    MLP_LR         = 1e-3
    FREEZE_ENCODER = False  # set True to freeze pretrained encoder
    EARLY_STOP_PATIENCE = 8


class DiffusionConfig:
    EPOCHS         = 25   # bumped from 10 - val loss was still decreasing every epoch, no plateau yet
    BATCH_SIZE     = 8
    LEARNING_RATE  = 1e-4
    WEIGHT_DECAY   = 1e-5
    GRAD_CLIP      = 1.0
    TIMESTEPS      = 200      # diffusion steps (T)
    BETA_START     = 1e-4
    BETA_END       = 0.02
    HIDDEN_DIM     = 256      # denoiser MLP width
    TIME_DIM       = 64       # timestep embedding dim
    LABEL_DIM      = 16       # tissue embedding dim
    SAMPLE_STEPS   = 50       # fast sampling steps for platform UI
    LOG_EVERY      = 10


class EvalConfig:
    METRICS        = ["pearson", "r2", "mse"]
    SAVE_FIGURES   = True
    FIGURE_DIR     = "outputs/figures"


class Config:
    data     = DataConfig()
    model    = ModelConfig()
    pretrain = PretrainConfig()
    finetune = FinetuneConfig()
    diffusion = DiffusionConfig()
    eval     = EvalConfig()
