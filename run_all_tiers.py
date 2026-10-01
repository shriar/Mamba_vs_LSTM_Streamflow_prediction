"""Sequentially run all parameter tiers for LSTM and Mamba on CAMELS.

Execution order:
    Tier 1: LSTM -> Mamba
    Tier 2: LSTM -> Mamba
    Tier 3: LSTM -> Mamba

Each model run executes in an isolated subprocess by default to ensure 100%
GPU VRAM reclamation between runs, preventing CUDA fragmentation.

Parallel Execution:
    When running both models, separate log files are used to avoid conflicts:
    - running_log_lstm.txt (for LSTM runs)
    - running_log_mamba.txt (for Mamba runs)
    This allows running LSTM and Mamba in separate terminals simultaneously.

Log Format:
    Log files use human-readable format:
    [timestamp] Tier X (Name) | MODEL | Params: ###### | Time: ##.#m | Median NSE: #.#### | KGE: #.#### | Version: # | Status: ######

Usage:
    # Run all 3 tiers (1..3) for BOTH LSTM and Mamba (default):
    python run_all_tiers.py

    # Run ONLY LSTM across all tiers:
    python run_all_tiers.py --model lstm
    # OR:
    python run_all_tiers.py --only-lstm

    # Run ONLY Mamba across all tiers:
    python run_all_tiers.py --model mamba
    # OR:
    python run_all_tiers.py --only-mamba

    # Run specific tiers (e.g. tiers 1, 2, and 3):
    python run_all_tiers.py --tiers 1 2 3

    # Force re-running even if basin_metrics.csv already exists:
    python run_all_tiers.py --force

    # Quick test with 1 epoch on tier 1 (e.g. mamba only):
    python run_all_tiers.py --tiers 1 --epoch 1 --model mamba --force

    # Evaluation only (skip training, assume checkpoints exist):
    python run_all_tiers.py --tiers 1 --model mamba --eval-only

    # Multi-layer LSTM experiments:
    python run_all_tiers.py --only-lstm --lstm-layers 4

    # LSTM with custom learning rate:
    python run_all_tiers.py --only-lstm --lstm-lr 1e-3

    # LSTM with custom weight decay:
    python run_all_tiers.py --only-lstm --lstm-weight-decay 1e-4

    # LSTM with cosine decay from lr to lstm_min_lr:
    python run_all_tiers.py --only-lstm --lstm-scheduler cosine --lstm-min-lr 1e-5

    # Mamba with FiLM modulation for static/dynamic split:
    python run_all_tiers.py --only-mamba --mamba-variant film

    # Mamba with hybrid FiLM (full concat + static-only FiLM):
    python run_all_tiers.py --only-mamba --mamba-variant hybrid

    # Mamba with adapter layers for static attribute injection:
    python run_all_tiers.py --only-mamba --mamba-variant adapter

    # Mamba with custom learning rates:
    python run_all_tiers.py --only-mamba --mamba-lr 2e-4 --mamba-max-lr 5e-4

    # Mamba with explicit weight decay:
    python run_all_tiers.py --only-mamba --mamba-weight-decay 1e-4

    # Mamba with OneCycleLR scheduler (default for LSTM-like behavior):
    python run_all_tiers.py --only-mamba --mamba-scheduler onecycle

    # Mamba with CosineAnnealingLR scheduler (recommended for SSM):
    python run_all_tiers.py --only-mamba --mamba-scheduler cosine

    # Mamba with prefix tokens (basin context tokens prepended to sequence):
    python run_all_tiers.py --only-mamba --mamba-variant prefix

    # Mamba with custom number of prefix tokens:
    python run_all_tiers.py --only-mamba --mamba-variant prefix --mamba-n-prefix-tokens 4

    # Mamba with dedicated residual block abstraction:
    python run_all_tiers.py --only-mamba --mamba-variant residual

    # Mamba with adapter layers for static attribute injection:
    python run_all_tiers.py --only-mamba --mamba-variant adapter

    # Mamba with custom adapter bottleneck dimension:
    python run_all_tiers.py --only-mamba --mamba-variant adapter --mamba-adapter-bottleneck 64

    # Custom output directory:
    python run_all_tiers.py --output-dir my_custom_output

Parallel Execution (Separate Terminals):
    # Terminal 1: Run all LSTM tiers
    python run_all_tiers.py --only-lstm

    # Terminal 2: Run all Mamba tiers (simultaneously)
    python run_all_tiers.py --only-mamba
"""

import argparse
from datetime import datetime
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

# =====================================================================
# Windows OpenMP dual-runtime guard (must run BEFORE numpy/torch/hydroDL
# imports, otherwise abort with:
#     OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll
#                     already initialized.
# conda numpy/scipy ship libiomp5md.dll; hydroDL/torch ship libomp.dll;
# both try to become the active OpenMP runtime.
# =====================================================================
if sys.platform.startswith("win") or os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*RNN module weights are not part of single contiguous chunk.*")

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))
_local_hydrodl = SCRIPT_DIR / "hydroDLpack"
if _local_hydrodl.is_dir():
    sys.path.insert(0, str(_local_hydrodl))

from config import ExperimentConfig, TIER_NAMES, set_seed


def append_to_running_log(entry: dict, script_dir: Path, model_type: str = "all"):
    """Append a completion record to model-specific log file immediately.

    Uses human-readable format that is also machine-parsable (pipe-delimited).
    Can be loaded with pandas: pd.read_csv(file, sep='|', skipinitialspace=True)

    Args:
        entry: Dictionary with run results
        script_dir: Directory containing log files
        model_type: 'lstm', 'mamba', or 'all' (for backward compatibility and single-model runs)
    """
    log_prefix = f"running_log_{model_type}" if model_type != "all" else "running_log"
    txt_file = script_dir / f"{log_prefix}.txt"

    # Extract values with defaults
    ts = entry.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    tier = str(entry.get("tier", "?"))
    tier_name = str(entry.get("tier_name", "?"))
    model = str(entry.get("model", "?"))
    params = str(entry.get("params", "N/A"))
    t_min = str(entry.get("train_time_min", "N/A"))
    nse = str(entry.get("median_nse", "N/A"))
    kge = str(entry.get("median_kge", "N/A"))
    status = str(entry.get("status", "UNKNOWN"))
    variant = str(entry.get("variant", ""))
    mamba_version = str(entry.get("mamba_version", ""))
    epoch = str(entry.get("epoch", "N/A"))

    # Build variant string for display
    variant_str = f" | Variant: {variant}" if variant else ""
    version_str = f" | Version: {mamba_version}" if mamba_version else ""
    rho_val = str(entry.get("rho", ""))
    rho_str = f" | Rho: {rho_val}" if rho_val else ""

    # Human-readable format (pipe-delimited for machine parsing)
    line = (
        f"[{ts}] Tier {tier} ({tier_name:<9}) | {model:<5} | "
        f"Params: {params:<10} | Time: {t_min:<6}m | "
        f"Median NSE: {nse:<7} | KGE: {kge:<7}{variant_str}{version_str}{rho_str} | Epochs: {epoch:<4} | Status: {status}\n"
    )
    with open(txt_file, "a", encoding="utf-8") as f:
        f.write(line)



def _evaluation_outputs_present(cfg: ExperimentConfig) -> bool:
    """Check if evaluation output files already exist with correct structure.

    Validates:
    - basin_metrics.csv has 671 rows with required columns (gauge_id, NSE, KGE)
    - checkpoint_metrics.csv exists for runs where epoch >= save_epoch
      (includes training_loss column)
    """
    output_dir = cfg.output_path()

    # 1. Check basin_metrics.csv has correct structure
    metrics_file = os.path.join(output_dir, "basin_metrics.csv")
    if not os.path.isfile(metrics_file):
        return False
    try:
        df = pd.read_csv(metrics_file)
        required_columns = ["gauge_id", "NSE", "KGE"]
        if not all(col in df.columns for col in required_columns):
            return False
        if len(df) != 671:
            return False
    except Exception:
        return False

    # 2. Check checkpoint_metrics.csv for runs where epoch >= save_epoch
    if cfg.epoch >= cfg.save_epoch:
        checkpoint_file = os.path.join(output_dir, "checkpoint_metrics.csv")
        if not os.path.isfile(checkpoint_file):
            return False
        try:
            cp_df = pd.read_csv(checkpoint_file)
            if "training_loss" not in cp_df.columns or cp_df.empty:
                return False
            if cp_df["training_loss"].isna().all():
                return False
        except Exception:
            return False

    return True


def is_evaluation_completed(cfg: ExperimentConfig) -> bool:
    """Check if evaluation has been completed for this run."""
    return _evaluation_outputs_present(cfg)


def is_run_completed(cfg: ExperimentConfig) -> bool:
    """Check if a run has completed its full training and evaluation.

    Validates:
    - model_Ep{epoch}.pt exists
    - basin_metrics.csv has 671 rows with required columns (gauge_id, NSE, KGE)
    - master.json contains expected configuration
    - checkpoint_metrics.csv exists for runs where epoch >= save_epoch (includes training_loss column)

    ``training_loss.csv`` is incremental and is not used as the completion
    marker because it can be truncated by an interrupted process.
    """
    output_dir = cfg.output_path()

    # 1. Check model checkpoint exists
    model_file = os.path.join(output_dir, f"model_Ep{cfg.epoch}.pt")
    if not os.path.isfile(model_file):
        return False

    # 2. Check evaluation outputs (basin_metrics + checkpoint_metrics) exist.
    # training_loss.csv is an incremental training artifact and may be
    # truncated if a later process was interrupted; it must not invalidate an
    # otherwise complete run whose final checkpoint and evaluation outputs are
    # present.
    if not _evaluation_outputs_present(cfg):
        return False

    # 3. Check master.json exists and has expected configuration
    master_file = os.path.join(output_dir, "master.json")
    if not os.path.isfile(master_file):
        return False

    try:
        import json
        with open(master_file, "r") as f:
            master_config = json.load(f)
        # Verify key configuration matches
        expected_checks = [
            ("nEpoch", cfg.epoch),
            ("hiddenSize", cfg.hidden_size),
        ]
        # Verify rho / batch match the requested run via train.miniBatch.
        # Fail closed: missing or malformed metadata must not mark a stale
        # directory as complete.
        mb = master_config.get("train", {}).get("miniBatch")
        if not isinstance(mb, (list, tuple)) or len(mb) < 2:
            return False
        if int(mb[1]) != int(cfg.rho) or int(mb[0]) != int(cfg.batch_size):
            return False
        # Persisted identity must match the requested run, not just its
        # hidden size and epoch (which can collide across experiments).
        experiment = master_config.get("experiment")
        if not isinstance(experiment, dict):
            return False
        identity_checks = {
            "model_type": cfg.model_type,
            "param_tier": cfg.param_tier,
            "hidden_size": cfg.hidden_size,
            "rho": cfg.rho,
            "batch_size": cfg.batch_size,
            "rho_scaling_experiment": cfg.rho_scaling_experiment,
        }
        for key, expected_value in identity_checks.items():
            if experiment.get(key) != expected_value:
                return False

        # Add model-specific checks
        if cfg.model_type == "lstm":
            expected_checks.append(("nLayer", cfg.lstm_layers))
            if "model" in master_config and "lstm_dropout" in master_config["model"]:
                expected_checks.append(("lstm_dropout", cfg.lstm_dropout))
            if "model" in master_config and "lr" in master_config["model"]:
                expected_checks.append(("lr", cfg.lr))
            if "model" in master_config and "lstm_optimizer" in master_config["model"]:
                expected_checks.append(("lstm_optimizer", cfg.lstm_optimizer))
            if "model" in master_config and "lstm_weight_decay" in master_config["model"]:
                expected_checks.append(("lstm_weight_decay", cfg.lstm_weight_decay))
            if "model" in master_config and "lstm_beta1" in master_config["model"]:
                expected_checks.append(("lstm_beta1", cfg.lstm_beta1))
            if "model" in master_config and "lstm_beta2" in master_config["model"]:
                expected_checks.append(("lstm_beta2", cfg.lstm_beta2))
            if "model" in master_config and "lstm_scheduler" in master_config["model"]:
                expected_checks.append(("lstm_scheduler", cfg.lstm_scheduler))
            if "model" in master_config and "lstm_grad_clip" in master_config["model"]:
                expected_checks.append(("lstm_grad_clip", cfg.lstm_grad_clip))
            if "model" in master_config and "lstm_weight_decay" in master_config["model"]:
                expected_checks.append(("lstm_weight_decay", cfg.lstm_weight_decay))
            if "model" in master_config and "lstm_min_lr" in master_config["model"]:
                expected_checks.append(("lstm_min_lr", cfg.lstm_min_lr))
        elif cfg.model_type == "mamba":
            expected_checks.append(("mamba_version", cfg.mamba_version))
            expected_checks.append(("mamba_variant", cfg.mamba_variant))
            expected_checks.append(("mamba_layers", cfg.mamba_layers))
            # Only check LR params if they exist in master config (for backward compatibility)
            if "model" in master_config and "mamba_lr" in master_config["model"]:
                expected_checks.append(("mamba_lr", cfg.mamba_lr))
            if "model" in master_config and "mamba_min_lr" in master_config["model"]:
                expected_checks.append(("mamba_min_lr", cfg.mamba_min_lr))
            if "model" in master_config and "mamba_weight_decay" in master_config["model"]:
                expected_checks.append(("mamba_weight_decay", cfg.mamba_weight_decay))
            if "model" in master_config and "mamba_beta1" in master_config["model"]:
                expected_checks.append(("mamba_beta1", cfg.mamba_beta1))
            if "model" in master_config and "mamba_beta2" in master_config["model"]:
                expected_checks.append(("mamba_beta2", cfg.mamba_beta2))
            if "model" in master_config and "mamba_warmup_steps" in master_config["model"]:
                expected_checks.append(("mamba_warmup_steps", cfg.mamba_warmup_steps))
            if "model" in master_config and "mamba_optimizer" in master_config["model"]:
                expected_checks.append(("mamba_optimizer", cfg.mamba_optimizer))
            if "model" in master_config and "mamba_scheduler" in master_config["model"]:
                expected_checks.append(("mamba_scheduler", cfg.mamba_scheduler))
            if "model" in master_config and "mamba_grad_clip" in master_config["model"]:
                expected_checks.append(("mamba_grad_clip", cfg.mamba_grad_clip))
            if "model" in master_config and "mamba_no_decay_params" in master_config["model"]:
                expected_checks.append(("mamba_no_decay_params", cfg.mamba_no_decay_params))
            if cfg.mamba_variant in ["film", "hybrid", "prefix", "adapter"]:
                expected_checks.append(("n_dynamic", len(cfg.varF)))
                expected_checks.append(("n_static", len(cfg.attrLst)))
            if cfg.mamba_variant == "prefix":
                expected_checks.append(("n_prefix_tokens", cfg.mamba_n_prefix_tokens))
            if cfg.mamba_variant in ["residual", "film", "hybrid", "adapter"]:
                expected_checks.append(("mamba_residual_scale_init", cfg.mamba_residual_scale_init))
            if cfg.mamba_variant == "adapter":
                expected_checks.append(("mamba_adapter_bottleneck", cfg.mamba_adapter_bottleneck))
        for key, expected_value in expected_checks:
            # Check top-level first (for nLayer, hiddenSize, nEpoch)
            if master_config.get(key) == expected_value:
                continue
            # Then check model/train sections (depending on the parameter).
            if "model" in master_config and master_config["model"].get(key) == expected_value:
                continue
            if "train" in master_config and master_config["train"].get(key) == expected_value:
                continue
            return False
    except Exception:
        return False

    return True


def get_run_results(cfg: ExperimentConfig) -> dict:
    """Read summary metrics for a completed run."""
    output_dir = cfg.output_path()
    metrics_file = os.path.join(output_dir, "basin_metrics.csv")
    params_file = os.path.join(output_dir, "total_params.txt")

    total_params = None
    if os.path.isfile(params_file):
        with open(params_file, "r") as f:
            for line in f:
                if "total_params:" in line:
                    total_params = line.split(":")[-1].strip()

    median_nse = np.nan
    median_kge = np.nan
    if os.path.isfile(metrics_file):
        try:
            df = pd.read_csv(metrics_file)
            if "NSE" in df.columns:
                median_nse = df["NSE"].median()
            if "KGE" in df.columns:
                median_kge = df["KGE"].median()
        except Exception:
            pass

    # Add variant and version info for Mamba models
    variant = ""
    mamba_version = ""
    if cfg.model_type == "mamba":
        variant = cfg.mamba_variant
        mamba_version = cfg.mamba_version

    return {
        "tier": cfg.param_tier,
        "tier_name": cfg.tier_name,
        "model": cfg.model_type.upper(),
        "hidden_size": cfg.hidden_size,
        "rho": cfg.rho,
        "params": total_params or "N/A",
        "median_nse": f"{median_nse:.4f}" if not np.isnan(median_nse) else "N/A",
        "median_kge": f"{median_kge:.4f}" if not np.isnan(median_kge) else "N/A",
        "variant": variant,
        "mamba_version": mamba_version,
        "epoch": cfg.epoch,
    }


def execute_single_run(model_type: str, tier: int, epoch: int = None, batch_size: int = None, rho: int = None, eval_only: bool = False, lstm_layers: int = None, mamba_layers: int = None, mamba_variant: str = None, mamba_version: int = None, mamba_lr: float = None, mamba_min_lr: float = None, mamba_weight_decay: float = None, mamba_scheduler: str = None, mamba_n_prefix_tokens: int = None, mamba_residual_scale_init: float = None, mamba_grad_clip: float = None, mamba_warmup_steps: int = None, mamba_no_decay_params: list = None, lr: float = None, lstm_grad_clip: float = None, lstm_weight_decay: float = None, lstm_scheduler: str = None, lstm_min_lr: float = None, mamba_adapter_bottleneck: int = None, rho_scaling_experiment: bool = False, force: bool = False, output_dir: str = None):
    """Execute a single model run (called in its own dedicated process).

    Args:
        model_type: 'lstm' or 'mamba'
        tier: Parameter tier (1-7)
        epoch: Number of training epochs (None = use config default)
        batch_size: Mini-batch size (None = use config default)
        rho: Sequence length (None = use config default)
        eval_only: If True, skip training and only run evaluation (requires existing checkpoints)
        lstm_layers: Number of LSTM layers (None = use config default)
        mamba_layers: Number of Mamba layers (None = use config default)
        mamba_variant: Mamba variant: 'standard', 'film', 'hybrid', 'prefix', 'adapter', or 'residual' (None = use config default)
        mamba_version: Mamba version: 1, 2, or 3 (None = use config default)
        mamba_lr: Mamba-specific learning rate (None = use config default)
        mamba_min_lr: Mamba-specific cosine minimum learning rate (None = use config default)
        mamba_weight_decay: Mamba-specific Adam weight decay (None = use config default)
        mamba_scheduler: Mamba scheduler: 'onecycle' or 'cosine' (None = use config default)
        mamba_n_prefix_tokens: Number of prefix tokens for prefix variant (None = use config default)
        mamba_grad_clip: Mamba gradient clipping value (None = use config default)
        mamba_warmup_steps: Mamba step-based warmup steps (None = use config default)
        mamba_no_decay_params: Mamba parameter patterns to exclude from weight decay (None = use config default)
        lr: LSTM learning rate (None = use config default)
        lstm_grad_clip: LSTM gradient clipping value (None = use config default)
        lstm_weight_decay: LSTM-specific Adam weight decay (None = use config default)
        lstm_scheduler: LSTM scheduler type (None = use config default)
        lstm_min_lr: LSTM cosine minimum learning rate (None = use config default)
        mamba_adapter_bottleneck: Bottleneck dimension for adapter variant (None = use config default)
        rho_scaling_experiment: If True, isolate outputs in the rho-scaling
            experiment family.
        force: If True, force re-run even if model checkpoints exist
        output_dir: Custom output directory name (None = use config default "output")
    """
    cfg_kwargs = {
        'model_type': model_type,
        'param_tier': tier,
        'rho_scaling_experiment': rho_scaling_experiment,
    }
    if output_dir is not None:
        cfg_kwargs['output_dir'] = output_dir
    if epoch is not None:
        cfg_kwargs['epoch'] = epoch
    if batch_size is not None:
        cfg_kwargs['batch_size'] = batch_size
    if rho is not None:
        cfg_kwargs['rho'] = rho
    if lstm_layers is not None:
        cfg_kwargs['lstm_layers'] = lstm_layers
    if mamba_layers is not None:
        cfg_kwargs['mamba_layers'] = mamba_layers
    if mamba_variant is not None:
        cfg_kwargs['mamba_variant'] = mamba_variant
    if mamba_version is not None:
        cfg_kwargs['mamba_version'] = mamba_version
    if mamba_lr is not None:
        cfg_kwargs['mamba_lr'] = mamba_lr
    if mamba_min_lr is not None:
        cfg_kwargs['mamba_min_lr'] = mamba_min_lr
    if mamba_weight_decay is not None:
        cfg_kwargs['mamba_weight_decay'] = mamba_weight_decay
    if mamba_scheduler is not None:
        cfg_kwargs['mamba_scheduler'] = mamba_scheduler
    if mamba_n_prefix_tokens is not None:
        cfg_kwargs['mamba_n_prefix_tokens'] = mamba_n_prefix_tokens
    if mamba_residual_scale_init is not None:
        cfg_kwargs['mamba_residual_scale_init'] = mamba_residual_scale_init
    if mamba_grad_clip is not None:
        cfg_kwargs['mamba_grad_clip'] = mamba_grad_clip
    if mamba_warmup_steps is not None:
        cfg_kwargs['mamba_warmup_steps'] = mamba_warmup_steps
    if mamba_no_decay_params is not None:
        cfg_kwargs['mamba_no_decay_params'] = mamba_no_decay_params
    if mamba_adapter_bottleneck is not None:
        cfg_kwargs['mamba_adapter_bottleneck'] = mamba_adapter_bottleneck
    if lr is not None:
        cfg_kwargs['lr'] = lr
    if lstm_grad_clip is not None:
        cfg_kwargs['lstm_grad_clip'] = lstm_grad_clip
    if lstm_weight_decay is not None:
        cfg_kwargs['lstm_weight_decay'] = lstm_weight_decay
    if lstm_scheduler is not None:
        cfg_kwargs['lstm_scheduler'] = lstm_scheduler
    if lstm_min_lr is not None:
        cfg_kwargs['lstm_min_lr'] = lstm_min_lr

    cfg = ExperimentConfig(**cfg_kwargs)
    set_seed(cfg)

    # Store force flag for use in training logic
    cfg._force = force

    removed_runs = cfg.remove_other_epoch_runs()
    if removed_runs:
        print(
            "Removed previous epoch output(s): "
            + ", ".join(path.name for path in removed_runs)
        )

    output_path = cfg.output_path()
    os.makedirs(output_path, exist_ok=True)

    layers_info = f" | Layers={cfg.lstm_layers}" if cfg.model_type == "lstm" else f" | Layers={cfg.mamba_layers}"
    variant_info = ""
    version_info = ""
    if cfg.model_type == "mamba":
        variant_info = f" | Variant={cfg.mamba_variant}"
        version_info = f" | Version={cfg.mamba_version}"

    # Device information
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_info = f" | Device: CUDA ({device_name})"
    else:
        device_info = " | Device: CPU"

    print(f"\n[{cfg.model_type.upper()}] Tier {cfg.param_tier} ({cfg.tier_name}) | H={cfg.hidden_size}{layers_info}{variant_info}{version_info}{device_info} | Epochs={cfg.epoch}")

    # 1. Load Data
    from data_utils import load_and_scale_data
    from training import train_model

    data = load_and_scale_data(cfg)
    train_x, train_y, train_c = data["train_x"], data["train_y"], data["train_c"]
    val_x, val_y, val_c = data["val_x"], data["val_y"], data["val_c"]
    scaler, basinarea, meanprep = data["scaler"], data["basinarea"], data["meanprep"]

    # 2. Build Model
    import json
    from collections import OrderedDict
    import hydroDL.master.master as hdl_master
    hdl_master.writeMasterFile = lambda mDict: (os.makedirs(mDict["out"], exist_ok=True), json.dump(mDict, open(os.path.join(mDict["out"], "master.json"), "w"), indent=4), mDict["out"])[2]
    hdl_master.readMasterFile = lambda out: json.load(open(os.path.join(out, "master.json"), "r"), object_pairs_hook=OrderedDict)

    from hydroDL.master import default
    from hydroDL.master.master import wrapMaster, writeMasterFile
    from hydroDL.model.crit import RmseLoss, NSELossBatch
    import hydroDL.core.logger as logger

    log = logger.get_logger("model.train_.train")
    nx = train_x.shape[-1] + train_c.shape[-1]
    ny = train_y.shape[-1]

    if cfg.model_type == "lstm":
        from CudnnLstmModel import CudnnLstmModel
        if torch.cuda.is_available():
            model = CudnnLstmModel(
                nx=nx, ny=ny, hiddenSize=cfg.hidden_size,
                nLayer=cfg.lstm_layers, dr=cfg.lstm_dropout
            ).to(cfg.device)
            opt_model = default.update(default.optLstm, hiddenSize=cfg.hidden_size)
        else:
            # HydroDL's CPU LSTM path is single-layer only. Do not silently
            # run a different architecture than the configured experiment.
            from hydroDL.model.rnn import CpuLstmModel
            if cfg.lstm_layers > 1:
                raise RuntimeError(
                    "Multi-layer LSTM experiments require CUDA; CPU execution "
                    "supports only lstm_layers=1 in HydroDL."
                )
            model = CpuLstmModel(nx=nx, ny=ny, hiddenSize=cfg.hidden_size).to(cfg.device)
            opt_model = default.update(default.optLstm, name="hydroDL.model.rnn.CpuLstmModel", hiddenSize=cfg.hidden_size)
    else:
        from mamba_model import MambaStreamflowModel, MambaStreamflowModelFiLM, MambaStreamflowModelFiLMHybrid, MambaStreamflowModelPrefixTokens, MambaStreamflowModelResidualBlocks, MambaStreamflowModelAdapter
        n_dynamic = len(cfg.varF)
        n_static = len(cfg.attrLst)

        if cfg.mamba_variant == "film":
            model = MambaStreamflowModelFiLM(
                nx=nx, ny=ny, hiddenSize=cfg.hidden_size,
                numLayers=cfg.mamba_layers, dState=cfg.mamba_d_state,
                dConv=cfg.mamba_d_conv, expand=cfg.mamba_expand,
                headdim=cfg.mamba_headdim, mambaVersion=cfg.mamba_version,
                dropout=cfg.mamba_dropout,
                n_dynamic=n_dynamic, n_static=n_static,
                residual_scale_init=cfg.mamba_residual_scale_init,
            ).to(cfg.device)
            opt_model = default.update(default.optLstm, name="mamba_model.MambaStreamflowModelFiLM", hiddenSize=cfg.hidden_size)
        elif cfg.mamba_variant == "hybrid":
            model = MambaStreamflowModelFiLMHybrid(
                nx=nx, ny=ny, hiddenSize=cfg.hidden_size,
                numLayers=cfg.mamba_layers, dState=cfg.mamba_d_state,
                dConv=cfg.mamba_d_conv, expand=cfg.mamba_expand,
                headdim=cfg.mamba_headdim, mambaVersion=cfg.mamba_version,
                dropout=cfg.mamba_dropout,
                n_dynamic=n_dynamic, n_static=n_static,
                residual_scale_init=cfg.mamba_residual_scale_init,
            ).to(cfg.device)
            opt_model = default.update(default.optLstm, name="mamba_model.MambaStreamflowModelFiLMHybrid", hiddenSize=cfg.hidden_size)
        elif cfg.mamba_variant == "prefix":
            model = MambaStreamflowModelPrefixTokens(
                nx=nx, ny=ny, hiddenSize=cfg.hidden_size,
                numLayers=cfg.mamba_layers, dState=cfg.mamba_d_state,
                dConv=cfg.mamba_d_conv, expand=cfg.mamba_expand,
                headdim=cfg.mamba_headdim, mambaVersion=cfg.mamba_version,
                dropout=cfg.mamba_dropout,
                n_dynamic=n_dynamic, n_static=n_static,
                n_prefix_tokens=cfg.mamba_n_prefix_tokens,
            ).to(cfg.device)
            opt_model = default.update(default.optLstm, name="mamba_model.MambaStreamflowModelPrefixTokens", hiddenSize=cfg.hidden_size)
        elif cfg.mamba_variant == "residual":
            model = MambaStreamflowModelResidualBlocks(
                nx=nx, ny=ny, hiddenSize=cfg.hidden_size,
                numLayers=cfg.mamba_layers, dState=cfg.mamba_d_state,
                dConv=cfg.mamba_d_conv, expand=cfg.mamba_expand,
                headdim=cfg.mamba_headdim, mambaVersion=cfg.mamba_version,
                dropout=cfg.mamba_dropout,
                residual_scale_init=cfg.mamba_residual_scale_init,
            ).to(cfg.device)
            opt_model = default.update(default.optLstm, name="mamba_model.MambaStreamflowModelResidualBlocks", hiddenSize=cfg.hidden_size)
        elif cfg.mamba_variant == "adapter":
            model = MambaStreamflowModelAdapter(
                nx=nx, ny=ny, hiddenSize=cfg.hidden_size,
                numLayers=cfg.mamba_layers, dState=cfg.mamba_d_state,
                dConv=cfg.mamba_d_conv, expand=cfg.mamba_expand,
                headdim=cfg.mamba_headdim, mambaVersion=cfg.mamba_version,
                dropout=cfg.mamba_dropout,
                n_dynamic=n_dynamic, n_static=n_static,
                bottleneck=cfg.mamba_adapter_bottleneck,
                residual_scale_init=cfg.mamba_residual_scale_init,
            ).to(cfg.device)
            opt_model = default.update(default.optLstm, name="mamba_model.MambaStreamflowModelAdapter", hiddenSize=cfg.hidden_size)
        else:
            model = MambaStreamflowModel(
                nx=nx, ny=ny, hiddenSize=cfg.hidden_size,
                numLayers=cfg.mamba_layers, dState=cfg.mamba_d_state,
                dConv=cfg.mamba_d_conv, expand=cfg.mamba_expand,
                headdim=cfg.mamba_headdim, mambaVersion=cfg.mamba_version,
                dropout=cfg.mamba_dropout,
            ).to(cfg.device)
            opt_model = default.update(default.optLstm, name="mamba_model.MambaStreamflowModel", hiddenSize=cfg.hidden_size)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,}")

    # Record parameter details
    with open(os.path.join(output_path, "total_params.txt"), "w") as f:
        f.write(f"model: {model.name}\n")
        if cfg.model_type == "mamba":
            f.write(f"version: Mamba-{cfg.mamba_version}\n")
            f.write(f"variant: {cfg.mamba_variant}\n")
            f.write(f"layers: {cfg.mamba_layers}\n")
            f.write(f"d_state: {cfg.mamba_d_state}\n")
            f.write(f"d_conv: {cfg.mamba_d_conv}\n")
            f.write(f"expand: {cfg.mamba_expand}\n")
            f.write(f"headdim: {cfg.mamba_headdim}\n")
            f.write(f"dropout: {cfg.mamba_dropout}\n")
            f.write(f"learning_rate: {cfg.mamba_lr}\n")
            f.write(f"min_learning_rate: {cfg.mamba_min_lr}\n")
            f.write(f"weight_decay: {cfg.mamba_weight_decay}\n")
            f.write(f"scheduler: {cfg.mamba_scheduler}\n")
            f.write(f"grad_clip: {cfg.mamba_grad_clip}\n")
            f.write(f"warmup_steps: {cfg.mamba_warmup_steps}\n")
            f.write(f"no_decay_params: {cfg.mamba_no_decay_params}\n")
            if cfg.mamba_variant in ["film", "hybrid", "prefix", "adapter"]:
                f.write(f"n_dynamic: {n_dynamic}\n")
                f.write(f"n_static: {n_static}\n")
            if cfg.mamba_variant == "prefix":
                f.write(f"n_prefix_tokens: {cfg.mamba_n_prefix_tokens}\n")
            if cfg.mamba_variant in ["residual", "film", "hybrid", "adapter"]:
                f.write(f"residual_scale_init: {cfg.mamba_residual_scale_init}\n")
            if cfg.mamba_variant == "adapter":
                f.write(f"adapter_bottleneck: {cfg.mamba_adapter_bottleneck}\n")
        elif cfg.model_type == "lstm":
            f.write(f"layers: {cfg.lstm_layers}\n")
            f.write(f"dropout: {cfg.lstm_dropout}\n")
            f.write(f"learning_rate: {cfg.lr}\n")

            f.write(f"min_learning_rate: {cfg.lstm_min_lr}\n")
            f.write(f"grad_clip: {cfg.lstm_grad_clip}\n")
            f.write(f"scheduler: {cfg.lstm_scheduler}\n")
        f.write(f"param_tier: {cfg.param_tier}\ntier_name: {cfg.tier_name}\n"
                f"hidden_size: {cfg.hidden_size}\ntotal_params: {total_params}\n"
                f"output_path: {cfg.save_path()}\n")

    # 3. Training configuration & Master file
    lossFun = RmseLoss() if cfg.flow_regime == 0 else NSELossBatch(np.nanstd(train_y, axis=1), device=cfg.device)
    optLoss = default.optLossRMSE if cfg.flow_regime == 0 else default.optLossNSEBatch
    opt_model = default.update(opt_model, nx=nx, ny=ny)

    # Add LSTM-specific parameters (these are recognized by hydroDL)
    if cfg.model_type == "lstm":
        opt_model = default.update(opt_model,
            nLayer=cfg.lstm_layers,
            dr=cfg.lstm_dropout,
        )

    opt_train = default.update(
        default.optTrainCamels,
        miniBatch=[cfg.batch_size, cfg.rho],
        nEpoch=cfg.epoch,
        saveEpoch=cfg.save_epoch,
        seed=cfg.seed,
        trainBuff=cfg.train_buff,
    )
    opt_all = wrapMaster(output_path, data["opt_data"], opt_model, optLoss, opt_train)

    # Ensure model section exists for direct edits (hydroDL may not add all keys via default.update)
    if "model" not in opt_all:
        opt_all["model"] = {}

    # Ensure train section exists
    if "train" not in opt_all:
        opt_all["train"] = {}

    # --- Persist ALL LSTM configuration (bypasses hydroDL's silent key filtering) ---
    if cfg.model_type == "lstm":
        opt_all["model"]["nLayer"] = cfg.lstm_layers
        opt_all["model"]["dr"] = cfg.lstm_dropout
        opt_all["model"]["lstm_layers"] = cfg.lstm_layers
        opt_all["model"]["lstm_dropout"] = cfg.lstm_dropout
        opt_all["model"]["lr"] = cfg.lr

        opt_all["model"]["scheduler"] = cfg.lstm_scheduler
        opt_all["model"]["lstm_grad_clip"] = cfg.lstm_grad_clip
        opt_all["model"]["lstm_weight_decay"] = cfg.lstm_weight_decay

    # --- Persist ALL Mamba configuration ---
    if cfg.model_type == "mamba":
        opt_all["model"]["mamba_version"] = cfg.mamba_version
        opt_all["model"]["mamba_variant"] = cfg.mamba_variant
        opt_all["model"]["mamba_layers"] = cfg.mamba_layers
        opt_all["model"]["mamba_d_state"] = cfg.mamba_d_state
        opt_all["model"]["mamba_d_conv"] = cfg.mamba_d_conv
        opt_all["model"]["mamba_expand"] = cfg.mamba_expand
        opt_all["model"]["mamba_headdim"] = cfg.mamba_headdim
        opt_all["model"]["mamba_dropout"] = cfg.mamba_dropout
        opt_all["model"]["mamba_lr"] = cfg.mamba_lr
        opt_all["model"]["mamba_min_lr"] = cfg.mamba_min_lr
        opt_all["model"]["mamba_weight_decay"] = cfg.mamba_weight_decay
        opt_all["model"]["mamba_beta1"] = cfg.mamba_beta1
        opt_all["model"]["mamba_beta2"] = cfg.mamba_beta2
        opt_all["model"]["mamba_warmup_steps"] = cfg.mamba_warmup_steps
        opt_all["model"]["mamba_optimizer"] = cfg.mamba_optimizer
        opt_all["model"]["mamba_scheduler"] = cfg.mamba_scheduler
        opt_all["model"]["mamba_grad_clip"] = cfg.mamba_grad_clip
        opt_all["model"]["mamba_no_decay_params"] = cfg.mamba_no_decay_params
        if cfg.mamba_variant in ["film", "hybrid", "prefix", "adapter"]:
            opt_all["model"]["n_dynamic"] = n_dynamic
            opt_all["model"]["n_static"] = n_static
        if cfg.mamba_variant == "prefix":
            opt_all["model"]["n_prefix_tokens"] = cfg.mamba_n_prefix_tokens
        if cfg.mamba_variant in ["residual", "film", "hybrid", "adapter"]:
            opt_all["model"]["mamba_residual_scale_init"] = cfg.mamba_residual_scale_init
        if cfg.mamba_variant == "adapter":
            opt_all["model"]["mamba_adapter_bottleneck"] = cfg.mamba_adapter_bottleneck
    else:
        opt_all["model"]["lstm_optimizer"] = cfg.lstm_optimizer
        opt_all["model"]["lstm_weight_decay"] = cfg.lstm_weight_decay
        opt_all["model"]["lstm_beta1"] = cfg.lstm_beta1
        opt_all["model"]["lstm_beta2"] = cfg.lstm_beta2
        opt_all["model"]["lstm_scheduler"] = cfg.lstm_scheduler
        opt_all["model"]["lstm_min_lr"] = cfg.lstm_min_lr

    # --- Persist common training config ---
    opt_all["train"]["flow_regime"] = cfg.flow_regime
    opt_all["train"]["loss_name"] = "RmseLoss" if cfg.flow_regime == 0 else "NSELossBatch"
    opt_all["train"]["lr"] = cfg.lr if cfg.model_type == "lstm" else cfg.mamba_lr
    if cfg.model_type == "mamba":
        opt_all["train"]["max_lr"] = cfg.mamba_lr  # Peak LR for Mamba cosine mode
    if cfg.model_type == "mamba":
        opt_all["train"]["min_lr"] = cfg.mamba_min_lr
        opt_all["train"]["weight_decay"] = cfg.mamba_weight_decay
    if cfg.model_type == "lstm":
        opt_all["train"]["min_lr"] = cfg.lstm_min_lr if cfg.lstm_scheduler == "cosine" else None
        opt_all["train"]["weight_decay"] = cfg.lstm_weight_decay
    opt_all["train"]["scheduler"] = cfg.lstm_scheduler if cfg.model_type == "lstm" else cfg.mamba_scheduler
    opt_all["train"]["valid_batch"] = cfg.valid_batch

    # --- Persist experiment-level metadata (identity, tier, dataset, time ranges) ---
    opt_all["experiment"] = {
        "model_type": cfg.model_type,
        "param_tier": cfg.param_tier,
        "tier_name": cfg.tier_name,
        "hidden_size": cfg.hidden_size,
        "rho": cfg.rho,
        "batch_size": cfg.batch_size,
        "rho_scaling_experiment": cfg.rho_scaling_experiment,
        "seed": cfg.seed,
        "data_opt": cfg.data_opt,
        "for_type": cfg.for_type,
        "t_train": cfg.t_train,
        "t_valid": cfg.t_valid,
        "subset_train": cfg.subset_train,
        "train_buff": cfg.train_buff,
        "save_path": cfg.save_path(),
    }

    writeMasterFile(opt_all)

    # 4. Train (skip if already completed or eval_only mode)
    model_file = os.path.join(output_path, f"model_Ep{cfg.epoch}.pt")
    # The final checkpoint is the authoritative training-completion marker.
    # training_loss.csv is incremental and can be incomplete after an
    # interrupted post-processing step without invalidating the checkpoint.
    training_completed = os.path.isfile(model_file)

    if eval_only:
        print(f"[EVAL-ONLY] Skipping training, loading model from {model_file}")
        if not os.path.isfile(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}. Cannot run eval-only mode.")
        model = torch.load(model_file, weights_only=False)
    elif training_completed and not cfg._force:
        print(f"[SKIP] Training already completed. Loading model from {model_file}")
        model = torch.load(model_file, weights_only=False)
    else:
        print(f"[TRAIN] Starting training...")
        t0 = time.time()
        # Use model-specific learning rates and scheduler
        if cfg.model_type == "mamba":
            if cfg.mamba_scheduler == "cosine":
                # For cosine: lr is the peak learning rate
                train_lr = cfg.mamba_lr  # Peak LR for cosine
                train_max_lr = cfg.mamba_lr  # Same as lr for cosine
                train_scheduler_type = "cosine"
                train_min_lr = cfg.mamba_min_lr
                train_warmup_steps = cfg.mamba_warmup_steps
                train_warmup_epochs = 0
            else:  # onecycle
                # For onecycle: use lr as base, max_lr as peak
                train_lr = cfg.mamba_lr
                train_max_lr = cfg.mamba_lr * 2.5  # Default peak for onecycle if needed
                train_scheduler_type = "onecycle"
                train_min_lr = None
                train_warmup_steps = 0
                train_warmup_epochs = 0
            train_grad_clip = cfg.mamba_grad_clip
            train_no_decay_params = cfg.mamba_no_decay_params
        else:
            train_lr = cfg.lr
            train_max_lr = cfg.lr
            train_scheduler_type = cfg.lstm_scheduler
            train_min_lr = cfg.lstm_min_lr if cfg.lstm_scheduler == "cosine" else None
            train_warmup_steps = 0
            train_warmup_epochs = 0
            train_grad_clip = cfg.lstm_grad_clip
            train_no_decay_params = None

        model = train_model(
            model, train_x, train_y, train_c, lossFun,
            n_epoch=cfg.epoch, mini_batch=[cfg.batch_size, cfg.rho],
            save_epoch=cfg.save_epoch, save_folder=output_path,
            bufftime=cfg.train_buff, lr=train_lr, max_lr=train_max_lr,
            scheduler_type=train_scheduler_type, min_lr=train_min_lr,
            weight_decay=(cfg.mamba_weight_decay if cfg.model_type == "mamba" else cfg.lstm_weight_decay),
            optimizer_type=(cfg.mamba_optimizer if cfg.model_type == "mamba" else cfg.lstm_optimizer),
            betas=((cfg.mamba_beta1, cfg.mamba_beta2) if cfg.model_type == "mamba" else (cfg.lstm_beta1, cfg.lstm_beta2)),
            warmup_epochs=train_warmup_epochs,
            warmup_steps=train_warmup_steps,
            grad_clip=train_grad_clip,
            no_decay_params=train_no_decay_params,
            use_amp=cfg.use_amp,
            log=log,
        )

    # 5. Evaluate (skip if evaluation outputs already exist)
    from evaluation import evaluate_final, evaluate_all_checkpoints
    if is_evaluation_completed(cfg) and not cfg._force:
        print(f"[SKIP] Evaluation already completed. "
              f"Loading metrics from {os.path.join(cfg.output_path(), 'basin_metrics.csv')}")
    else:
        print(f"[EVAL] Starting evaluation...")
        evaluate_final(cfg, val_x, val_y, val_c, train_x, scaler, basinarea, meanprep)

        if cfg.epoch >= cfg.save_epoch:
            evaluate_all_checkpoints(cfg, val_x, val_y, val_c, train_x, scaler, basinarea, meanprep)


def print_summary_table(results_list):
    """Print an aligned, non-wrapping summary table of completed tiers."""
    if not results_list:
        return
    header = (
        f"{'Tier':<5} {'Name':<10} {'Model':<6} {'Params':<12} "
        f"{'Hidden':<7} {'Time':<8} {'Med NSE':<9} {'Med KGE':<9} {'Status'}"
    )
    sep = "-" * len(header)
    print("\n" + "=" * len(header))
    print("CAMELS BENCHMARK RESULTS SUMMARY".center(len(header)))
    print("=" * len(header))
    print(header)
    print(sep)
    for r in results_list:
        p_raw = str(r.get("params", "N/A")).replace(",", "")
        p_str = f"{int(p_raw):,}" if p_raw.isdigit() else str(r.get("params", "N/A"))
        t_min = str(r.get("train_time_min", "0.0"))
        t_str = f"{t_min}m" if not t_min.endswith("m") else t_min
        print(
            f"{str(r.get('tier')):<5} "
            f"{str(r.get('tier_name')):<10} "
            f"{str(r.get('model')):<6} "
            f"{p_str:<12} "
            f"{str(r.get('hidden_size')):<7} "
            f"{t_str:<8} "
            f"{str(r.get('median_nse')):<9} "
            f"{str(r.get('median_kge')):<9} "
            f"{str(r.get('status'))}"
        )
    print("=" * len(header) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Sequential Runner: Train all tiers of LSTM and Mamba on CAMELS"
    )
    parser.add_argument(
        "--tiers", type=int, nargs="+", choices=tuple(range(1, 4)), default=None,
        help="List of tiers to run (1 to 3: 200K, 500K, 1M). Default: all 3 tiers",
    )
    parser.add_argument(
        "--model", "--models", dest="models", type=str, nargs="+", default=None,
        help="Model(s) to run: 'lstm', 'mamba', or 'all'. Default: from config",
    )
    parser.add_argument(
        "--only-lstm", action="store_true", default=False,
        help="Convenience shortcut: Run ONLY LSTM across tiers",
    )
    parser.add_argument(
        "--only-mamba", action="store_true", default=False,
        help="Convenience shortcut: Run ONLY Mamba across tiers",
    )
    parser.add_argument(
        "--lstm-layers", type=int, nargs="+", default=None,
        help="Number of LSTM layers (default: from config; can specify multiple e.g., --lstm-layers 1 2 3)",
    )
    parser.add_argument(
        "--mamba-layers", type=int, nargs="+", default=None,
        help="Number of Mamba layers (default: from config; can specify multiple e.g., --mamba-layers 1 2 3)",
    )
    parser.add_argument(
        "--mamba-variant", type=str,
        choices=("standard", "film", "hybrid", "prefix", "residual", "adapter"), default=None,
        help="Mamba variant (default: from config)",
    )
    parser.add_argument(
        "--mamba-version", type=int, choices=(1, 2, 3), default=None,
        help="Mamba version: 1, 2, or 3 (default: from config)",
    )
    parser.add_argument(
        "--mamba-lr", type=float, default=None,
        help="Mamba-specific learning rate (default: from config)",
    )
    parser.add_argument(
        "--mamba-min-lr", type=float, default=None,
        help="Mamba-specific cosine minimum learning rate (default: from config)",
    )
    parser.add_argument(
        "--mamba-weight-decay", type=float, default=None,
        help="Mamba-specific Adam weight decay (default: from config)",
    )
    parser.add_argument(
        "--mamba-scheduler", type=str,
        choices=("onecycle", "cosine"), default=None,
        help="Mamba scheduler (default: from config)",
    )
    parser.add_argument(
        "--mamba-n-prefix-tokens", type=int, default=None,
        help="Number of prefix tokens for prefix variant (default: from config)",
    )
    parser.add_argument(
        "--mamba-residual-scale-init", type=float, default=None,
        help="Initial residual scale for residual blocks variant (default: from config)",
    )
    parser.add_argument(
        "--mamba-adapter-bottleneck", type=int, default=None,
        help="Bottleneck dimension for adapter variant (default: from config)",
    )
    parser.add_argument(
        "--mamba-grad-clip", type=float, default=None,
        help="Mamba gradient clipping value (default: from config)",
    )
    parser.add_argument(
        "--mamba-warmup-steps", type=int, default=None,
        help="Mamba step-based warmup steps (default: from config)",
    )
    parser.add_argument(
        "--lstm-grad-clip", type=float, default=None,
        help="LSTM gradient clipping value (default: from config)",
    )
    parser.add_argument(
        "--lstm-weight-decay", type=float, default=None,
        help="LSTM-specific Adam weight decay (default: from config)",
    )
    parser.add_argument(
        "--lstm-scheduler", type=str,
        choices=("onecycle", "cosine"), default=None,
        help="LSTM scheduler (default: from config)",
    )
    parser.add_argument(
        "--lstm-lr", type=float, default=None,
        help="LSTM-specific learning rate (default: from config)",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="LSTM learning rate (default: from config)",
    )
    parser.add_argument(
        "--lstm-min-lr", type=float, default=None,
        help="LSTM cosine minimum learning rate (default: from config)",
    )
    parser.add_argument(
        "--epoch", type=int, default=None,
        help="Number of epochs per model (default: from config)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Mini-batch size (default: from config)",
    )
    parser.add_argument(
        "--rho", type=int, default=None,
        help="Sequence length lookback days (default: from config). "
             "Single value; use --rhos for a sweep.",
    )
    parser.add_argument(
        "--rhos", type=int, nargs="+", default=None,
        help="Sequence lengths to sweep, e.g. --rhos 365 730 1095. "
             "Runs every tier x model x rho. Overrides --rho.",
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Force re-run even if basin_metrics.csv exists (default: False)",
    )
    parser.add_argument(
        "--eval-only", action="store_true", default=False,
        help="Skip training and only run evaluation (assumes model checkpoints exist)",
    )
    parser.add_argument(
        "--in-process", action="store_true", default=False,
        help="Run in the same Python process instead of isolated subprocesses",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="Custom output directory name (default: 'output')",
    )
    # Internal flag for worker subprocess
    parser.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--single-model", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--single-tier", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--single-eval-only", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--single-lstm-layers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-mamba-layers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-epoch", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-batch-size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-rho", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-force", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--single-rho-scaling", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--single-mamba-variant", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-mamba-version", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-mamba-lr", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-mamba-min-lr", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-mamba-weight-decay", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-mamba-scheduler", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-mamba-n-prefix-tokens", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-mamba-residual-scale-init", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-mamba-adapter-bottleneck", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-mamba-grad-clip", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-mamba-warmup-steps", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-lstm-grad-clip", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-lstm-weight-decay", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-lstm-scheduler", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-lstm-lr", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-lr", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-lstm-min-lr", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-output-dir", type=str, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Normalize LSTM learning rate arguments (use --lstm-lr if --lr is not provided)
    if args.lstm_lr is not None and args.lr is None:
        args.lr = args.lstm_lr
    elif args.lr is not None and args.lstm_lr is None:
        args.lstm_lr = args.lr

    # -----------------------------------------------------------------------
    # Automatic all-depth orchestration
    # -----------------------------------------------------------------------
    # When both models are requested without an explicit depth, run the full
    # controlled 1L/2L/3L experiment. Each recursive invocation enters the
    # normal tier loop below with one matching depth for both architectures.
    requested_models = {model.lower() for model in (args.models or [])}
    run_both_models = (
        not args.only_lstm
        and not args.only_mamba
        and (
            args.models is None
            or "all" in requested_models
            or {"lstm", "mamba"}.issubset(requested_models)
        )
    )

    # If both models are selected and only one depth is supplied, use that
    # depth for both models so the comparison remains controlled.
    if run_both_models and (args.lstm_layers is None) != (args.mamba_layers is None):
        selected_depth = args.lstm_layers if args.lstm_layers is not None else args.mamba_layers
        args.lstm_layers = selected_depth
        args.mamba_layers = selected_depth

    # Convert single values to lists for uniform processing
    if args.lstm_layers is not None and not isinstance(args.lstm_layers, (list, tuple)):
        args.lstm_layers = [args.lstm_layers]
    if args.mamba_layers is not None and not isinstance(args.mamba_layers, (list, tuple)):
        args.mamba_layers = [args.mamba_layers]

    # Handle multiple layer specifications for single model
    # Check if user specified multiple layers for a single model
    lstm_multiple_layers = args.lstm_layers is not None and isinstance(args.lstm_layers, list) and len(args.lstm_layers) > 1
    mamba_multiple_layers = args.mamba_layers is not None and isinstance(args.mamba_layers, list) and len(args.mamba_layers) > 1
    
    if not args.single and (lstm_multiple_layers or mamba_multiple_layers):
        exit_code = 0
        
        # Determine which model(s) to iterate layers for
        models_to_iterate = []
        if lstm_multiple_layers:
            models_to_iterate.append(("lstm", args.lstm_layers))
        if mamba_multiple_layers:
            models_to_iterate.append(("mamba", args.mamba_layers))
        
        for model_type, layers_list in models_to_iterate:
            for layer_depth in layers_list:
                command = [
                    sys.executable,
                    str(SCRIPT_DIR / "run_all_tiers.py"),
                    "--model", model_type,
                    f"--{model_type}-layers", str(layer_depth),
                ]
                if args.tiers is not None:
                    command.extend(["--tiers", *[str(tier) for tier in args.tiers]])
                if args.epoch is not None:
                    command.extend(["--epoch", str(args.epoch)])
                if args.batch_size is not None:
                    command.extend(["--batch_size", str(args.batch_size)])
                if args.rhos is not None:
                    command.extend(["--rhos", *[str(r) for r in args.rhos]])
                elif args.rho is not None:
                    command.extend(["--rho", str(args.rho)])
                if args.force:
                    command.append("--force")
                if args.eval_only:
                    command.append("--eval-only")
                if args.in_process:
                    command.append("--in-process")
                
                # Add model-specific parameters
                if model_type == "mamba":
                    if args.mamba_variant is not None:
                        command.extend(["--mamba-variant", args.mamba_variant])
                    if args.mamba_version is not None:
                        command.extend(["--mamba-version", str(args.mamba_version)])
                    if args.mamba_lr is not None:
                        command.extend(["--mamba-lr", str(args.mamba_lr)])
                    if args.mamba_min_lr is not None:
                        command.extend(["--mamba-min-lr", str(args.mamba_min_lr)])
                    if args.mamba_weight_decay is not None:
                        command.extend(["--mamba-weight-decay", str(args.mamba_weight_decay)])
                    if args.mamba_scheduler is not None:
                        command.extend(["--mamba-scheduler", args.mamba_scheduler])
                    if args.mamba_n_prefix_tokens is not None:
                        command.extend(["--mamba-n-prefix-tokens", str(args.mamba_n_prefix_tokens)])
                    if args.mamba_residual_scale_init is not None:
                        command.extend(["--mamba-residual-scale-init", str(args.mamba_residual_scale_init)])
                    if args.mamba_adapter_bottleneck is not None:
                        command.extend(["--mamba-adapter-bottleneck", str(args.mamba_adapter_bottleneck)])
                    if args.mamba_grad_clip is not None:
                        command.extend(["--mamba-grad-clip", str(args.mamba_grad_clip)])
                    if args.mamba_warmup_steps is not None:
                        command.extend(["--mamba-warmup-steps", str(args.mamba_warmup_steps)])
                elif model_type == "lstm":
                    if args.lstm_lr is not None:
                        command.extend(["--lstm-lr", str(args.lstm_lr)])
                    if args.lstm_grad_clip is not None:
                        command.extend(["--lstm-grad-clip", str(args.lstm_grad_clip)])
                    if args.lstm_weight_decay is not None:
                        command.extend(["--lstm-weight-decay", str(args.lstm_weight_decay)])
                    if args.lstm_scheduler is not None:
                        command.extend(["--lstm-scheduler", args.lstm_scheduler])
                    if args.lstm_min_lr is not None:
                        command.extend(["--lstm-min-lr", str(args.lstm_min_lr)])
                if args.output_dir is not None:
                    command.extend(["--output-dir", args.output_dir])

                print(f"\\n{'=' * 80}\\nStarting {model_type.upper()} {layer_depth}L experiment\\n{'=' * 80}")
                result = subprocess.run(command, check=False)
                if result.returncode != 0:
                    exit_code = result.returncode
                    print(f"[ERROR] {model_type.upper()} {layer_depth}L failed with code {result.returncode}.")
                    break
            if exit_code != 0:
                break
        return exit_code

    # Convert single values to uniform processing (for the main loop)
    if args.lstm_layers is not None and isinstance(args.lstm_layers, list) and len(args.lstm_layers) == 1:
        args.lstm_layers = args.lstm_layers[0]
    if args.mamba_layers is not None and isinstance(args.mamba_layers, list) and len(args.mamba_layers) == 1:
        args.mamba_layers = args.mamba_layers[0]

    if (
        not args.single
        and run_both_models
        and args.lstm_layers is None
        and args.mamba_layers is None
    ):
        print("No layer specified: running all controlled depths (1, 2, and 3).")
        exit_code = 0
        for depth in (1, 2, 3):
            command = [
                sys.executable,
                str(SCRIPT_DIR / "run_all_tiers.py"),
                "--model", "lstm", "mamba",
                "--lstm-layers", str(depth),
                "--mamba-layers", str(depth),
            ]
            if args.tiers is not None:
                command.extend(["--tiers", *[str(tier) for tier in args.tiers]])
            if args.epoch is not None:
                command.extend(["--epoch", str(args.epoch)])
            if args.batch_size is not None:
                command.extend(["--batch_size", str(args.batch_size)])
            if args.rhos is not None:
                command.extend(["--rhos", *[str(r) for r in args.rhos]])
            elif args.rho is not None:
                command.extend(["--rho", str(args.rho)])
            if args.mamba_variant is not None:
                command.extend(["--mamba-variant", args.mamba_variant])
            if args.mamba_version is not None:
                command.extend(["--mamba-version", str(args.mamba_version)])
            if args.mamba_lr is not None:
                command.extend(["--mamba-lr", str(args.mamba_lr)])
            if args.mamba_min_lr is not None:
                command.extend(["--mamba-min-lr", str(args.mamba_min_lr)])
            if args.mamba_weight_decay is not None:
                command.extend(["--mamba-weight-decay", str(args.mamba_weight_decay)])
            if args.mamba_scheduler is not None:
                command.extend(["--mamba-scheduler", args.mamba_scheduler])
            if args.mamba_n_prefix_tokens is not None:
                command.extend(["--mamba-n-prefix-tokens", str(args.mamba_n_prefix_tokens)])
            if args.mamba_residual_scale_init is not None:
                command.extend(["--mamba-residual-scale-init", str(args.mamba_residual_scale_init)])
            if args.mamba_adapter_bottleneck is not None:
                command.extend(["--mamba-adapter-bottleneck", str(args.mamba_adapter_bottleneck)])
            if args.mamba_grad_clip is not None:
                command.extend(["--mamba-grad-clip", str(args.mamba_grad_clip)])
            if args.mamba_warmup_steps is not None:
                command.extend(["--mamba-warmup-steps", str(args.mamba_warmup_steps)])
            if args.lstm_lr is not None:
                command.extend(["--lstm-lr", str(args.lstm_lr)])
            if args.lstm_grad_clip is not None:
                command.extend(["--lstm-grad-clip", str(args.lstm_grad_clip)])
            if args.lstm_weight_decay is not None:
                command.extend(["--lstm-weight-decay", str(args.lstm_weight_decay)])
            if args.lstm_scheduler is not None:
                command.extend(["--lstm-scheduler", args.lstm_scheduler])
            if args.lstm_min_lr is not None:
                command.extend(["--lstm-min-lr", str(args.lstm_min_lr)])
            if args.output_dir is not None:
                command.extend(["--output-dir", args.output_dir])
            if args.force:
                command.append("--force")
            if args.eval_only:
                command.append("--eval-only")
            if args.in_process:
                command.append("--in-process")

            print(f"\\n{'=' * 80}\\nStarting depth {depth} experiment\\n{'=' * 80}")
            print(f"Running both LSTM and Mamba with {depth} layer(s) across specified tiers")
            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                exit_code = result.returncode
                print(f"[ERROR] Depth {depth} failed with code {result.returncode}.")
                break
        return exit_code

    # -----------------------------------------------------------------------
    # Handle multiple layer specifications for single model
    # -----------------------------------------------------------------------
    # If user specifies multiple layers for a single model, iterate through them
    if not args.single:
        lstm_multiple_layers = args.lstm_layers is not None and isinstance(args.lstm_layers, list) and len(args.lstm_layers) > 1
        mamba_multiple_layers = args.mamba_layers is not None and isinstance(args.mamba_layers, list) and len(args.mamba_layers) > 1
        
        if lstm_multiple_layers or mamba_multiple_layers:
            print("=" * 80)
            print("MULTI-LAYER EXPERIMENT DETECTED".center(80))
            print("=" * 80)
            if lstm_multiple_layers:
                print(f"LSTM Layers to Run: {args.lstm_layers}")
            if mamba_multiple_layers:
                print(f"Mamba Layers to Run: {args.mamba_layers}")
            print("=" * 80)
        
        if lstm_multiple_layers or mamba_multiple_layers:
            exit_code = 0
            
            # Determine which model(s) to iterate layers for
            models_to_iterate = []
            if lstm_multiple_layers:
                models_to_iterate.append(("lstm", args.lstm_layers))
            if mamba_multiple_layers:
                models_to_iterate.append(("mamba", args.mamba_layers))
            
            for model_type, layers_list in models_to_iterate:
                for layer_depth in layers_list:
                    command = [
                        sys.executable,
                        str(SCRIPT_DIR / "run_all_tiers.py"),
                        "--model", model_type,
                        f"--{model_type}-layers", str(layer_depth),
                    ]
                    if args.tiers is not None:
                        command.extend(["--tiers", *[str(tier) for tier in args.tiers]])
                    if args.epoch is not None:
                        command.extend(["--epoch", str(args.epoch)])
                    if args.batch_size is not None:
                        command.extend(["--batch_size", str(args.batch_size)])
                    if args.rhos is not None:
                        command.extend(["--rhos", *[str(r) for r in args.rhos]])
                    elif args.rho is not None:
                        command.extend(["--rho", str(args.rho)])
                    if args.force:
                        command.append("--force")
                    if args.eval_only:
                        command.append("--eval-only")
                    if args.in_process:
                        command.append("--in-process")
                    
                    # Add model-specific parameters
                    if model_type == "mamba":
                        if args.mamba_variant is not None:
                            command.extend(["--mamba-variant", args.mamba_variant])
                        if args.mamba_version is not None:
                            command.extend(["--mamba-version", str(args.mamba_version)])
                        if args.mamba_lr is not None:
                            command.extend(["--mamba-lr", str(args.mamba_lr)])
                        if args.mamba_min_lr is not None:
                            command.extend(["--mamba-min-lr", str(args.mamba_min_lr)])
                        if args.mamba_weight_decay is not None:
                            command.extend(["--mamba-weight-decay", str(args.mamba_weight_decay)])
                        if args.mamba_scheduler is not None:
                            command.extend(["--mamba-scheduler", args.mamba_scheduler])
                        if args.mamba_n_prefix_tokens is not None:
                            command.extend(["--mamba-n-prefix-tokens", str(args.mamba_n_prefix_tokens)])
                        if args.mamba_residual_scale_init is not None:
                            command.extend(["--mamba-residual-scale-init", str(args.mamba_residual_scale_init)])
                        if args.mamba_adapter_bottleneck is not None:
                            command.extend(["--mamba-adapter-bottleneck", str(args.mamba_adapter_bottleneck)])
                        if args.mamba_grad_clip is not None:
                            command.extend(["--mamba-grad-clip", str(args.mamba_grad_clip)])
                        if args.mamba_warmup_steps is not None:
                            command.extend(["--mamba-warmup-steps", str(args.mamba_warmup_steps)])
                    elif model_type == "lstm":
                        if args.lstm_lr is not None:
                            command.extend(["--lstm-lr", str(args.lstm_lr)])
                        if args.lstm_grad_clip is not None:
                            command.extend(["--lstm-grad-clip", str(args.lstm_grad_clip)])
                        if args.lstm_weight_decay is not None:
                            command.extend(["--lstm-weight-decay", str(args.lstm_weight_decay)])
                        if args.lstm_scheduler is not None:
                            command.extend(["--lstm-scheduler", args.lstm_scheduler])
                        if args.lstm_min_lr is not None:
                            command.extend(["--lstm-min-lr", str(args.lstm_min_lr)])
                    if args.output_dir is not None:
                        command.extend(["--output-dir", args.output_dir])

                    print(f"\\n{'=' * 80}\\nStarting {model_type.upper()} {layer_depth}L experiment\\n{'=' * 80}")
                    print(f"Running {model_type.upper()} with {layer_depth} layer(s) across specified tiers")
                    result = subprocess.run(command, check=False)
                    if result.returncode != 0:
                        exit_code = result.returncode
                        print(f"[ERROR] {model_type.upper()} {layer_depth}L failed with code {result.returncode}.")
                        break
                if exit_code != 0:
                    break
            return exit_code

    # -----------------------------------------------------------------------
    # Subprocess execution worker
    # -----------------------------------------------------------------------
    if args.single:
        cfg_kwargs = {
            'model_type': args.single_model,
            'tier': args.single_tier,
            'eval_only': getattr(args, 'single_eval_only', False),
            'force': getattr(args, 'single_force', False),
            'rho_scaling_experiment': getattr(args, 'single_rho_scaling', False),
            'mamba_variant': getattr(args, 'single_mamba_variant', None),
            'mamba_version': getattr(args, 'single_mamba_version', None),
            'mamba_lr': getattr(args, 'single_mamba_lr', None),
            'mamba_min_lr': getattr(args, 'single_mamba_min_lr', None),
            'mamba_weight_decay': getattr(args, 'single_mamba_weight_decay', None),
            'mamba_scheduler': getattr(args, 'single_mamba_scheduler', None),
            'mamba_n_prefix_tokens': getattr(args, 'single_mamba_n_prefix_tokens', None),
            'mamba_residual_scale_init': getattr(args, 'single_mamba_residual_scale_init', None),
            'mamba_adapter_bottleneck': getattr(args, 'single_mamba_adapter_bottleneck', None),
            'mamba_grad_clip': getattr(args, 'single_mamba_grad_clip', None),
            'mamba_warmup_steps': getattr(args, 'single_mamba_warmup_steps', None),
            'lr': getattr(args, 'single_lstm_lr', None) or getattr(args, 'single_lr', None),
            'lstm_grad_clip': getattr(args, 'single_lstm_grad_clip', None),
            'lstm_weight_decay': getattr(args, 'single_lstm_weight_decay', None),
            'lstm_scheduler': getattr(args, 'single_lstm_scheduler', None),
            'lstm_min_lr': getattr(args, 'single_lstm_min_lr', None),
            'output_dir': getattr(args, 'single_output_dir', None),
        }
        if hasattr(args, 'single_epoch') and args.single_epoch is not None:
            cfg_kwargs['epoch'] = args.single_epoch
        if hasattr(args, 'single_batch_size') and args.single_batch_size is not None:
            cfg_kwargs['batch_size'] = args.single_batch_size
        if hasattr(args, 'single_rho') and args.single_rho is not None:
            cfg_kwargs['rho'] = args.single_rho
        if hasattr(args, 'single_lstm_layers') and args.single_lstm_layers is not None:
            cfg_kwargs['lstm_layers'] = args.single_lstm_layers
        if hasattr(args, 'single_mamba_layers') and args.single_mamba_layers is not None:
            cfg_kwargs['mamba_layers'] = args.single_mamba_layers

        execute_single_run(**cfg_kwargs)
        return

    # -----------------------------------------------------------------------
    # Resolve models to run
    # -----------------------------------------------------------------------
    if args.only_lstm:
        models_to_run = ["lstm"]
    elif args.only_mamba:
        models_to_run = ["mamba"]
    elif args.models is None:
        # Use default from config if not specified
        models_to_run = ["lstm", "mamba"]
    else:
        raw_models = [m.lower() for m in args.models]
        if "all" in raw_models or ("lstm" in raw_models and "mamba" in raw_models):
            models_to_run = ["lstm", "mamba"]
        elif "lstm" in raw_models:
            models_to_run = ["lstm"]
        elif "mamba" in raw_models:
            models_to_run = ["mamba"]
        else:
            raise ValueError(
                f"Unknown model choice: {args.models}. Expected 'lstm', 'mamba', or 'all'."
            )

    # Use model-specific logs when a specific model is selected (enables parallel execution)
    # Only use generic running_log.csv when running "all" models in a single process
    use_model_specific_logs = len(models_to_run) == 1

    # -----------------------------------------------------------------------
    # Orchestrator: Loop through tiers, 1 tier of selected model(s) at once
    # -----------------------------------------------------------------------
    # Use default tiers if not specified
    if args.tiers is None:
        tiers_to_run = [1, 2, 3]
    else:
        tiers_to_run = sorted(args.tiers)

    # Resolve rho sweep: --rhos wins over --rho; None = config default (single run)
    if args.rhos is not None:
        rhos_to_run: list = list(args.rhos)
    elif args.rho is not None:
        rhos_to_run = [args.rho]
    else:
        rhos_to_run = [None]

    print("=" * 80)
    print("CAMELS MODEL BENCHMARK ORCHESTRATOR".center(80))
    print("=" * 80)
    print(f"Tiers to Run     : {tiers_to_run}")
    print(f"Models per Tier  : {models_to_run}")
    print(f"Rhos to Run      : {rhos_to_run} (days; None = config default)")

    # Device information
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"Device          : CUDA ({device_name})")
    else:
        print(f"Device          : CPU")

    # Get default config values for display - use appropriate model type to avoid warnings
    if len(models_to_run) == 1 and models_to_run[0] == "mamba":
        default_cfg = ExperimentConfig(
            model_type="mamba",
            mamba_layers=args.mamba_layers or 3,
        )
    else:
        default_cfg = ExperimentConfig(
            model_type="lstm",
            **({"lstm_layers": args.lstm_layers} if args.lstm_layers is not None else {}),
        )

    print(f"Epochs per Model : {args.epoch if args.epoch is not None else default_cfg.epoch}")
    print(f"Batch Size      : {args.batch_size if args.batch_size is not None else default_cfg.batch_size}")
    rho_display = rhos_to_run if len(rhos_to_run) > 1 or rhos_to_run[0] is not None else default_cfg.rho
    print(f"Lookback (Rho)   : {rho_display} days")

    # Model-specific configuration display
    if "lstm" in models_to_run:
        lstm_layers_display = args.lstm_layers if args.lstm_layers is not None else default_cfg.lstm_layers
        if isinstance(lstm_layers_display, list) and len(lstm_layers_display) > 1:
            print(f"LSTM Layers     : {lstm_layers_display} (multi-layer experiment)")
        elif isinstance(lstm_layers_display, list) and len(lstm_layers_display) == 1:
            print(f"LSTM Layers     : {lstm_layers_display[0]} (single layer from multi-layer spec)")
        else:
            print(f"LSTM Layers     : {lstm_layers_display}")
        lstm_lr_display = getattr(args, 'lr', None)
        lstm_lr_display = lstm_lr_display if lstm_lr_display is not None else default_cfg.lr

        lstm_grad_clip_display = getattr(args, 'lstm_grad_clip', None)
        lstm_grad_clip_display = lstm_grad_clip_display if lstm_grad_clip_display is not None else default_cfg.lstm_grad_clip
        lstm_scheduler_display = getattr(args, 'lstm_scheduler', None)
        lstm_scheduler_display = lstm_scheduler_display if lstm_scheduler_display is not None else default_cfg.lstm_scheduler
        lstm_min_lr_display = args.lstm_min_lr if args.lstm_min_lr is not None else default_cfg.lstm_min_lr
        print(f"LSTM LR         : {lstm_lr_display} (min: {lstm_min_lr_display})")
        print(f"LSTM Grad Clip  : {lstm_grad_clip_display}")
        print(f"LSTM Scheduler  : {lstm_scheduler_display}")

    if "mamba" in models_to_run:
        mamba_version_display = args.mamba_version if args.mamba_version is not None else default_cfg.mamba_version
        mamba_variant_display = args.mamba_variant if args.mamba_variant else f"{default_cfg.mamba_variant} (default)"
        mamba_layers_display = args.mamba_layers if args.mamba_layers is not None else default_cfg.mamba_layers
        print(f"Mamba Layers    : {mamba_layers_display}")
        print(f"Mamba Version   : {mamba_version_display}")
        print(f"Mamba Variant   : {mamba_variant_display}")
        mamba_lr_display = args.mamba_lr if args.mamba_lr is not None else default_cfg.mamba_lr
        mamba_min_lr_display = args.mamba_min_lr if args.mamba_min_lr is not None else default_cfg.mamba_min_lr
        mamba_weight_decay_display = args.mamba_weight_decay if args.mamba_weight_decay is not None else default_cfg.mamba_weight_decay
        mamba_scheduler_display = args.mamba_scheduler if args.mamba_scheduler is not None else default_cfg.mamba_scheduler
        mamba_grad_clip_display = args.mamba_grad_clip if args.mamba_grad_clip is not None else default_cfg.mamba_grad_clip
        mamba_warmup_steps_display = args.mamba_warmup_steps if args.mamba_warmup_steps is not None else default_cfg.mamba_warmup_steps
        mamba_adapter_bottleneck_display = args.mamba_adapter_bottleneck if args.mamba_adapter_bottleneck is not None else default_cfg.mamba_adapter_bottleneck
        print(f"Mamba LR        : {mamba_lr_display} (min: {mamba_min_lr_display})")
        print(f"Mamba WD        : {mamba_weight_decay_display}")
        print(f"Mamba Scheduler : {mamba_scheduler_display}")
        print(f"Mamba Grad Clip : {mamba_grad_clip_display}")
        print(f"Mamba Warmup    : {mamba_warmup_steps_display} steps")
        if mamba_variant_display and "adapter" in mamba_variant_display.lower():
            print(f"Mamba Adapter   : {mamba_adapter_bottleneck_display} bottleneck")

    print(f"Skip Finished    : {not args.force}")
    print(f"Eval Only       : {args.eval_only}")
    print(f"Subprocess Mode  : {not args.in_process}")
    if use_model_specific_logs:
        log_names = [f"running_log_{m}.txt" for m in models_to_run]
        print(f"Log Files        : {', '.join(log_names)}")
    else:
        print(f"Log Files        : running_log.txt")
    print("=" * 80)

    summary_records = []
    total_start_time = time.time()

    for tier in tiers_to_run:
        tier_name = TIER_NAMES.get(tier, f"Tier_{tier}")
        print(f"\n>>> ENTERING TIER {tier}: {tier_name.upper()} <<<\n")

        for model_type in models_to_run:
          for rho_val in rhos_to_run:
              # Use config defaults for any parameters not specified in command
              cfg_kwargs = {
                  'model_type': model_type,
                  'param_tier': tier,
                  'rho_scaling_experiment': args.rhos is not None,
              }
              if args.epoch is not None:
                  cfg_kwargs['epoch'] = args.epoch
              if args.batch_size is not None:
                  cfg_kwargs['batch_size'] = args.batch_size
              if rho_val is not None:
                  cfg_kwargs['rho'] = rho_val
              if args.lstm_layers is not None:
                  cfg_kwargs['lstm_layers'] = args.lstm_layers
              if args.mamba_layers is not None and model_type == "mamba":
                  cfg_kwargs['mamba_layers'] = args.mamba_layers
              if args.mamba_variant is not None and model_type == "mamba":
                  cfg_kwargs['mamba_variant'] = args.mamba_variant
              if args.mamba_version is not None and model_type == "mamba":
                  cfg_kwargs['mamba_version'] = args.mamba_version
              if args.mamba_lr is not None and model_type == "mamba":
                  cfg_kwargs['mamba_lr'] = args.mamba_lr
              if args.mamba_min_lr is not None and model_type == "mamba":
                  cfg_kwargs['mamba_min_lr'] = args.mamba_min_lr
              if args.mamba_weight_decay is not None and model_type == "mamba":
                  cfg_kwargs['mamba_weight_decay'] = args.mamba_weight_decay
              if args.mamba_scheduler is not None and model_type == "mamba":
                  cfg_kwargs['mamba_scheduler'] = args.mamba_scheduler
              if args.mamba_n_prefix_tokens is not None and model_type == "mamba":
                  cfg_kwargs['mamba_n_prefix_tokens'] = args.mamba_n_prefix_tokens
              if args.mamba_residual_scale_init is not None and model_type == "mamba":
                  cfg_kwargs['mamba_residual_scale_init'] = args.mamba_residual_scale_init
              if args.mamba_adapter_bottleneck is not None and model_type == "mamba":
                  cfg_kwargs['mamba_adapter_bottleneck'] = args.mamba_adapter_bottleneck
              if args.mamba_grad_clip is not None and model_type == "mamba":
                  cfg_kwargs['mamba_grad_clip'] = args.mamba_grad_clip
              if args.mamba_warmup_steps is not None and model_type == "mamba":
                  cfg_kwargs['mamba_warmup_steps'] = args.mamba_warmup_steps
              if args.lstm_lr is not None and model_type == "lstm":
                  cfg_kwargs['lr'] = args.lstm_lr
              if args.lstm_grad_clip is not None and model_type == "lstm":
                  cfg_kwargs['lstm_grad_clip'] = args.lstm_grad_clip
              if args.lstm_weight_decay is not None and model_type == "lstm":
                  cfg_kwargs['lstm_weight_decay'] = args.lstm_weight_decay
              if args.lstm_scheduler is not None and model_type == "lstm":
                  cfg_kwargs['lstm_scheduler'] = args.lstm_scheduler
              if args.lstm_min_lr is not None and model_type == "lstm":
                  cfg_kwargs['lstm_min_lr'] = args.lstm_min_lr
              if args.output_dir is not None:
                  cfg_kwargs['output_dir'] = args.output_dir

              cfg = ExperimentConfig(**cfg_kwargs)

              # Check if already completed
              if not args.force and is_run_completed(cfg):
                  print(f"[SKIP] Tier {tier} ({tier_name}) - {model_type.upper()} already exists.")
                  res = get_run_results(cfg)
                  res["status"] = "PREVIOUSLY_DONE"
                  res["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                  res["epoch"] = cfg.epoch
                  res["train_time_min"] = "0.0"
                  res["output_path"] = cfg.output_path()
                  summary_records.append(res)
                  log_prefix = model_type if use_model_specific_logs else "all"
                  append_to_running_log(res, SCRIPT_DIR, log_prefix)
                  continue

              run_start = time.time()
              if args.in_process:
                  # Direct in-process execution
                  cfg_kwargs = {
                      'model_type': model_type,
                      'tier': tier,
                      'eval_only': args.eval_only,
                      'force': args.force,
                      'rho_scaling_experiment': args.rhos is not None,
                  }
                  if args.epoch is not None:
                      cfg_kwargs['epoch'] = args.epoch
                  if args.batch_size is not None:
                      cfg_kwargs['batch_size'] = args.batch_size
                  if rho_val is not None:
                      cfg_kwargs['rho'] = rho_val
                  if args.lstm_layers is not None:
                      cfg_kwargs['lstm_layers'] = args.lstm_layers
                  if args.mamba_layers is not None and model_type == "mamba":
                      cfg_kwargs['mamba_layers'] = args.mamba_layers
                  if args.mamba_variant is not None and model_type == "mamba":
                      cfg_kwargs['mamba_variant'] = args.mamba_variant
                  if args.mamba_version is not None and model_type == "mamba":
                      cfg_kwargs['mamba_version'] = args.mamba_version
                  if args.mamba_lr is not None and model_type == "mamba":
                      cfg_kwargs['mamba_lr'] = args.mamba_lr
                  if args.mamba_min_lr is not None and model_type == "mamba":
                      cfg_kwargs['mamba_min_lr'] = args.mamba_min_lr
                  if args.mamba_weight_decay is not None and model_type == "mamba":
                      cfg_kwargs['mamba_weight_decay'] = args.mamba_weight_decay
                  if args.mamba_scheduler is not None and model_type == "mamba":
                      cfg_kwargs['mamba_scheduler'] = args.mamba_scheduler
                  if args.mamba_n_prefix_tokens is not None and model_type == "mamba":
                      cfg_kwargs['mamba_n_prefix_tokens'] = args.mamba_n_prefix_tokens
                  if args.mamba_residual_scale_init is not None and model_type == "mamba":
                      cfg_kwargs['mamba_residual_scale_init'] = args.mamba_residual_scale_init
                  if args.mamba_adapter_bottleneck is not None and model_type == "mamba":
                      cfg_kwargs['mamba_adapter_bottleneck'] = args.mamba_adapter_bottleneck
                  if args.mamba_grad_clip is not None and model_type == "mamba":
                      cfg_kwargs['mamba_grad_clip'] = args.mamba_grad_clip
                  if args.mamba_warmup_steps is not None and model_type == "mamba":
                      cfg_kwargs['mamba_warmup_steps'] = args.mamba_warmup_steps
                  if args.lstm_lr is not None and model_type == "lstm":
                      cfg_kwargs['lr'] = args.lstm_lr
                  if args.lstm_grad_clip is not None and model_type == "lstm":
                      cfg_kwargs['lstm_grad_clip'] = args.lstm_grad_clip
                  if args.lstm_weight_decay is not None and model_type == "lstm":
                      cfg_kwargs['lstm_weight_decay'] = args.lstm_weight_decay
                  if args.lstm_scheduler is not None and model_type == "lstm":
                      cfg_kwargs['lstm_scheduler'] = args.lstm_scheduler
                  if args.lstm_min_lr is not None and model_type == "lstm":
                      cfg_kwargs['lstm_min_lr'] = args.lstm_min_lr
                  if args.output_dir is not None:
                      cfg_kwargs['output_dir'] = args.output_dir

                  execute_single_run(**cfg_kwargs)
                  if torch.cuda.is_available():
                      torch.cuda.empty_cache()
              else:
                  # Isolated subprocess execution
                  cmd = [
                      sys.executable,
                      str(SCRIPT_DIR / "run_all_tiers.py"),
                      "--single",
                      "--single-model", model_type,
                      "--single-tier", str(tier),
                  ]
                  # Only add parameters if explicitly specified
                  if args.epoch is not None:
                      cmd.extend(["--single-epoch", str(args.epoch)])
                  if args.batch_size is not None:
                      cmd.extend(["--single-batch-size", str(args.batch_size)])
                  if rho_val is not None:
                      cmd.extend(["--single-rho", str(rho_val)])
                  if args.rhos is not None:
                      cmd.append("--single-rho-scaling")
                  if args.lstm_layers is not None:
                      cmd.extend(["--single-lstm-layers", str(args.lstm_layers)])
                  if args.mamba_layers is not None and model_type == "mamba":
                      cmd.extend(["--single-mamba-layers", str(args.mamba_layers)])
                  if args.mamba_variant is not None and model_type == "mamba":
                      cmd.extend(["--single-mamba-variant", args.mamba_variant])
                  if args.mamba_version is not None and model_type == "mamba":
                      cmd.extend(["--single-mamba-version", str(args.mamba_version)])
                  if args.mamba_lr is not None and model_type == "mamba":
                      cmd.extend(["--single-mamba-lr", str(args.mamba_lr)])
                  if args.mamba_min_lr is not None and model_type == "mamba":
                      cmd.extend(["--single-mamba-min-lr", str(args.mamba_min_lr)])
                  if args.mamba_weight_decay is not None and model_type == "mamba":
                      cmd.extend(["--single-mamba-weight-decay", str(args.mamba_weight_decay)])
                  if args.mamba_scheduler is not None and model_type == "mamba":
                      cmd.extend(["--single-mamba-scheduler", args.mamba_scheduler])
                  if args.mamba_n_prefix_tokens is not None and model_type == "mamba":
                      cmd.extend(["--single-mamba-n-prefix-tokens", str(args.mamba_n_prefix_tokens)])
                  if args.mamba_residual_scale_init is not None and model_type == "mamba":
                      cmd.extend(["--single-mamba-residual-scale-init", str(args.mamba_residual_scale_init)])
                  if args.mamba_adapter_bottleneck is not None and model_type == "mamba":
                      cmd.extend(["--single-mamba-adapter-bottleneck", str(args.mamba_adapter_bottleneck)])
                  if args.mamba_grad_clip is not None and model_type == "mamba":
                      cmd.extend(["--single-mamba-grad-clip", str(args.mamba_grad_clip)])
                  if args.mamba_warmup_steps is not None and model_type == "mamba":
                      cmd.extend(["--single-mamba-warmup-steps", str(args.mamba_warmup_steps)])
                  if args.lstm_lr is not None and model_type == "lstm":
                      cmd.extend(["--single-lstm-lr", str(args.lstm_lr)])
                  if args.lstm_grad_clip is not None and model_type == "lstm":
                      cmd.extend(["--single-lstm-grad-clip", str(args.lstm_grad_clip)])
                  if args.lstm_weight_decay is not None and model_type == "lstm":
                      cmd.extend(["--single-lstm-weight-decay", str(args.lstm_weight_decay)])
                  if args.lstm_scheduler is not None and model_type == "lstm":
                      cmd.extend(["--single-lstm-scheduler", args.lstm_scheduler])
                  if args.lstm_min_lr is not None and model_type == "lstm":
                      cmd.extend(["--single-lstm-min-lr", str(args.lstm_min_lr)])
                  if args.output_dir is not None:
                      cmd.extend(["--single-output-dir", args.output_dir])
                  # Add eval-only flag for subprocess if needed
                  if args.eval_only:
                      cmd.append("--single-eval-only")
                  # Add force flag for subprocess
                  if args.force:
                      cmd.append("--single-force")
                  variant_display = f" (variant: {args.mamba_variant})" if args.mamba_variant and model_type == "mamba" else ""
                  print(f"--- Running {model_type.upper()}{variant_display} | Tier {tier} ({tier_name}) ---")
                  result = subprocess.run(cmd, check=False)
                  if result.returncode != 0:
                      print(f"\n[ERROR] Tier {tier} {model_type.upper()} failed with code {result.returncode}!")
                      res = {
                          "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                          "tier": tier,
                          "tier_name": tier_name,
                          "model": model_type.upper(),
                          "hidden_size": cfg.hidden_size,
                          "params": "ERROR",
                          "train_time_min": f"{(time.time() - run_start) / 60:.1f}",
                          "median_nse": "ERROR",
                          "median_kge": "ERROR",
                          "variant": cfg.mamba_variant if model_type == "mamba" else "",
                          "mamba_version": cfg.mamba_version if model_type == "mamba" else "",
                          "status": f"FAILED (code {result.returncode})",
                          "output_path": cfg.output_path(),
                          "epoch": cfg.epoch,
                      }
                      summary_records.append(res)
                      log_prefix = model_type if use_model_specific_logs else "all"
                      append_to_running_log(res, SCRIPT_DIR, log_prefix)
                      continue

              elapsed = (time.time() - run_start) / 60
              res = get_run_results(cfg)
              res["status"] = f"COMPLETED ({elapsed:.1f}m)"
              res["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
              res["train_time_min"] = f"{elapsed:.1f}"
              res["output_path"] = cfg.output_path()
              summary_records.append(res)
              log_prefix = model_type if use_model_specific_logs else "all"
              append_to_running_log(res, SCRIPT_DIR, log_prefix)
              print(f"✓ [{model_type.upper()} Tier {tier}] Done in {elapsed:.1f}m | Median NSE: {res.get('median_nse')} | KGE: {res.get('median_kge')}\n")

    total_elapsed = (time.time() - total_start_time) / 3600
    print(f"All requested tiers finished in {total_elapsed:.2f} hours.")
    print_summary_table(summary_records)


if __name__ == "__main__":
    main()
