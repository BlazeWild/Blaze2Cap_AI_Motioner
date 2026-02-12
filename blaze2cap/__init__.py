# -*- coding: utf-8 -*-
"""
Blaze2Cap: BlazePose to Motion Capture
Real-Time Motion Prediction from 2D/3D Pose Landmarks
"""

__version__ = "0.1.0"

# =============================================================================
# Core Module Imports - Use: from blaze2cap import MotionTransformer, etc.
# =============================================================================

# --- Models ---
from blaze2cap.modules.models import (
    MotionTransformer,
    TemporalTransfomerEncoder,
    TransformerBlock,
    CausalSelfAttention,
    FeedForward,
    PositionalEncoding,
    LayerNorm,
    QuickGELU
)

# --- Data ---
from blaze2cap.modules.data_loader import PoseSequenceDataset

# --- Loss ---
from blaze2cap.modeling.loss import MotionLoss
# --- Evaluation ---
from blaze2cap.modeling.eval_motion import MotionEvaluator, evaluate_motion

# --- Utilities ---
from blaze2cap.utils.checkpoint import save_checkpoint, load_checkpoint, auto_resume
from blaze2cap.utils.logging_ import setup_logging
from blaze2cap.utils.train_utils import (
    CudaPreFetcher,
    set_random_seed,
    Timer,
    get_timestamp,
    init_distributed_mode
)

# --- Convenience Exports ---
__all__ = [
    # Models
    "MotionTransformer",
    "TemporalTransfomerEncoder",
    "TransformerBlock",
    "CausalSelfAttention",
    "FeedForward",
    "PositionalEncoding",
    "LayerNorm",
    "QuickGELU",
    # Data
    "PoseSequenceDataset",
    # Loss
    "MotionCorrectionLoss",
    "LossBase",
    # Evaluation
    "MotionEvaluator",
    "evaluate_motion",
    # Utilities
    "save_checkpoint",
    "load_checkpoint",
    "auto_resume",
    "setup_logging",
    "CudaPreFetcher",
    "set_random_seed",
    "Timer",
    "get_timestamp",
    "init_distributed_mode",
]
