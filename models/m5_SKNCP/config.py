"""
config.py - Hyperparameters and dataset paths
"""
from pathlib import Path

# ── Dataset path ──────────────────────────────────────────────────
DATASET_DIR = Path("../datasets")   # adjust to your dataset location

# ── Data split ────────────────────────────────────────────────────
TRAIN_RATIO = 0.80   # chronological split: first 80% → internal train

# ── Laplace smoothing for CTR estimation ──────────────────────────
LAPLACE_K = 20       # smoothing strength; rare ads/categories → global CTR

# ── SKNCP (Semantic K-Nearest-Neighbor Click Prediction) ──────────
SKNCP_K    = 100     # number of nearest neighbor queries to retrieve
SKNCP_BATCH = 300    # batch size for matrix multiplication

# ── Coordinate Ascent ─────────────────────────────────────────────
WEIGHT_MIN   = 0.5   # minimum weight per feature (prevents HistCTR → 0)
WEIGHT_MAX   = 3.0   # maximum weight per feature
WEIGHT_STEP  = 0.1   # grid search step size
MAX_ITER     = 10    # maximum coordinate ascent iterations

# ── Threshold (OOF) ───────────────────────────────────────────────
OOF_FOLDS    = 5     # number of folds for OOF threshold estimation
THRESHOLD_MIN = 0.001
THRESHOLD_MAX = 0.10
THRESHOLD_STEPS = 500

# ── Fixed offsets (from EDA) ──────────────────────────────────────
import math
LOGIN_OFFSET = math.log(1.7)   # not-logged-in users have ~1.7x higher CTR
HIST_CTR_FLOOR_RATIO = 0.15   # HistCTR floor = global_CTR * this ratio
