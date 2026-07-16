"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import tensorflow as tf


def set_seed_tf(seed: int) -> None:
    """Set Python, NumPy, and TensorFlow seeds on a best-effort basis."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism(True)
    except Exception:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
