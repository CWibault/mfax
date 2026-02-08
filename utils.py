import jax
import wandb

import os
import pickle


def wandb_log_info(info):
    """Log metrics to wandb."""
    jax.debug.print("info: {}", info)
    jax.experimental.io_callback(wandb.log, None, info)


def save_pkl(obj, directory, filename):
    """Save object to pickle file."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename), "wb") as f:
        pickle.dump(obj, f)
    return None


def load_pkl(path):
    """Load object from pickle file."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj
