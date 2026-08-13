"""
I/O utilities for saving and loading results.
"""
from __future__ import annotations
import json
import os
import shutil
from typing import Any, Dict, Optional
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_json(data: Any, path: str, indent: int = 2) -> bool:
    """Save data as JSON."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temp_path = path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=NumpyEncoder, indent=indent, ensure_ascii=False)
        os.replace(temp_path, path)
        return True
    except Exception as e:
        print(f"[IO] Error saving {path}: {e}")
        return False


def save_jsonl(records: list, path: str) -> bool:
    """Save records as JSON Lines."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temp_path = path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, cls=NumpyEncoder, ensure_ascii=False) + "\n")
        os.replace(temp_path, path)
        return True
    except Exception as e:
        print(f"[IO] Error saving {path}: {e}")
        return False


def save_npz_compressed(arrays: Dict[str, np.ndarray], path: str) -> bool:
    """Save numpy arrays as compressed NPZ."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temp_path = path + ".tmp"
        np.savez_compressed(temp_path, **arrays)
        os.replace(temp_path + ".npz", path)
        return True
    except Exception as e:
        print(f"[IO] Error saving {path}: {e}")
        return False


def setup_output_directory(output_root: str, run_name: str, overwrite_policy: str = "reject") -> str:
    """Create and set up output directory.

    Args:
        output_root: Root output directory
        run_name: Run name (empty for auto-generated)
        overwrite_policy: "reject" or "allow"

    Returns:
        Output directory path

    Raises:
        FileExistsError: If directory exists and overwrite_policy is "reject"
    """
    if not run_name:
        import time
        run_name = f"run_{int(time.time())}"

    output_dir = os.path.join(output_root, run_name)

    if os.path.exists(output_dir):
        if overwrite_policy == "reject":
            raise FileExistsError(f"Output directory {output_dir} already exists")
        elif overwrite_policy == "remove":
            shutil.rmtree(output_dir)

    # Create directory structure
    os.makedirs(os.path.join(output_dir, "config"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "traces"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    return output_dir


def save_run_manifest(manifest: dict, output_dir: str) -> bool:
    """Save run manifest."""
    path = os.path.join(output_dir, "run_manifest.json")
    return save_json(manifest, path)


def save_stage_outputs(
    stage: int,
    data: dict,
    output_dir: str,
) -> bool:
    """Save stage-specific outputs."""
    os.makedirs(output_dir, exist_ok=True)
    return True