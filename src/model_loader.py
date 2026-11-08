"""Download and load the official SpeedNet model."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
from huggingface_hub import snapshot_download

MODEL_REPO = "x2q/speednet"
CHECKPOINT_NAME = "speednet_v5.pt"

def load_speednet_model(cache_dir: str | Path | None = None) -> tuple[Any, str, Path]:
    """Return the official model, selected device, and downloaded repository path."""
    local_dir = Path(snapshot_download(repo_id=MODEL_REPO, cache_dir=str(cache_dir) if cache_dir else None))
    import sys
    if str(local_dir) not in sys.path:
        sys.path.insert(0, str(local_dir))
    from modeling_speednet import SpeedNet
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = local_dir / CHECKPOINT_NAME
    if not checkpoint.exists():
        raise FileNotFoundError(f"SpeedNet checkpoint not found: {checkpoint}")
    model = SpeedNet()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model, device, local_dir

