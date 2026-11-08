"""SpeedNet inference orchestration."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd

def estimate_speed(model: Any, video_path: str | Path, device: str, long_context: bool = False) -> pd.DataFrame:
    """Run the official SpeedNet predict_video function."""
    import sys
    repo_dir = next((Path(p) for p in sys.path if (Path(p) / "modeling_speednet.py").exists()), None)
    if repo_dir is None:
        raise RuntimeError("The downloaded SpeedNet implementation is not available.")
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from modeling_speednet import predict_video
    result = predict_video(model, str(video_path), device=device, long_context=long_context)
    speeds = result.get("speed_smooth_mps")
    timestamps = result.get("t")
    if speeds is None or timestamps is None or len(speeds) == 0:
        raise RuntimeError("SpeedNet returned no predictions.")
    frame = pd.DataFrame({"timestamp": timestamps, "speed_mps": speeds})
    frame["speed_kmh"] = frame["speed_mps"] * 3.6
    return frame

