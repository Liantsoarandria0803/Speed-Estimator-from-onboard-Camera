"""Timestamp alignment and GPS evaluation metrics."""
from __future__ import annotations
from io import BytesIO, StringIO
import numpy as np
import pandas as pd

def prepare_ground_truth(csv_data: str | bytes) -> pd.DataFrame:
    data = BytesIO(csv_data) if isinstance(csv_data, bytes) else StringIO(csv_data)
    raw = pd.read_csv(data)
    columns = {str(c).strip().lower(): c for c in raw.columns}
    time_col = next((columns[c] for c in ("timestamp", "time", "t", "seconds") if c in columns), None)
    speed_col = next((columns[c] for c in ("speed_kmh", "speed", "ground_truth_kmh", "gps_speed_kmh") if c in columns), None)
    if time_col is None or speed_col is None:
        raise ValueError("GPS CSV must contain a timestamp column and a speed_kmh column.")
    result = pd.DataFrame({"timestamp": pd.to_numeric(raw[time_col], errors="coerce"), "ground_truth_kmh": pd.to_numeric(raw[speed_col], errors="coerce")}).dropna()
    if result.empty:
        raise ValueError("GPS CSV contains no valid numeric measurements.")
    return result.sort_values("timestamp").drop_duplicates("timestamp")

def align_predictions(predictions: pd.DataFrame, ground_truth: pd.DataFrame, tolerance_seconds: float = 0.5) -> pd.DataFrame:
    aligned = pd.merge_asof(predictions.sort_values("timestamp"), ground_truth.sort_values("timestamp"), on="timestamp", direction="nearest", tolerance=tolerance_seconds).dropna(subset=["ground_truth_kmh"])
    if aligned.empty:
        raise ValueError("No prediction and GPS timestamps overlap within the selected tolerance.")
    return aligned

def calculate_metrics(aligned: pd.DataFrame) -> dict[str, float]:
    error = aligned["speed_kmh"].to_numpy() - aligned["ground_truth_kmh"].to_numpy()
    absolute = np.abs(error)
    actual = aligned["ground_truth_kmh"].to_numpy()
    nonzero = np.abs(actual) > 1e-8
    metrics = {"MAE (km/h)": float(np.mean(absolute)), "RMSE (km/h)": float(np.sqrt(np.mean(error**2)))}
    if np.any(nonzero):
        metrics["MAPE (%)"] = float(np.mean(absolute[nonzero] / np.abs(actual[nonzero])) * 100)
    return metrics

