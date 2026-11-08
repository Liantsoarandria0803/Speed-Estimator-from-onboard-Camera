import pandas as pd
import pytest
from src.evaluation import align_predictions, calculate_metrics, prepare_ground_truth

def test_metrics():
    aligned = pd.DataFrame({"speed_kmh": [10.0, 20.0], "ground_truth_kmh": [12.0, 18.0]})
    metrics = calculate_metrics(aligned)
    assert metrics["MAE (km/h)"] == pytest.approx(2.0)
    assert metrics["RMSE (km/h)"] == pytest.approx(2.0)

def test_ground_truth_and_alignment():
    truth = prepare_ground_truth("timestamp,speed_kmh\n0,10\n1,20\n")
    predictions = pd.DataFrame({"timestamp": [0.1, 1.1], "speed_kmh": [11, 19]})
    result = align_predictions(predictions, truth, tolerance_seconds=0.2)
    assert len(result) == 2

def test_invalid_ground_truth_schema():
    with pytest.raises(ValueError, match="timestamp"):
        prepare_ground_truth("foo,bar\n1,2\n")

