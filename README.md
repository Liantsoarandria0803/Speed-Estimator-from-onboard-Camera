# SpeedNet

SpeedNet is a Streamlit research prototype for estimating vehicle speed from forward-facing dashcam video using the official pretrained [x2q/speednet](https://huggingface.co/x2q/speednet) model.

It is not a certified speed measurement instrument. Predictions may be affected by scene type, camera viewpoint, motion blur, lighting, and the model's learned scale assumptions.

## Features

- MP4, MOV, and AVI upload with metadata, size, duration, and corruption checks
- Official SpeedNet model downloaded from Hugging Face at runtime
- CPU-first inference with optional CUDA use when available
- Short-window and long-context inference modes
- Interactive speed-over-time dashboard
- Average, minimum, maximum, duration, and prediction-count summaries
- Downloadable prediction CSV
- Optional timestamp-aligned GPS evaluation with MAE, RMSE, and MAPE

## Local execution

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The first inference downloads the model repository. Internet access is required for that first run; no model weights are committed to this project.

## GPS CSV format

The evaluator accepts timestamp columns named `timestamp`, `time`, `t`, or `seconds`, and speed columns named `speed_kmh`, `speed`, `ground_truth_kmh`, or `gps_speed_kmh`.

```csv
timestamp,speed_kmh
0.0,32.4
0.5,33.1
1.0,34.0
```

Measurements are aligned to predictions using nearest timestamps within 0.5 seconds. MAPE excludes zero-valued ground-truth speeds because percentage error is undefined there.

## Streamlit Community Cloud

1. Push this repository to GitHub without videos or model weights.
2. Create a Streamlit Community Cloud app pointing to `app.py`.
3. Use Python dependencies from `requirements.txt`.
4. Ensure the deployment has enough memory for PyTorch and temporary video processing.

The application uses temporary files for uploads and downloads the model through `snapshot_download` at runtime. For CPU deployment, keep videos reasonably short and use the default short-window mode unless the footage is from a closed course.

## Tests

```bash
pytest -q
```

## Model integration

The application imports `SpeedNet` and `predict_video` from the downloaded official repository. The official pipeline computes Farneback optical flow, uses RGB context frames, performs GRU inference, and returns per-frame speed in meters per second. The UI converts those values to km/h for presentation.

