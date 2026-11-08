"""SpeedNet Streamlit application."""
from __future__ import annotations
import tempfile
from pathlib import Path
import streamlit as st
from components.dashboard import render_evaluation, render_results
from src.inference import estimate_speed
from src.model_loader import load_speednet_model
from src.video_processor import validate_video

st.set_page_config(page_title="SpeedNet", page_icon="🚗", layout="wide")
st.title("SpeedNet")
st.caption("Dashcam vehicle-speed estimation powered by a pretrained deep-learning model")

@st.cache_resource(show_spinner=False)
def cached_model():
    return load_speednet_model()

with st.sidebar:
    st.header("Settings")
    max_size = st.number_input("Maximum upload size (MB)", min_value=50, max_value=2000, value=500, step=50)
    max_duration = st.number_input("Maximum video duration (seconds)", min_value=10, max_value=3600, value=600, step=10)
    long_context = st.checkbox("Long-context mode", help="Use for closed-course or indoor-track footage.")
    st.divider()
    st.markdown("**Model:** SpeedNet v5\n\n**Runtime:** CPU-compatible\n\n**Output:** m/s and km/h")

st.markdown("Upload a forward-facing dashcam video to estimate speed over time.")
uploaded_video = st.file_uploader("Choose a video", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    suffix = Path(uploaded_video.name).suffix.lower()
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(uploaded_video.getbuffer())
            temp_path = Path(handle.name)
        metadata = validate_video(temp_path, int(max_size), int(max_duration))
        st.video(uploaded_video)
        st.success(f"Valid video: {metadata.width}×{metadata.height}, {metadata.fps:.2f} FPS, {metadata.duration_seconds:.1f} seconds")
        if st.button("Estimate speed", type="primary"):
            try:
                with st.spinner("Downloading/loading SpeedNet and processing video..."):
                    model, device, _ = cached_model()
                    predictions = estimate_speed(model, temp_path, device, long_context=long_context)
                st.session_state["predictions"] = predictions
                st.session_state["duration"] = metadata.duration_seconds
                st.session_state["device"] = device
                st.success(f"Inference complete on {device}.")
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
    except (ValueError, OSError) as exc:
        st.error(str(exc))
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)

predictions = st.session_state.get("predictions")
if predictions is not None:
    render_results(predictions, st.session_state["duration"])
    st.divider()
    gps_csv = st.file_uploader("Optional GPS ground-truth CSV", type=["csv"], key="gps")
    render_evaluation(predictions, gps_csv)

with st.expander("About the model"):
    st.write("SpeedNet estimates ego-vehicle speed from optical flow and RGB context. Accuracy depends on camera viewpoint, lighting, motion blur, scene type, and the model's learned scale assumptions. It must not be used as a certified speed instrument.")

