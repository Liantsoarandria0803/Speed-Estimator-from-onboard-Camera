"""Dashboard rendering helpers."""
from __future__ import annotations
import pandas as pd
import streamlit as st
from src.visualization import comparison_figure, speed_figure

def render_results(predictions: pd.DataFrame, duration: float) -> None:
    st.subheader("Results dashboard")
    st.caption("These are model estimates, not certified speed measurements.")
    a, b, c, d = st.columns(4)
    a.metric("Average", f"{predictions.speed_kmh.mean():.1f} km/h")
    b.metric("Maximum", f"{predictions.speed_kmh.max():.1f} km/h")
    c.metric("Minimum", f"{predictions.speed_kmh.min():.1f} km/h")
    d.metric("Duration", f"{duration:.1f} s")
    st.plotly_chart(speed_figure(predictions), use_container_width=True)
    st.caption(f"{len(predictions):,} speed predictions")
    st.dataframe(predictions, use_container_width=True, hide_index=True)
    st.download_button("Download predictions CSV", predictions.to_csv(index=False).encode("utf-8"), "speednet_predictions.csv", "text/csv")

def render_evaluation(predictions: pd.DataFrame, uploaded_csv) -> None:
    st.subheader("Optional GPS evaluation")
    st.caption("Upload a CSV with timestamp/time/t/seconds and speed_kmh/speed columns.")
    if uploaded_csv is None:
        st.info("No ground-truth file provided. Evaluation metrics are not invented.")
        return
    from src.evaluation import align_predictions, calculate_metrics, prepare_ground_truth
    try:
        truth = prepare_ground_truth(uploaded_csv.getvalue())
        aligned = align_predictions(predictions, truth)
        metrics = calculate_metrics(aligned)
        cols = st.columns(len(metrics))
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name, f"{value:.2f}")
        st.plotly_chart(comparison_figure(aligned), use_container_width=True)
        st.caption(f"Aligned {len(aligned):,} measurements within ±0.5 seconds.")
    except (ValueError, KeyError) as exc:
        st.error(str(exc))

