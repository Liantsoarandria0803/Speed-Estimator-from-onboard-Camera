"""Plotly visualizations."""
import pandas as pd
import plotly.graph_objects as go

def speed_figure(predictions: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=predictions["timestamp"], y=predictions["speed_kmh"], mode="lines", name="Estimated speed"))
    fig.update_layout(xaxis_title="Time (s)", yaxis_title="Speed (km/h)", hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10))
    return fig

def comparison_figure(aligned: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=aligned["timestamp"], y=aligned["speed_kmh"], mode="lines", name="Estimated"))
    fig.add_trace(go.Scatter(x=aligned["timestamp"], y=aligned["ground_truth_kmh"], mode="lines", name="GPS ground truth"))
    fig.update_layout(xaxis_title="Time (s)", yaxis_title="Speed (km/h)", hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10))
    return fig

