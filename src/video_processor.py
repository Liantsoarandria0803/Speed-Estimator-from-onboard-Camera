"""Video validation and metadata utilities."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import cv2

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi"}

@dataclass(frozen=True)
class VideoMetadata:
    duration_seconds: float
    fps: float
    width: int
    height: int
    frame_count: int
    codec: str

def inspect_video(path: str | Path) -> VideoMetadata:
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported video format. Please upload MP4, MOV, or AVI.")
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError("The uploaded video is empty or unavailable.")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("OpenCV could not open this video. It may be corrupted.")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0 or duration <= 0:
            raise ValueError("Video metadata is incomplete or invalid.")
        codec_value = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
        codec = "".join(chr((codec_value >> (8 * i)) & 0xFF) for i in range(4)).strip() or "unknown"
        return VideoMetadata(duration, fps, width, height, frame_count, codec)
    finally:
        capture.release()

def validate_video(path: str | Path, max_size_mb: int = 500, max_duration_seconds: int = 600) -> VideoMetadata:
    path = Path(path)
    if path.stat().st_size > max_size_mb * 1024 * 1024:
        raise ValueError(f"Video exceeds the {max_size_mb} MB size limit.")
    metadata = inspect_video(path)
    if metadata.duration_seconds > max_duration_seconds:
        raise ValueError(f"Video exceeds the {max_duration_seconds // 60}-minute duration limit.")
    return metadata

