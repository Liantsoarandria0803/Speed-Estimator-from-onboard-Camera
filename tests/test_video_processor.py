from pathlib import Path
import pytest
from src.video_processor import inspect_video

def test_rejects_unsupported_extension(tmp_path: Path):
    path = tmp_path / "clip.mkv"
    path.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="Unsupported video format"):
        inspect_video(path)

def test_rejects_empty_video(tmp_path: Path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        inspect_video(path)

