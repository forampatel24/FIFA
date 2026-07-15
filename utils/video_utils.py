import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class VideoMetadata:
    path: str
    fps: float
    width: int
    height: int
    total_frames: int
    duration_sec: float
    fourcc: str

    @property
    def resolution(self):
        return f"{self.width}x{self.height}"

    @property
    def duration_str(self):
        mins = int(self.duration_sec // 60)
        secs = int(self.duration_sec % 60)
        return f"{mins:02d}:{secs:02d}"


class VideoLoader:
    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}

    def __init__(self, path: str):
        self.path = Path(path)
        self.cap = None
        self.metadata = None

        if self.path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported format: {self.path.suffix}")

        self._open()

    def _open(self):
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise IOError(f"Could not open video: {self.path}")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0
        fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4))

        self.metadata = VideoMetadata(
            path=str(self.path),
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames,
            duration_sec=duration_sec,
            fourcc=fourcc,
        )

    def read_frame(self):
        if self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def seek(self, frame_idx: int):
        if self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __iter__(self):
        return self

    def __next__(self):
        frame = self.read_frame()
        if frame is None:
            raise StopIteration
        return frame

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


def get_file_size_str(path: str) -> str:
    size_bytes = Path(path).stat().st_size
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024 ** 2:.1f} MB"
    else:
        return f"{size_bytes / 1024 ** 3:.2f} GB"
