"""Frame extraction from video using ffmpeg + OpenCV."""
from __future__ import annotations
import base64, io, logging, subprocess
from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

@dataclass
class ExtractedFrame:
    timestamp: float
    image: np.ndarray
    strategy: str

    def to_base64(self, max_width: int = 1280, quality: int = 85) -> str:
        img = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        if img.width > max_width:
            ratio = max_width / img.width
            new_size = (max_width, int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

class FrameExtractor:
    def extract_interval(self, video_path: str, interval: float = 5.0, max_frames: int = 30) -> list[ExtractedFrame]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps
        frames = []
        t = 0.0
        while t < duration and len(frames) < max_frames:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret: break
            frames.append(ExtractedFrame(timestamp=round(t, 2), image=frame, strategy="interval"))
            t += interval
        cap.release()
        return frames

    def extract_scene_changes(self, video_path: str, threshold: float = 0.3, max_frames: int = 30) -> list[ExtractedFrame]:
        try:
            return self._ffmpeg_scene_detect(video_path, threshold, max_frames)
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning(f"ffmpeg scene detection failed: {e}. Falling back to OpenCV.")
            return self._opencv_scene_detect(video_path, threshold, max_frames)

    def _ffmpeg_scene_detect(self, video_path: str, threshold: float, max_frames: int) -> list[ExtractedFrame]:
        # Use fps filter to sample at 2fps before scene detection — much faster than decoding every frame
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps=2,select='gt(scene,{threshold})',showinfo",
            "-vsync", "vfr", "-f", "null", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[:200]}")
        timestamps = []
        for line in result.stderr.split("\n"):
            if "pts_time:" in line:
                parts = line.split("pts_time:")
                if len(parts) > 1:
                    try: timestamps.append(float(parts[1].split()[0]))
                    except ValueError: continue
        timestamps = timestamps[:max_frames]
        cap = cv2.VideoCapture(video_path)
        frames = []
        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
            ret, frame = cap.read()
            if ret: frames.append(ExtractedFrame(timestamp=round(ts, 2), image=frame, strategy="scene_change"))
        cap.release()
        return frames

    def _opencv_scene_detect(self, video_path: str, threshold: float, max_frames: int) -> list[ExtractedFrame]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # Sample every 0.5 seconds instead of every frame — 60x faster for 30fps video
        step = max(1, int(fps * 0.5))
        frames, prev_gray, frame_idx = [], None, 0
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    score = float(np.mean(diff)) / 255.0
                    if score > threshold:
                        frames.append(ExtractedFrame(timestamp=round(frame_idx / fps, 2), image=frame, strategy="scene_change"))
                prev_gray = gray
            frame_idx += 1
        cap.release()
        return frames

    def extract_combined(self, video_path: str, threshold: float = 0.3, interval: float = 5.0, max_frames: int = 30) -> list[ExtractedFrame]:
        scene_frames = self.extract_scene_changes(video_path, threshold, max_frames)
        interval_frames = self.extract_interval(video_path, interval, max_frames)
        all_frames = scene_frames + interval_frames
        all_frames.sort(key=lambda f: f.timestamp)
        merged = []
        for f in all_frames:
            if not merged or abs(f.timestamp - merged[-1].timestamp) > 0.5:
                merged.append(f)
        return merged[:max_frames]

    def extract_at(self, video_path: str, timestamp: float) -> ExtractedFrame:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        cap.release()
        if not ret: raise RuntimeError(f"Cannot read frame at {timestamp}s")
        return ExtractedFrame(timestamp=round(timestamp, 2), image=frame, strategy="interval")

    def extract_dense(self, video_path: str, start: float, end: float, max_frames: int = 0) -> list[ExtractedFrame]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {video_path}")
        frames, t = [], start
        while t <= end:
            if max_frames > 0 and len(frames) >= max_frames: break
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if ret: frames.append(ExtractedFrame(timestamp=round(t, 2), image=frame, strategy="interval"))
            t += 1.0
        cap.release()
        return frames
