"""Frame deduplication using dHash + Hamming distance."""
from __future__ import annotations
import cv2
import numpy as np
from videoscan.core.frame_extractor import ExtractedFrame

class Deduplicator:
    def __init__(self, hamming_threshold: int = 3, hash_size: int = 8):
        self.hamming_threshold = hamming_threshold
        self.hash_size = hash_size

    def dhash(self, image: np.ndarray) -> int:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Gradient hash (dHash): encodes relative pixel differences
        resized = cv2.resize(gray, (self.hash_size + 1, self.hash_size))
        diff = resized[:, 1:] > resized[:, :-1]
        gradient_hash = sum(2**i for i, v in enumerate(diff.flatten()) if v)
        # Brightness hash: encode absolute mean brightness as 8 quantized bits
        # so that uniform images of different brightness produce different hashes.
        # Each bit i is set if mean > (i+1)*255/9, giving 8 thermometer-code bits.
        mean_val = float(np.mean(gray))
        brightness_bits = sum(
            2**i for i in range(8) if mean_val > (i + 1) * 255 / 9
        )
        # Combine: brightness occupies the upper 8 bits, dHash the lower bits
        bit_count = self.hash_size * self.hash_size
        return brightness_bits << bit_count | gradient_hash

    def hamming_distance(self, hash1: int, hash2: int) -> int:
        return bin(hash1 ^ hash2).count("1")

    def deduplicate(self, frames: list[ExtractedFrame]) -> list[ExtractedFrame]:
        if not frames: return []
        result = [frames[0]]
        prev_hash = self.dhash(frames[0].image)
        for frame in frames[1:]:
            curr_hash = self.dhash(frame.image)
            if self.hamming_distance(prev_hash, curr_hash) > self.hamming_threshold:
                result.append(frame)
            prev_hash = curr_hash
        return result
