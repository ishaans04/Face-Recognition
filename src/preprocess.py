"""Preprocessing helpers.

Face detection, alignment, and normalization are handled internally by
InsightFace's FaceAnalysis pipeline (see embedding.py). This module only
covers the lightweight sanity checks applied before an image reaches that
pipeline.
"""


def ensure_min_resolution(image, min_size=64):
    h, w = image.shape[:2]
    if h < min_size or w < min_size:
        raise ValueError(f"Image resolution too small ({w}x{h}); minimum is {min_size}x{min_size}")
    return image
