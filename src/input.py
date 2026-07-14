"""Load and validate images from disk."""
import os

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_image(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")

    # cv2.imread() silently fails on Windows for paths/filenames containing
    # non-ASCII characters (returns None instead of raising). Reading the
    # bytes ourselves and decoding avoids that failure mode.
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image (unsupported or corrupt file): {path}")
    return image


def load_images(directory):
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Directory not found: {directory}")

    images = []
    paths = []
    for filename in sorted(os.listdir(directory)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        full_path = os.path.join(directory, filename)
        images.append(load_image(full_path))
        paths.append(full_path)

    if not images:
        raise ValueError(f"No valid images found in: {directory}")

    return images, paths
