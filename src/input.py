"""Load and validate images from disk."""
import os

import cv2

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_image(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imread(path)
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
