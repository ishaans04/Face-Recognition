"""Persist and load a multi-identity embedding gallery (for 1:N identification)."""
import os

import numpy as np


def save_gallery(labels, embeddings, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.savez(path, labels=np.array(labels), embeddings=np.stack(embeddings))


def load_gallery(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Gallery not found: {path} (run 'build-gallery' first)")
    data = np.load(path)
    return list(data["labels"]), data["embeddings"]
