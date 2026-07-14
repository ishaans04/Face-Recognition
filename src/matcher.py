"""Similarity computation and match decisioning."""
import numpy as np

import config


def cosine_similarity(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def is_match(similarity, threshold=None):
    threshold = config.THRESHOLD if threshold is None else threshold
    return similarity > threshold


def verify(reference_embedding, test_embedding, threshold=None):
    similarity = cosine_similarity(reference_embedding, test_embedding)
    return similarity, is_match(similarity, threshold)
