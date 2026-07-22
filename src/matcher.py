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


def identify(gallery_embeddings, gallery_labels, test_embedding, threshold=None, top_k=None):
    """Match a test embedding against every identity in a gallery.

    gallery_embeddings/test_embedding are unit-norm (aggregator + embedding.py both
    L2-normalize), so cosine similarity against the whole gallery reduces to one
    matrix-vector dot product instead of a per-identity loop.
    """
    threshold = config.THRESHOLD if threshold is None else threshold
    top_k = config.TOP_K if top_k is None else top_k

    similarities = np.asarray(gallery_embeddings) @ np.asarray(test_embedding)
    order = np.argsort(similarities)[::-1][:top_k]
    ranked = [(gallery_labels[i], float(similarities[i])) for i in order]

    best_label, best_score = ranked[0]
    identity = best_label if best_score > threshold else "Unknown"
    return identity, best_score, ranked
