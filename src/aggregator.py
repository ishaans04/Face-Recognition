"""Combine multiple reference embeddings into one representative vector."""
import numpy as np


def aggregate_embeddings(embeddings):
    """Mean-aggregate embeddings, then re-normalize to unit length."""
    if not embeddings:
        raise ValueError("No embeddings to aggregate")

    mean_embedding = np.mean(np.stack(embeddings), axis=0)
    norm = np.linalg.norm(mean_embedding)
    if norm > 0:
        mean_embedding = mean_embedding / norm

    return mean_embedding
