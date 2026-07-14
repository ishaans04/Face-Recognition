"""ArcFace embedding extraction via InsightFace."""
import numpy as np
from insightface.app import FaceAnalysis

import config

_app = None


def load_model():
    """Lazily initialize and cache the InsightFace FaceAnalysis app."""
    global _app
    if _app is None:
        _app = FaceAnalysis(name=config.MODEL_NAME, root=config.MODELS_DIR)
        _app.prepare(ctx_id=config.CTX_ID, det_size=config.DET_SIZE, det_thresh=config.DET_THRESH)
    return _app


def get_embedding(image):
    """Detect, align, and embed the primary face in an image (512-d vector)."""
    app = load_model()
    faces = app.get(image)

    if not faces:
        raise ValueError("No face detected in image")

    if len(faces) > 1:
        # Largest bounding box wins when multiple faces are detected.
        faces = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )

    return faces[0].normed_embedding.astype(np.float32)


def get_embeddings(images):
    return [get_embedding(image) for image in images]
