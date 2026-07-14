# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Setup:
```
pip install -r requirements.txt
```

Verify one test image against a reference set (2-3 images of the same person):
```
python main.py verify --reference-dir data/reference/ --test-image data/test/photo.jpg [--threshold 0.6]
```

Batch-evaluate labeled pairs and print accuracy/precision/recall/F1/confusion matrix:
```
python main.py evaluate --pairs-csv pairs.csv [--threshold 0.6]
```
`pairs.csv` columns: `reference_dir,test_image,label` where `label` is `1` (same identity) or `0` (different).

There is no test suite, linter, or build step configured yet.

## Architecture

Linear pipeline, each stage a separate module under `src/`, orchestrated by `main.py`:

```
input.py -> preprocess.py -> embedding.py -> aggregator.py -> matcher.py -> evaluator.py
```

- **src/input.py** — loads/validates images from `data/reference/` and `data/test/` (extension whitelist, existence checks). Returns raw `cv2` BGR arrays.
- **src/preprocess.py** — only lightweight sanity checks (`ensure_min_resolution`). Actual face detection and alignment happen *inside* InsightFace's `FaceAnalysis.get()` call in `embedding.py`, not here.
- **src/embedding.py** — lazily loads and caches a single module-level InsightFace `FaceAnalysis` app (`load_model()`), then extracts a 512-d normalized ArcFace embedding per face via `get_embedding()`. If multiple faces are detected in one image, the largest bounding box is used. Raises `ValueError` if no face is found.
- **src/aggregator.py** — combines the 2-3 reference embeddings for one identity into a single vector by mean + re-normalization (`aggregate_embeddings`).
- **src/matcher.py** — cosine similarity between the aggregated reference embedding and the test embedding, thresholded against `config.THRESHOLD` to produce a boolean match.
- **src/evaluator.py** — turns lists of true/predicted labels (from batch `evaluate` runs) into accuracy/precision/recall/F1/confusion-matrix via scikit-learn.
- **main.py** — the only place that wires modules together. `build_reference_embedding()` loads + embeds + aggregates a reference directory; `run_verify()` and `run_evaluate()` are the two CLI entry points (`verify` and `evaluate` subcommands).
- **config.py** — single source of truth for `THRESHOLD`, `MODEL_NAME` (InsightFace model pack, default `buffalo_l`), `CTX_ID` (`-1` = CPU, `>=0` = GPU id), `DET_SIZE`, and the `data/`/`models/` directory paths. Import this instead of hardcoding paths or the threshold elsewhere.

### Key conventions

- Embeddings are always L2-normalized (both per-face in `embedding.py` and after aggregation in `aggregator.py`), so cosine similarity in `matcher.py` reduces to a dot product in practice, though it's computed generally.
- The InsightFace `FaceAnalysis` app is cached as a module-level singleton in `embedding.py` (`_app`) — it is expensive to initialize, so avoid re-instantiating it per call.
- No database, UI, or real-time camera support — this is a CLI batch/one-shot pipeline (`data/` on disk in, printed metrics/decision out).
