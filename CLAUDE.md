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

Embed a dataset of N identities (one subfolder per person under `data/dataset/`) into a cached gallery:
```
python main.py build-gallery --dataset-dir data/dataset/ --output models/gallery.npz
```

Identify one test image against everyone in a prebuilt gallery (1:N search), printing the best match, a confidence score, and a ranked top-k list:
```
python main.py identify --gallery models/gallery.npz --test-image data/test/photo.jpg [--threshold 0.6] [--top-k 5]
```

There is no test suite, linter, or build step configured yet.

## Architecture

Linear pipeline, each stage a separate module under `src/`, orchestrated by `main.py`:

```
input.py -> preprocess.py -> embedding.py -> aggregator.py -> matcher.py -> evaluator.py
                                                            \-> gallery.py (1:N cache)
```

- **src/input.py** — loads/validates images from `data/reference/`, `data/test/`, and `data/dataset/<identity>/` (extension whitelist, existence checks). Returns raw `cv2` BGR arrays.
- **src/preprocess.py** — only lightweight sanity checks (`ensure_min_resolution`). Actual face detection and alignment happen *inside* InsightFace's `FaceAnalysis.get()` call in `embedding.py`, not here.
- **src/embedding.py** — lazily loads and caches a single module-level InsightFace `FaceAnalysis` app (`load_model()`), then extracts a 512-d normalized ArcFace embedding per face via `get_embedding()`. If multiple faces are detected in one image, the largest bounding box is used. Raises `ValueError` if no face is found.
- **src/aggregator.py** — combines the 2-3 reference embeddings for one identity into a single vector by mean + re-normalization (`aggregate_embeddings`). Used both for the 1:1 reference set and per-identity in the 1:N gallery.
- **src/matcher.py** — `verify()` does 1:1 cosine similarity between an aggregated reference embedding and a test embedding, thresholded against `config.THRESHOLD`. `identify()` does 1:N search: a single vectorized dot product of the test embedding against every gallery row (all rows are unit-norm, so this *is* cosine similarity), returning the best label (or `"Unknown"` if below threshold) plus a ranked top-k list.
- **src/gallery.py** — serializes/deserializes a `{identity: embedding}` gallery to/from a `.npz` file (`save_gallery`/`load_gallery`). Pure persistence — no face/embedding logic lives here, to keep orchestration in `main.py`.
- **src/evaluator.py** — turns lists of true/predicted labels (from batch `evaluate` runs) into accuracy/precision/recall/F1/confusion-matrix via scikit-learn.
- **main.py** — the only place that wires modules together. `build_reference_embedding()` loads + embeds + aggregates one identity's directory (reused for both a 1:1 reference set and, per-identity, inside `build_gallery()`); `run_verify()`/`run_evaluate()` are the 1:1 CLI entry points, `run_build_gallery()`/`run_identify()` are the 1:N CLI entry points (`verify`, `evaluate`, `build-gallery`, `identify` subcommands).
- **config.py** — single source of truth for `THRESHOLD`, `MODEL_NAME` (InsightFace model pack, default `buffalo_l`), `CTX_ID` (`-1` = CPU, `>=0` = GPU id), `DET_SIZE`, `TOP_K` (default `identify` result count), and the `data/`/`models/`/`DATASET_DIR`/`GALLERY_PATH` paths. Import this instead of hardcoding paths or the threshold elsewhere.

### Key conventions

- Embeddings are always L2-normalized (both per-face in `embedding.py` and after aggregation in `aggregator.py`), so cosine similarity in `matcher.py` reduces to a dot product in practice, though it's computed generally.
- The InsightFace `FaceAnalysis` app is cached as a module-level singleton in `embedding.py` (`_app`) — it is expensive to initialize, so avoid re-instantiating it per call.
- Gallery embeddings for `identify` are precomputed by `build-gallery` and cached to disk (`.npz`) rather than re-embedded on every query — `identify` only re-embeds the single test image.
- No database, UI, or real-time camera support — this is a CLI batch/one-shot pipeline (`data/` on disk in, printed metrics/decision out).
