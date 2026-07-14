# Face Verification System (ArcFace-based)

A modular face verification pipeline using ArcFace embeddings (via InsightFace), multi-image
enrollment, and similarity-based decision logic.

## Pipeline

```
INPUT -> PREPROCESS -> EMBEDDING -> AGGREGATION -> MATCHING -> EVALUATION -> OUTPUT
```

- **input.py** — loads and validates images from disk
- **preprocess.py** — lightweight sanity checks (detection/alignment is handled internally by InsightFace)
- **embedding.py** — loads the ArcFace model and extracts 512-d face embeddings
- **aggregator.py** — mean-aggregates multiple reference embeddings into one
- **matcher.py** — cosine similarity + threshold decision
- **evaluator.py** — accuracy, precision, recall, F1, confusion matrix

## Setup

```
pip install -r requirements.txt
```

## Usage

Verify a single test image against a reference set (2-3 images of the same person):

```
python main.py verify --reference-dir data/reference/ --test-image data/test/photo.jpg
```

Batch-evaluate a set of labeled pairs (CSV with columns `reference_dir,test_image,label`,
where `label` is `1` for same identity and `0` for different):

```
python main.py evaluate --pairs-csv pairs.csv
```

## Configuration

See `config.py`:

- `THRESHOLD` — cosine similarity cutoff for a match (default `0.6`)
- `MODEL_NAME` — InsightFace model pack (default `buffalo_l`)
- `CTX_ID` — `-1` for CPU, `>=0` for a GPU device id
