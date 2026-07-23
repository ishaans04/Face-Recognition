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
- **matcher.py** — cosine similarity + threshold decision (1:1 `verify`) and gallery search (1:N `identify`)
- **gallery.py** — saves/loads a cached multi-identity embedding gallery (`.npz`)
- **evaluator.py** — accuracy, precision, recall, F1, confusion matrix

## Setup

```
pip install -r requirements.txt
```

## Usage

### 1:1 verification

Verify a single test image against a reference set (2-3 images of the same person):

```
python main.py verify --reference-dir data/reference/ --test-image data/test/photo.jpg
```

Batch-evaluate a set of labeled pairs (CSV with columns `reference_dir,test_image,label`,
where `label` is `1` for same identity and `0` for different):

```
python main.py evaluate --pairs-csv pairs.csv
```

### 1:N identification

Lay out a dataset of N different people, one subfolder per identity, each with 2-3+ images:

```
data/dataset/
  alice/photo1.jpg ...
  bob/photo1.jpg ...
```

Embed the whole dataset once into a cached gallery:

```
python main.py build-gallery --dataset-dir data/dataset/ --output models/gallery.npz
```

Then identify a test image against everyone in the gallery — prints the best-matching
identity, a confidence score, and a ranked top-k list (`Unknown` if the best score doesn't
clear the threshold):

```
python main.py identify --gallery models/gallery.npz --test-image data/test/photo.jpg --top-k 5
```

## Running on Google Colab

`colab/face_recognition_colab.ipynb` runs the same pipeline on Colab (with optional free GPU),
persisting `data/` and `models/` across sessions via a Google Drive mount. Upload this notebook
to [Colab](https://colab.research.google.com/) or open it via GitHub, then follow the cells in order.

## Configuration

See `config.py`:

- `THRESHOLD` — cosine similarity cutoff for a match (default `0.6`)
- `MODEL_NAME` — InsightFace model pack (default `buffalo_l`)
- `CTX_ID` — `-1` for CPU, `>=0` for a GPU device id
- `TOP_K` — default number of ranked matches returned by `identify` (default `5`)
- `DATASET_DIR` / `GALLERY_PATH` — default dataset and gallery-cache locations for `build-gallery`/`identify`
