import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
REFERENCE_DIR = os.path.join(DATA_DIR, "reference")
TEST_DIR = os.path.join(DATA_DIR, "test")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
GALLERY_PATH = os.path.join(MODELS_DIR, "gallery.npz")

THRESHOLD = 0.6
MODEL_NAME = "buffalo_l"
IMAGE_SIZE = (112, 112)
TOP_K = 5

DET_SIZE = (640, 640)
DET_THRESH = 0.4  # lower (e.g. 0.3) if real faces are still being missed
CTX_ID = -1  # -1 for CPU, >=0 for GPU device id
