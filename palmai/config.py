"""Central configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _env(name: str, default):
    value = os.environ.get(name)
    return type(default)(value) if value is not None else default


# --- Model -----------------------------------------------------------------
MODELS_DIR = Path(os.environ.get("PALM_MODELS_DIR", BASE_DIR))  # every *.pt here is selectable in the UI
DEFAULT_MODEL = os.environ.get("PALM_DEFAULT_MODEL", "yolov26l-mixed-data-70epoch.pt")
MODEL_CACHE_SIZE = _env("PALM_MODEL_CACHE", 2)   # models kept in memory at once
MODEL_IMGSZ = _env("PALM_IMGSZ", 640)            # resolution the model was trained at
# 0.15 recovers noticeably more real palms than Ultralytics' 0.25 default on Google Earth
# imagery of Al Ula farms (+14% on a dense farm) without visible false positives.
DEFAULT_CONFIDENCE = _env("PALM_CONF", 0.15)
MAX_DETECTIONS = _env("PALM_MAX_DET", 20000)

# --- Scale handling ---------------------------------------------------------
# Counts depend heavily on how big palms look to the model. Tests on Al Ula Google Earth
# exports showed the best counts when palms appear ~22-36 px wide in the 640 px model input,
# so "auto" mode measures palm size and picks the tiling window to hit TARGET_PALM_PX.
TARGET_PALM_PX = _env("PALM_TARGET_PX", 24)
PROBE_WINDOW = _env("PALM_PROBE_WINDOW", 1280)      # source px; probe windows are run at half scale
FALLBACK_WINDOW = _env("PALM_FALLBACK_WINDOW", 1600)  # used if the probe finds no palms
WINDOW_MIN = _env("PALM_WINDOW_MIN", 480)
WINDOW_MAX = _env("PALM_WINDOW_MAX", 12000)
TILE_SIZE = _env("PALM_TILE_SIZE", 640)             # window for the fixed "tiled" mode
TILE_OVERLAP = _env("PALM_TILE_OVERLAP", 0.25)
TILE_BATCH = _env("PALM_TILE_BATCH", 8)

# --- Uploads ----------------------------------------------------------------
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}
MAX_UPLOAD_MB = _env("PALM_MAX_UPLOAD_MB", 100)
MAX_IMAGE_PIXELS = _env("PALM_MAX_PIXELS", 250_000_000)

# --- Results persistence ------------------------------------------------------
RESULTS_DIR = Path(os.environ.get("PALM_RESULTS_DIR", BASE_DIR / "results"))
MAX_STORED_RESULTS = _env("PALM_MAX_STORED", 200)  # oldest pairs are pruned beyond this
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = Path(os.environ.get("PALM_DATA_DIR", BASE_DIR / "data"))  # farm boundaries and Earth captures

# --- Farm analysis ----------------------------------------------------------------
EXPECTED_PALM_M = _env("PALM_EXPECTED_M", 7.0)      # mature crown width, used when too few palms to measure scale
FARM_MARGIN = _env("PALM_FARM_MARGIN", 0.08)        # context around a farm in its crop (fraction of its size)
FARM_MARGIN_MIN_M = _env("PALM_FARM_MARGIN_MIN_M", 12)
FARM_DETECT_MAX_PX = _env("PALM_FARM_DETECT_PX", 6000)  # larger farms are downscaled before detection
FARM_IMAGE_MAX_PX = _env("PALM_FARM_IMAGE_PX", 2000)    # longest side of the saved farm image
PREVIEW_PX = _env("PALM_PREVIEW_PX", 3000)               # width of map preview images

# --- Region grid captures ----------------------------------------------------------------
REGION_MAX_CAPTURES = _env("PALM_REGION_MAX_CAPTURES", 900)   # one region run never starts more captures than this
REGION_KEEP_ALL_RUNS = _env("PALM_REGION_KEEP_ALL", 0)        # 1 keeps every finished run (cells are ~12 MB each)

# --- Live imagery from Google Earth Pro -----------------------------------------------
LIVE_RESOLUTION = os.environ.get("PALM_LIVE_RES", "4k").lower()     # Earth Save Image size: "4k" (3840x2160, default, faster) or "8k" (8192x4320)
AUTO_IMAGERY_DATE = _env("PALM_AUTO_DATE", 1)      # pick a well-lit Historical Imagery date before every capture job
LIVE_RANGE_M = _env("PALM_LIVE_RANGE_M", 600)      # Earth camera range; 600 m puts a mature palm at ~90 px
LIVE_OVERLAP = _env("PALM_LIVE_OVERLAP", 0.25)      # keeps the corner markers inside neighbouring captures
LIVE_MARGIN_DEG = _env("PALM_LIVE_MARGIN_DEG", 0.0001)  # ~10 m extra imagery around the (already padded) polygon
LIVE_MAX_AGE_HOURS = _env("PALM_LIVE_MAX_AGE_H", 24)    # reuse a live capture this long before re-capturing
LIVE_MAX_CAPTURES = _env("PALM_LIVE_MAX_CAPTURES", 40)  # one job never starts more captures than this
