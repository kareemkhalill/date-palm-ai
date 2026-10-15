"""Upload validation/decoding and result persistence."""
import io
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from . import config

Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS


class InvalidImage(Exception):
    """Raised with a user-safe message when an upload cannot be used."""


def decode_upload(filename: str, data: bytes) -> np.ndarray:
    """Validate an uploaded file and return it as a BGR array (EXIF orientation applied)."""
    if Path(filename or "").suffix.lower() not in config.ALLOWED_EXTENSIONS:
        raise InvalidImage("Unsupported image format. Please upload a JPG, JPEG, or PNG image.")
    if not data:
        raise InvalidImage("The uploaded file is empty.")
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format not in config.ALLOWED_FORMATS:
                raise InvalidImage("Unsupported image format. Please upload a JPG, JPEG, or PNG image.")
            rgb = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise InvalidImage("This file could not be read as an image. Please try another image.")
    return np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1])


def save_result(original_bgr: np.ndarray, annotated_bgr: np.ndarray) -> tuple[str, str]:
    """Persist both images under unique names; returns (original_name, annotated_name)."""
    stem = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
    for kind, image, params in (
        ("original", original_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]),
        ("annotated", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92]),
    ):
        folder = config.RESULTS_DIR / kind
        folder.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(folder / f"{stem}.jpg"), image, params):
            raise OSError(f"Could not write {kind} image")
    prune_results()
    return f"{stem}.jpg", f"{stem}.jpg"


def prune_results() -> None:
    """Keep only the newest MAX_STORED_RESULTS original/annotated pairs."""
    for kind in ("original", "annotated"):
        folder = config.RESULTS_DIR / kind
        files = sorted(folder.glob("*.jpg"))  # names start with a timestamp
        for stale in files[:-config.MAX_STORED_RESULTS]:
            stale.unlink(missing_ok=True)
