"""YOLO26 date-palm detector: model registry, single-pass and tiled inference, annotation."""
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO

from . import config

log = logging.getLogger("palmai.detector")


@dataclass
class Detection:
    boxes: np.ndarray          # (N, 6): x1, y1, x2, y2, confidence, class
    mode: str                  # "auto", "single" or "tiled"
    tiles: int
    inference_ms: int
    window: int | None = None  # tiling window in source pixels (None for single pass)

    @property
    def count(self) -> int:
        return len(self.boxes)

    @property
    def average_confidence(self) -> float | None:
        return float(self.boxes[:, 4].mean()) if self.count else None


@dataclass
class LoadedModel:
    name: str
    yolo: YOLO
    palm_classes: list[int]    # only these classes are counted (multi-class models also detect roads, fields...)


def tile_origins(length: int, tile: int, overlap: float) -> list[int]:
    """Start offsets along one axis; the last tile is shifted back so every tile is full size."""
    if length <= tile:
        return [0]
    step = max(1, int(tile * (1 - overlap)))
    starts = list(range(0, length - tile, step))
    starts.append(length - tile)
    return starts


def list_models() -> list[str]:
    """Weight files available in the models folder, default model first."""
    names = sorted(p.name for p in config.MODELS_DIR.glob("*.pt"))
    if config.DEFAULT_MODEL in names:
        names.remove(config.DEFAULT_MODEL)
        names.insert(0, config.DEFAULT_MODEL)
    return names


class PalmDetector:
    def __init__(self) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device_name = torch.cuda.get_device_name(0) if self.device != "cpu" else "CPU"
        self._lock = threading.Lock()  # one inference/model load at a time; keeps GPU memory bounded
        self._cache: OrderedDict[str, LoadedModel] = OrderedDict()
        available = list_models()
        if not available:
            raise FileNotFoundError(f"No .pt model files found in {config.MODELS_DIR}")
        self.default_model = available[0]
        self._get_model(self.default_model)

    # --- model management -----------------------------------------------------
    def _get_model(self, name: str) -> LoadedModel:
        """Return a loaded model, loading it (and evicting the least recently used) if needed."""
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        if name not in list_models():  # also blocks path traversal: only files in MODELS_DIR
            raise ValueError(f"Unknown model: {name}")
        yolo = YOLO(str(config.MODELS_DIR / name))
        palm = [i for i, label in yolo.names.items() if "palm" in label.lower()]
        if not palm:
            raise ValueError(f"{name} has no palm class (classes: {list(yolo.names.values())})")
        loaded = LoadedModel(name, yolo, palm)
        blank = np.zeros((config.MODEL_IMGSZ, config.MODEL_IMGSZ, 3), np.uint8)
        self._predict([blank], config.DEFAULT_CONFIDENCE, loaded)  # warm-up
        self._cache[name] = loaded
        while len(self._cache) > config.MODEL_CACHE_SIZE:
            self._cache.popitem(last=False)
        log.info("Model %s loaded on %s (palm classes %s)", name, self.device_name, palm)
        return loaded

    def _predict(self, images: list[np.ndarray], conf: float, model: LoadedModel) -> list[np.ndarray]:
        try:
            results = self._run(images, conf, model)
        except RuntimeError as exc:
            # Another program (e.g. Google Earth) can take the GPU memory; carry on with the CPU.
            if self.device == "cpu" or not any(s in str(exc).lower() for s in ("out of memory", "cublas", "engine")):
                raise
            log.warning("GPU unavailable (%s); switching to CPU", str(exc).splitlines()[0])
            torch.cuda.empty_cache()
            self.device, self.device_name = "cpu", "CPU (GPU memory unavailable)"
            results = self._run(images, conf, model)
        boxes = [r.boxes.data.cpu() for r in results]
        if len(model.palm_classes) > 1:
            # Models trained on merged datasets can have several palm classes
            # (e.g. "Date-Palm" and "Palm tree"); keep one box per palm.
            boxes = [b[nms(b[:, :4], b[:, 4], 0.5)] if len(b) else b for b in boxes]
        return [b.numpy() for b in boxes]

    def _run(self, images: list[np.ndarray], conf: float, model: LoadedModel):
        return model.yolo.predict(
            images, conf=conf, imgsz=config.MODEL_IMGSZ, device=self.device,
            classes=model.palm_classes, max_det=config.MAX_DETECTIONS, verbose=False,
        )

    # --- inference --------------------------------------------------------------
    def detect(self, image_bgr: np.ndarray, conf: float, mode: str = "auto",
               model_name: str | None = None, tile: int | None = None,
               palm_px: float | None = None) -> Detection:
        """Run inference on a BGR image and return filtered detections in original-image pixels.

        `palm_px` is the expected palm width in source pixels when known (e.g. from a georeferenced
        capture); it is used when too few palms are found to measure the scale from the image.

        Modes:
          auto   - measure palm size, then tile with a window that shows palms to the model at
                   its preferred size (see config.TARGET_PALM_PX). Best for Google Earth exports,
                   whose zoom (and so palm size in pixels) varies from image to image.
          single - whole image in one pass, resized to the model size.
          tiled  - fixed native-resolution windows (`tile`, default config.TILE_SIZE).
        A window is fed to the model at 640 px, so a 1280 px window shows palms at half size.
        """
        with self._lock:
            model = self._get_model(model_name or self.default_model)
            started = time.perf_counter()
            window = None
            if mode == "single":
                boxes, tiles = self._predict([image_bgr], conf, model)[0], 1
            else:
                window = (tile or config.TILE_SIZE) if mode == "tiled" else self._auto_window(image_bgr, model, palm_px)
                boxes, tiles = self._detect_tiled(image_bgr, conf, model, window)
        elapsed = int((time.perf_counter() - started) * 1000)
        return Detection(boxes, mode, tiles, elapsed, window)

    def _auto_window(self, image: np.ndarray, model: LoadedModel, palm_hint_px: float | None = None) -> int:
        """Pick the tiling window from the measured palm size.

        Probes a few windows spread across the image at half scale, takes the median palm
        size and chooses the window that makes palms about TARGET_PALM_PX wide for the model.
        """
        window = config.PROBE_WINDOW
        for _ in range(3):  # re-measure at the chosen scale: palms are judged more reliably there
            palm_px = self._probe_palm_size(image, model, window)
            if palm_px is None:
                if palm_hint_px:
                    return int(np.clip(palm_hint_px * config.MODEL_IMGSZ / config.TARGET_PALM_PX,
                                       config.WINDOW_MIN, config.WINDOW_MAX))
                return config.FALLBACK_WINDOW if window == config.PROBE_WINDOW else window
            wanted = int(np.clip(palm_px * config.MODEL_IMGSZ / config.TARGET_PALM_PX,
                                 config.WINDOW_MIN, config.WINDOW_MAX))
            if abs(wanted - window) < 0.2 * window:
                return wanted
            window = wanted
        return window

    def _probe_palm_size(self, image: np.ndarray, model: LoadedModel, window: int) -> float | None:
        """Median palm width (source px) from five windows spread across the image."""
        height, width = image.shape[:2]
        xs = [int(f * max(0, width - window)) for f in (0.2, 0.5, 0.8)]
        ys = [int(f * max(0, height - window)) for f in (0.2, 0.5, 0.8)]
        spots = ((xs[0], ys[1]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[1], ys[2]), (xs[2], ys[1]))
        crops = [image[y:y + window, x:x + window] for x, y in spots]
        found = [b for b in self._predict(crops, config.DEFAULT_CONFIDENCE, model) if len(b)]
        if not found:
            return None
        boxes = np.concatenate(found)
        if len(boxes) < 5:
            return None
        return float(np.median(np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])))

    def _detect_tiled(self, image: np.ndarray, conf: float, model: LoadedModel,
                      size: int) -> tuple[np.ndarray, int]:
        height, width = image.shape[:2]
        origins = [(x, y) for y in tile_origins(height, size, config.TILE_OVERLAP)
                   for x in tile_origins(width, size, config.TILE_OVERLAP)]

        collected, tile_ids = [], []
        for start in range(0, len(origins), config.TILE_BATCH):
            batch = origins[start:start + config.TILE_BATCH]
            crops = [image[y:y + size, x:x + size] for x, y in batch]
            for offset, (preds, (x, y)) in enumerate(zip(self._predict(crops, conf, model), batch)):
                if len(preds):
                    preds = preds.copy()
                    preds[:, [0, 2]] += x
                    preds[:, [1, 3]] += y
                    collected.append(preds)
                    tile_ids.append(np.full(len(preds), start + offset))
        if not collected:
            return np.zeros((0, 6), np.float32), len(origins)
        return self._merge(np.concatenate(collected), np.concatenate(tile_ids)), len(origins)

    @staticmethod
    def _merge(boxes: np.ndarray, tile_ids: np.ndarray) -> np.ndarray:
        """Remove duplicates of palms seen in two overlapping tiles.

        Standard NMS handles near-identical boxes. A second pass removes a palm that was
        cut by one tile's border when the neighbouring tile saw it whole (the cut box is
        mostly contained in the full one), but only across different tiles so touching
        palms inside the same tile are never merged.
        """
        t = torch.from_numpy(boxes)
        order = nms(t[:, :4], t[:, 4], 0.5)  # indices, sorted by descending confidence
        t, tile_ids = t[order], torch.from_numpy(tile_ids)[order]

        xyxy = t[:, :4]
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        keep = torch.ones(len(t), dtype=torch.bool)
        for i in range(len(t)):
            if not keep[i]:
                continue
            lt = torch.maximum(xyxy[i, :2], xyxy[i + 1:, :2])
            rb = torch.minimum(xyxy[i, 2:], xyxy[i + 1:, 2:])
            inter = (rb - lt).clamp(min=0).prod(dim=1)
            share_of_smaller = inter / torch.minimum(areas[i], areas[i + 1:]).clamp(min=1e-6)
            duplicate = (share_of_smaller > 0.7) & (tile_ids[i + 1:] != tile_ids[i])
            keep[i + 1:] &= ~duplicate
        return t[keep].numpy()

    # --- annotation ---------------------------------------------------------------
    @staticmethod
    def annotate(image_bgr: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Mark each detected palm crown with a dot at the centre of its detection box."""
        canvas = image_bgr.copy()
        if not len(boxes):
            return canvas
        sides = np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
        longest = max(image_bgr.shape[:2])
        # Dot scales with the typical palm size, with a floor so it stays visible on huge images.
        radius = int(np.clip(np.median(sides) * 0.09, max(3, longest / 700), 24))
        ring = max(1, radius // 4)
        for x1, y1, x2, y2 in boxes[:, :4]:
            centre = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            cv2.circle(canvas, centre, radius + ring, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, centre, radius, (30, 30, 235), -1, cv2.LINE_AA)
        return canvas
