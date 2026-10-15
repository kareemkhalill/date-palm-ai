"""Date Palm AI web application.

Run with:  python app.py        (or: uvicorn app:app --reload)
"""
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from palmai import config
from palmai.detector import PalmDetector, list_models
from palmai.earth_api import build_router as build_earth_router
from palmai.farms import FarmService
from palmai.farms_api import build_router
from palmai.images import InvalidImage, decode_upload, save_result

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("palmai")

detector: PalmDetector | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global detector
    detector = PalmDetector()  # loaded once at startup, reused for every request
    yield


app = FastAPI(title="Date Palm AI", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=2000)
farm_service = FarmService()
app.include_router(build_router(lambda: detector, farm_service))
app.include_router(build_earth_router(farm_service, lambda: detector))


@app.get("/health")
def health():
    return {
        "model_ready": detector is not None,
        "models": list_models(),
        "default_model": detector.default_model if detector else None,
        "device": detector.device_name if detector else None,
        "default_confidence": config.DEFAULT_CONFIDENCE,
        "max_upload_mb": config.MAX_UPLOAD_MB,
    }


@app.post("/predict")
def predict(
    image: UploadFile = File(...),
    confidence: float = Form(config.DEFAULT_CONFIDENCE),
    mode: str = Form("auto"),
    model: str = Form(""),
):
    if detector is None:
        raise HTTPException(503, "The AI model is still loading. Please try again in a moment.")
    if not 0.01 <= confidence <= 0.95:
        raise HTTPException(400, "Confidence must be between 0.01 and 0.95.")
    if mode not in {"auto", "single", "tiled"}:
        raise HTTPException(400, "Unknown processing mode.")
    if model and model not in list_models():
        raise HTTPException(400, "Unknown model.")

    data = image.file.read(config.MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"This image is larger than the {config.MAX_UPLOAD_MB} MB limit.")

    try:
        original = decode_upload(image.filename, data)
    except InvalidImage as exc:
        raise HTTPException(400, str(exc))
    del data

    try:
        detection = detector.detect(original, confidence, mode, model or None)
        annotated = detector.annotate(original, detection.boxes)
        original_name, annotated_name = save_result(original, annotated)
    except Exception:
        log.exception("Inference failed for %s", image.filename)
        raise HTTPException(500, "We couldn't analyze this image. Please try another image.")

    height, width = original.shape[:2]
    return {
        "filename": image.filename,
        "count": detection.count,
        "average_confidence": detection.average_confidence,
        "confidence_threshold": confidence,
        "image_width": width,
        "image_height": height,
        "original_url": f"/results/original/{original_name}",
        "annotated_url": f"/results/annotated/{annotated_name}",
        "model": model or detector.default_model,
        "device": detector.device_name,
        "mode": detection.mode,
        "tile_window": detection.window,
        "tiles": detection.tiles,
        "inference_ms": detection.inference_ms,
    }


config.RESULTS_DIR.mkdir(exist_ok=True)
app.mount("/results", StaticFiles(directory=config.RESULTS_DIR), name="results")
(config.DATA_DIR / "live").mkdir(parents=True, exist_ok=True)
app.mount("/live", StaticFiles(directory=config.DATA_DIR / "live"), name="live")   # Earth capture status strips
(config.DATA_DIR / "saved").mkdir(parents=True, exist_ok=True)
app.mount("/saved", StaticFiles(directory=config.DATA_DIR / "saved"), name="saved")   # saved captures' imagery-date strips
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(config.STATIC_DIR / "index.html")


@app.get("/farms")
def farm_map():
    return FileResponse(config.STATIC_DIR / "farms.html")


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000)
