"""HTTP API for the farm map: boundaries, imagery, per-farm / per-area palm counts."""
import logging
from typing import Callable, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from . import config, regions
from .detector import PalmDetector, list_models
from .farms import FarmService

log = logging.getLogger("palmai.farms_api")


class AnalyzeRequest(BaseModel):
    codes: list[str] = Field(min_length=1, max_length=8000)
    model: str = ""
    confidence: float = config.DEFAULT_CONFIDENCE
    source: Literal["live", "saved"] = "live"      # live = capture new from Google Earth (and save it); saved = use a saved capture
    capture_id: str = ""                          # which saved capture to use (blank = the newest)
    refresh: bool = False
    imagery_date: Literal["", "newest", "keep"] = ""   # "" = newest well-lit date (default); "newest" = Earth's very latest, even if dark; "keep" = leave Earth's date as it is


class RegionRequest(BaseModel):
    region: str
    mode: Literal["farms", "region"] = "farms"      # farms = only grid cells that touch farms; region = the whole region
    range_m: int = 800                              # Earth camera height: 600 / 800 / 1000 m
    model: str = ""
    confidence: float = config.DEFAULT_CONFIDENCE
    auto_date: bool = True                          # pick a well-lit imagery date in Earth first
    resume: bool = True                             # continue an interrupted run of the same region/mode/camera


class AreaRequest(BaseModel):
    name: str = ""
    geometry: dict


def build_router(get_detector: Callable[[], PalmDetector | None], service: FarmService) -> APIRouter:
    router = APIRouter(prefix="/api/farms", tags=["farms"])
    jobs = service.jobs

    def detector() -> PalmDetector:
        d = get_detector()
        if d is None:
            raise HTTPException(503, "The AI model is still loading. Please try again in a moment.")
        return d

    def checked(model: str, confidence: float) -> str:
        if not 0.01 <= confidence <= 0.95:
            raise HTTPException(400, "Confidence must be between 0.01 and 0.95.")
        if model and model not in list_models():
            raise HTTPException(400, "Unknown model.")
        return model or detector().default_model

    def guarded(fn):
        try:
            return fn()
        except KeyError as exc:
            raise HTTPException(404, f"Unknown farm, area or capture: {exc.args[0]}")
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @router.get("/meta")
    def meta():
        return guarded(service.meta)

    @router.get("/geojson")
    def farms_geojson(region: str | None = None):
        return JSONResponse(guarded(lambda: service.farms_geojson(region)))

    @router.get("/regions")
    def regions_geojson():
        return JSONResponse(guarded(service.regions_geojson))

    @router.get("/captures")
    def captures():
        return [{"name": c.name, "kind": c.kind, "region": c.region, "bounds": c.bounds, "imagery_date": c.imagery_date,
                 "captured_at": c.captured_at, "available": c.available} for c in service.captures()]

    @router.get("/preview/{name}.jpg")
    def preview(name: str):
        path = guarded(lambda: service.preview(name))
        return FileResponse(path, headers={"Cache-Control": "max-age=300"})

    @router.get("/export.csv")
    def export(region: str | None = None, kind: Literal["farm", "area"] = "farm"):
        return Response(service.export_csv(region, kind), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="palm_counts_{kind}{"_" + region if region else ""}.csv"'})

    # --- custom areas -------------------------------------------------------------------
    @router.get("/areas")
    def areas():
        return JSONResponse(service.areas_geojson())

    @router.post("/areas")
    def add_area(body: AreaRequest):
        return guarded(lambda: service.add_area(body.name, body.geometry))

    @router.delete("/areas/{key}")
    def delete_area(key: str):
        guarded(lambda: service.delete_area(key.upper()))
        return {"deleted": key}

    # --- imagery + counting -------------------------------------------------------------
    @router.post("/plan")
    def plan(body: AnalyzeRequest):
        return guarded(lambda: service.capture_plan(body.codes))

    @router.post("/estimate")
    def estimate(body: AnalyzeRequest):
        return guarded(lambda: service.estimate(body.codes, body.source, body.refresh))

    @router.post("/analyze")
    def analyze(body: AnalyzeRequest):
        """Stored imagery + one polygon: answered directly. Anything else runs as a background job."""
        model = checked(body.model, body.confidence)
        det = detector()
        if len(body.codes) == 1 and body.source == "saved":      # no Earth needed: answer straight away
            try:
                return {"results": [guarded(lambda: service.analyze(body.codes[0], det, model, body.confidence, "saved",
                                                                    capture_id=body.capture_id))]}
            except HTTPException:
                raise
            except Exception:
                log.exception("Analysis of %s failed", body.codes[0])
                raise HTTPException(500, "We couldn't analyze this farm. Please try again.")
        for code in body.codes:           # fail fast on unknown codes
            guarded(lambda c=code: service.target(c))
        try:
            return {"job": jobs.start(body.codes, det, model, body.confidence, body.source, body.refresh, body.imagery_date, body.capture_id)}
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))

    # --- whole-region grid captures -----------------------------------------------------
    @router.post("/regions/estimate")
    def region_estimate(body: RegionRequest):
        info = guarded(lambda: regions.estimate(service, body.region, body.mode, body.range_m, body.auto_date))
        info["unfinished"] = regions.unfinished_run(body.region, body.mode, body.range_m)
        return info

    @router.post("/regions/run")
    def region_run(body: RegionRequest):
        model = checked(body.model, body.confidence)
        det = detector()
        guarded(lambda: regions.estimate(service, body.region, body.mode, body.range_m, False))   # validates the request
        try:
            job_id = jobs.start_task(lambda job, cancelled: regions.run_region(
                service, det, body.region, body.mode, body.range_m, model, body.confidence, body.auto_date, body.resume, job, cancelled))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return {"job": job_id}

    @router.get("/regions/result")
    def region_result(region: str):
        return regions.latest_result(region) or {}

    @router.get("/jobs/{job_id}")
    def job(job_id: str):
        j = jobs.jobs.get(job_id)
        if j is None:
            raise HTTPException(404, "Unknown job.")
        return j

    @router.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        if not jobs.cancel(job_id):
            raise HTTPException(404, "Unknown job.")
        return {"cancelling": job_id}

    @router.get("/{code}/captures")
    def saved_captures(code: str):
        guarded(lambda: service.target(code))
        return service.saved_captures(code)

    @router.delete("/{code}/captures/{capture_id}")
    def delete_saved_capture(code: str, capture_id: str):
        guarded(lambda: service.delete_saved_capture(code, capture_id))
        return {"deleted": capture_id}

    @router.get("/{code}")
    def farm(code: str):
        def get():
            f, info = service.target(code)
            saved = service.summary().get(f.code)
            base = {"code": f.code, "kind": info["kind"], "name": info.get("name"), "region": f.region, "type": f.type,
                    "status": f.status, "area_ha": round(f.area_ha, 3), "saved_captures": len(service.saved_captures(f.code))}
            return {**(saved or {}), **{k: v for k, v in base.items() if not (saved and k in saved)},
                    "analyzed": bool(saved), "points": service.saved_points(f.code) if saved else []}
        return guarded(get)

    return router
