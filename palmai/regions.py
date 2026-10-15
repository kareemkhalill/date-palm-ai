"""Region runs: capture a whole region from Google Earth as a grid of views, then count every farm in it.

  farm areas only : only grid cells that touch a farm polygon are captured (far fewer than the whole region)
  whole region    : every cell that touches the region polygon is captured, so palms outside farms are found too

For both modes the run
  1. picks a well-lit imagery date in Google Earth (optional),
  2. captures the grid cells (resumable; each cell is georeferenced exactly),
  3. detects palms in the cells and reports those that lie in the region but outside every farm polygon,
  4. counts each farm by cropping it out of the cells (and saves that crop for the farm, like a single-farm capture).
"""
import csv
import json
import logging
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from shapely import union_all
from shapely.geometry import box

from . import capture as plan, config, earth, earth_capture, geo
from .detector import PalmDetector
from .farms import DARK_BRIGHTNESS, FarmService, _lift_shadows
from .imagery import Capture

log = logging.getLogger("palmai.regions")

RANGES = (600, 800, 1000)
MODES = ("farms", "region")
MARGIN_DEG = 0.0002          # ~20 m around farms
GRID_OVERLAP = 0.12
SECONDS_PER_CELL = 90        # measured; shown in the estimate
DATE_SECONDS = 150
EDGE_PX = 3                  # a detection this close to a cell border is cut off
INTERIOR_MARGIN = 0.03
MERGE_DISTANCE_M = 1.5       # palms closer than this seen in two cells are the same palm
DARK_CELL = 45               # cell brightness below this is reported as too dark
REGIONS_DIR = config.DATA_DIR / "regions"
OUTPUT_DIR = config.RESULTS_DIR / "regions"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def plan_region(service: FarmService, region: str, mode: str, range_m: int) -> tuple[list[plan.Cell], list[geo.Farm]]:
    if mode not in MODES:
        raise ValueError("Mode must be 'farms' or 'region'.")
    if range_m not in RANGES:
        raise ValueError(f"Camera range must be one of {RANGES} m.")
    boundaries = service.boundaries
    farms = boundaries.farms_in(region)
    if not farms:
        raise ValueError(f"No farms found for region {region!r}.")
    if mode == "region":
        if region not in boundaries.regions:
            raise ValueError(f"No region polygon for {region!r} in the KML.")
        area = union_all([boundaries.regions[region].buffer(MARGIN_DEG)] + [f.geometry for f in farms])
    else:
        area = union_all([f.geometry.buffer(MARGIN_DEG) for f in farms])
    return plan.plan_grid(area, range_m, GRID_OVERLAP, name_prefix=slug(region)), farms


def estimate(service: FarmService, region: str, mode: str, range_m: int, auto_date: bool = True) -> dict:
    cells, farms = plan_region(service, region, mode, range_m)
    seconds = len(cells) * SECONDS_PER_CELL + (DATE_SECONDS if auto_date else 0) + len(cells) * 6 + len(farms) * 4
    other = "region" if mode == "farms" else "farms"
    other_cells = len(plan_region(service, region, other, range_m)[0])
    return {"region": region, "mode": mode, "range_m": range_m, "captures": len(cells), "farms": len(farms),
            "minutes": round(seconds / 60), "other_mode_captures": other_cells,
            "palm_px": round(7.0 / (0.0846 * range_m / 600)), "too_many": len(cells) > config.REGION_MAX_CAPTURES,
            "limit": config.REGION_MAX_CAPTURES}


def _runs(region: str, mode: str, range_m: int) -> list[Path]:
    base = REGIONS_DIR / f"{slug(region)}_{mode}_{range_m}"
    return sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name) if base.exists() else []


def unfinished_run(region: str, mode: str, range_m: int) -> dict | None:
    for run in reversed(_runs(region, mode, range_m)):
        meta = json.loads((run / "run.json").read_text(encoding="utf-8")) if (run / "run.json").exists() else {}
        if meta.get("status") != "complete":
            done = len(json.loads((run / "manifest.json").read_text(encoding="utf-8"))["captures"]) if (run / "manifest.json").exists() else 0
            return {"run_id": run.name, "captured": done, "total": meta.get("cells"), "status": meta.get("status")}
    return None


def latest_result(region: str) -> dict | None:
    path = OUTPUT_DIR / slug(region) / "latest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _manifest(run_dir: Path) -> list[dict]:
    path = run_dir / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))["captures"] if path.exists() else []


def run_region(service: FarmService, detector: PalmDetector, region: str, mode: str, range_m: int, model: str,
               confidence: float, auto_date: bool, resume: bool, job: dict, cancelled) -> dict:
    cells, farms = plan_region(service, region, mode, range_m)
    if len(cells) > config.REGION_MAX_CAPTURES:
        raise ValueError(f"This needs {len(cells)} captures (limit {config.REGION_MAX_CAPTURES}). Use a higher camera or farm-areas-only mode.")
    started = time.perf_counter()
    prior = unfinished_run(region, mode, range_m) if resume else None
    run_id = prior["run_id"] if prior else datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = REGIONS_DIR / f"{slug(region)}_{mode}_{range_m}" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_file = run_dir / "run.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if prior and meta_file.exists() else {}
    meta.update(region=region, mode=mode, range_m=range_m, cells=len(cells), status="capturing", started=meta.get("started", datetime.now().isoformat(timespec="seconds")))
    meta_file.write_text(json.dumps(meta), encoding="utf-8")

    n, m = len(cells), len(farms)
    job["total"] = n + n + m
    captured = {c["name"] for c in _manifest(run_dir) if (run_dir / c["file"]).exists()}

    # 1. imagery date --------------------------------------------------------------------------------
    if not captured:
        job.update(phase="date", message="Preparing Google Earth: shading off, choosing a well-lit imagery date…", done=0)
        mid = cells[len(cells) // 2]
        try:
            meta["date_choice"] = earth_capture.prepare_earth(mid.lon, mid.lat, range_m, auto_date, cancelled)
        except earth_capture.CaptureError as exc:
            raise ValueError(str(exc)) from exc
        meta_file.write_text(json.dumps(meta), encoding="utf-8")
    if cancelled():
        return {"cancelled": True}

    # 2. capture the grid ------------------------------------------------------------------------------
    try:
        for i, cell in enumerate(cells):
            if cell.name in captured:
                continue
            if cancelled():
                break
            job.update(phase="capture", done=i, message=f"Capturing {i + 1} of {n} from Google Earth — please don't use the mouse or keyboard")
            try:
                earth_capture.capture_view(cell.lon, cell.lat, cell.range_m, cell.name, cancelled, out_dir=run_dir,
                                           keep_camera=True, settle_min=6)
            except earth_capture.CaptureError as exc:
                meta.update(status="interrupted", error=str(exc))
                meta_file.write_text(json.dumps(meta), encoding="utf-8")
                raise ValueError(f"{exc} ({i} of {n} captures are saved; run again with “Resume” to continue.)") from exc
    finally:
        earth.camera.clear()
    if cancelled():
        meta["status"] = "interrupted"
        meta_file.write_text(json.dumps(meta), encoding="utf-8")
        return {"cancelled": True, "captured": len(_manifest(run_dir)), "total": n}

    entries = [e for e in _manifest(run_dir) if (run_dir / e["file"]).exists()]
    caps = [Capture(e["name"], run_dir / e["file"], tuple(e["bounds"]), e["width"], e["height"], "", e.get("imagery_date", ""),
                    e.get("captured_at", ""), e.get("status_image", ""), "live") for e in entries]
    brightness = {e["name"]: e.get("brightness") for e in entries}
    dark_cells = [name for name, b in brightness.items() if b is not None and b < DARK_CELL]

    # 3. palms outside farms -------------------------------------------------------------------------------
    meta["status"] = "detecting"
    meta_file.write_text(json.dumps(meta), encoding="utf-8")
    farm_union = union_all([f.geometry for f in farms])
    region_poly = service.boundaries.regions.get(region)
    outside = _detect_outside(caps, brightness, detector, model, confidence, farm_union, region_poly, job, n, cancelled)
    if cancelled():
        return {"cancelled": True}

    # 4. count every farm -----------------------------------------------------------------------------------
    meta["status"] = "counting"
    meta_file.write_text(json.dumps(meta), encoding="utf-8")
    order = sorted(farms, key=lambda f: (-round(f.geometry.centroid.y, 3), f.geometry.centroid.x))   # neighbours share cells (cache hits)
    results, skipped = [], []
    for k, farm in enumerate(order):
        if cancelled():
            break
        job.update(phase="count", done=2 * n + k, message=f"Counting farms: {farm.code} ({k + 1} of {m})")
        try:
            r = service.analyze(farm.code, detector, model, confidence, "live", False, lambda *_: None, cancelled, caps_override=caps)
            row = {k2: r[k2] for k2 in ("code", "count", "area_ha", "palms_per_ha", "coverage_pct", "kind")}
            job["results"].append(row)
            results.append({**row, "type": farm.type, "status": farm.status, "mean_confidence": r["mean_confidence"],
                            "imagery_quality": r["imagery_quality"], "warning": r["warning"]})
        except ValueError as exc:
            skipped.append({"code": farm.code, "reason": str(exc)})
    status = "interrupted" if cancelled() else "complete"
    meta["status"] = status
    meta_file.write_text(json.dumps(meta), encoding="utf-8")

    summary = _write_outputs(region, mode, range_m, run_id, model, confidence, meta, results, skipped, outside, dark_cells, n,
                             time.perf_counter() - started)
    if status == "complete":
        _prune_old_runs(region, mode, range_m, run_id)
    return summary


def _detect_outside(caps: list[Capture], brightness: dict, detector: PalmDetector, model: str, confidence: float,
                    farm_union, region_poly, job: dict, n_cells: int, cancelled) -> dict:
    lons, lats, confs = [], [], []
    for i, cap in enumerate(caps):
        if cancelled():
            break
        job.update(phase="detect", done=n_cells + i, message=f"Finding palms outside farms: capture {i + 1} of {len(caps)}")
        image = cv2.imread(str(cap.path))
        if image is None:
            continue
        h, w = image.shape[:2]
        gi = geo.GeoImage(*cap.bounds, w, h)
        b_val = brightness.get(cap.name)
        detect_on = _lift_shadows(image, b_val) if b_val is not None and b_val < DARK_BRIGHTNESS else image
        det = detector.detect(detect_on, confidence, "auto", model, palm_px=config.EXPECTED_PALM_M / gi.metres_per_pixel)
        b = det.boxes
        if not len(b):
            continue
        cut = (b[:, 0] <= EDGE_PX) | (b[:, 1] <= EDGE_PX) | (b[:, 2] >= w - EDGE_PX) | (b[:, 3] >= h - EDGE_PX)
        lon, lat = gi.pixel_to_lonlat((b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2)
        drop = np.zeros(len(b), bool)
        for other in caps:                      # a palm cut by this cell's border is dropped when a neighbour sees it whole
            if other.name == cap.name:
                continue
            west, south, east, north = other.bounds
            mx, my = (east - west) * INTERIOR_MARGIN, (north - south) * INTERIOR_MARGIN
            drop |= cut & (lon > west + mx) & (lon < east - mx) & (lat > south + my) & (lat < north - my)
        lons.append(lon[~drop]); lats.append(lat[~drop]); confs.append(b[~drop, 4])
        del image
    if not lons:
        return {"lon": np.array([]), "lat": np.array([]), "conf": np.array([])}
    lon, lat, conf = np.concatenate(lons), np.concatenate(lats), np.concatenate(confs)
    keep = _merge_duplicates(lon, lat, conf)
    lon, lat, conf = lon[keep], lat[keep], conf[keep]
    outside = ~geo.points_in_geometry(farm_union, lon, lat)
    if region_poly is not None:
        outside &= geo.points_in_geometry(region_poly, lon, lat)
    return {"lon": lon[outside], "lat": lat[outside], "conf": conf[outside]}


def _merge_duplicates(lon: np.ndarray, lat: np.ndarray, conf: np.ndarray) -> np.ndarray:
    if len(lon) < 2:
        return np.arange(len(lon))
    per_lon, per_lat = geo.metres_per_degree(float(np.mean(lat)))
    xy = np.column_stack([lon * per_lon, lat * per_lat])
    tree = cKDTree(xy)
    removed = np.zeros(len(lon), bool)
    for i in np.argsort(-conf):
        if removed[i]:
            continue
        for j in tree.query_ball_point(xy[i], MERGE_DISTANCE_M):
            if j != i:
                removed[j] = True
    return np.where(~removed)[0]


def _write_outputs(region, mode, range_m, run_id, model, confidence, meta, results, skipped, outside, dark_cells, n_cells, seconds) -> dict:
    out = OUTPUT_DIR / slug(region) / run_id
    out.mkdir(parents=True, exist_ok=True)
    total = int(sum(r["count"] for r in results))
    outside_n = int(len(outside["lon"]))
    summary = {
        "region": region, "mode": mode, "range_m": range_m, "run_id": run_id, "model": model, "confidence": confidence,
        "created": datetime.now().isoformat(timespec="seconds"), "captures": n_cells, "dark_captures": dark_cells,
        "date_choice": meta.get("date_choice"), "farms_counted": len(results), "farms_total_palms": total,
        "palms_outside_farms": outside_n, "palms_total": total + outside_n,
        "farms_skipped": skipped, "farms_with_warning": [r["code"] for r in results if r["warning"]],
        "minutes": round(seconds / 60, 1), "farms": results,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (OUTPUT_DIR / slug(region) / "latest.json").write_text(json.dumps(summary), encoding="utf-8")
    with open(out / "farms.csv", "w", newline="", encoding="utf-8-sig") as fh:
        fields = ["code", "type", "status", "count", "area_ha", "palms_per_ha", "mean_confidence", "coverage_pct", "imagery_quality"]
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: r["code"]))
        fh.write(f"TOTAL,,,{total}\nOUTSIDE FARMS,,,{outside_n}\nALL PALMS,,,{total + outside_n}\n")
    with open(out / "outside_farms.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["lon", "lat", "confidence"])
        for x, y, c in zip(outside["lon"], outside["lat"], outside["conf"]):
            writer.writerow([f"{x:.7f}", f"{y:.7f}", f"{c:.3f}"])
    points = "".join(f"<Placemark><styleUrl>#p</styleUrl><Point><coordinates>{x:.7f},{y:.7f},0</coordinates></Point></Placemark>"
                     for x, y in zip(outside["lon"], outside["lat"]))
    (out / "outside_farms.kml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        '<Style id="p"><IconStyle><scale>0.4</scale><color>ffeb8f2f</color><Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon></IconStyle>'
        f'<LabelStyle><scale>0</scale></LabelStyle></Style><name>Palms outside farms - {region}</name>{points}</Document></kml>', encoding="utf-8")
    return summary


def _prune_old_runs(region: str, mode: str, range_m: int, keep: str) -> None:
    """Only the newest finished run per region/mode/camera is kept (cell captures are large)."""
    if config.REGION_KEEP_ALL_RUNS:
        return
    for run in _runs(region, mode, range_m):
        if run.name != keep:
            shutil.rmtree(run, ignore_errors=True)
