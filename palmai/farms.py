"""Farm service: farm polygons from the KML + imagery -> palm count per polygon.

Any polygon is analyzed the same way (a farm from the KML, or an area the user drew):
  1. get imagery for it (live from Google Earth Pro, or stored captures) and stitch it into one crop,
  2. run the detector on the crop and keep the palms whose centre lies inside the polygon,
  3. save a labelled image (polygon outlined, palms dotted, code + count) and the result.
"""
import csv
import io
import json
import logging
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from shapely import STRtree, union_all
from shapely.geometry import box, mapping, shape
from shapely.validation import make_valid

from . import config, geo, imagery
from .detector import PalmDetector
from .imagery import Capture

log = logging.getLogger("palmai.farms")

OUTSIDE_DIM = 0.45          # brightness of the area outside the polygon in result images
OUTLINE_BGR = (0, 215, 255)
DOT_BGR = (30, 30, 235)
RESULTS_SUBDIR = "farms"
MAX_AREA_KM2 = 25
SAVED_DIR = config.DATA_DIR / "saved"      # data/saved/<farm code>/<time>.jpg + .json


DARK_BRIGHTNESS = 60        # mean grey level inside the polygon below which palms start to vanish in shadow
VERY_DARK_BRIGHTNESS = 40   # below this the count is unreliable


def _lift_shadows(image: np.ndarray, brightness: float) -> np.ndarray:
    """Gamma-brighten dark imagery so the detector can see crowns in shadow (used for detection only)."""
    gamma = float(np.clip(np.log(0.35) / np.log(max(brightness, 4) / 255), 0.4, 1.0))
    lut = np.array([255 * ((i / 255) ** gamma) for i in range(256)], np.uint8)
    return cv2.LUT(image, lut)


class Cancelled(Exception):
    pass


class _ImageCache:
    """Keeps a few decoded captures in memory (8K images are ~100 MB each)."""

    def __init__(self, size: int = 3) -> None:
        self.size, self._items = size, OrderedDict()

    def get(self, capture: Capture) -> np.ndarray:
        key = (capture.name, capture.path.stat().st_mtime)
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        image = cv2.imread(str(capture.path))
        if image is None:
            raise OSError(f"Cannot read {capture.path.name}")
        self._items[key] = image
        while len(self._items) > self.size:
            self._items.popitem(last=False)
        return image


class FarmService:
    def __init__(self) -> None:
        self.farms_file = config.DATA_DIR / "farms.json"
        self.areas_file = config.DATA_DIR / "areas.json"
        self.results_dir = config.RESULTS_DIR / RESULTS_SUBDIR
        self._lock = threading.RLock()
        self._images = _ImageCache()
        self._boundaries: geo.Boundaries | None = None
        self._boundaries_mtime = 0.0
        self._coverage: dict[str, float] = {}
        self._coverage_key: tuple | None = None
        self._summary: dict[str, dict] | None = None
        self.jobs = JobManager(self)

    # --- data -----------------------------------------------------------------------
    @property
    def boundaries(self) -> geo.Boundaries:
        if not self.farms_file.exists():
            raise ValueError("No farm boundaries imported yet. Run: python tools/import_kml.py <your KML>")
        mtime = self.farms_file.stat().st_mtime
        with self._lock:
            if self._boundaries is None or mtime != self._boundaries_mtime:
                self._boundaries = geo.Boundaries.load(self.farms_file)
                self._boundaries_mtime = mtime
                self._coverage_key = None
            return self._boundaries

    def captures(self, source: str = "all") -> list[Capture]:
        stored = imagery.stored_captures() if source in ("all", "stored") else []
        live = imagery.live_captures() if source in ("all", "live") else []
        return stored + live + (self._saved_overlays() if source in ("all", "saved") else [])

    def _saved_overlays(self) -> list[Capture]:
        """Newest saved capture of every polygon, drawn on the map."""
        out = []
        for folder in SAVED_DIR.iterdir() if SAVED_DIR.exists() else []:
            saved = self.saved_captures(folder.name)
            if saved:
                m = saved[0]
                out.append(Capture(m["name"], folder / m["file"], tuple(m["bounds"]), m["width"], m["height"], folder.name,
                                   f"saved {m['captured_at'][:10]}", m["captured_at"], "", "saved"))
        return out

    def available_captures(self, source: str = "all") -> list[Capture]:
        return [c for c in self.captures(source) if c.available]

    def farm(self, code: str) -> geo.Farm:
        farm = self.boundaries.farms.get(code.upper()) or self.boundaries.farms.get(code)
        if farm is None:
            raise KeyError(code)
        return farm

    def coverage(self) -> dict[str, float]:
        """Percentage of every farm's area that the stored captures cover."""
        caps = self.available_captures("stored")
        key = (tuple(sorted(c.name for c in caps)), self._boundaries_mtime)
        with self._lock:
            if key != self._coverage_key:
                farms = list(self.boundaries.farms.values())
                if caps:
                    covered = union_all([c.box for c in caps])
                    tree = STRtree([f.geometry for f in farms])
                    near = set(tree.query(covered))
                    self._coverage = {f.code: (100 * f.geometry.intersection(covered).area / f.geometry.area
                                               if i in near and f.geometry.area else 0.0) for i, f in enumerate(farms)}
                else:
                    self._coverage = {f.code: 0.0 for f in farms}
                self._coverage_key = key
            return self._coverage

    # --- areas drawn by the user ------------------------------------------------------------
    def areas(self) -> dict[str, dict]:
        return json.loads(self.areas_file.read_text(encoding="utf-8")) if self.areas_file.exists() else {}

    def add_area(self, name: str, geometry: dict) -> dict:
        try:
            geom = make_valid(shape(geometry))
        except Exception as exc:  # noqa: BLE001
            raise ValueError("That is not a valid polygon.") from exc
        polys = [g for g in getattr(geom, "geoms", [geom]) if g.geom_type == "Polygon" and not g.is_empty]
        if not polys:
            raise ValueError("Draw a polygon with at least three corners.")
        geom = union_all(polys)
        km2 = geo.area_m2(geom) / 1e6
        if km2 > MAX_AREA_KM2:
            raise ValueError(f"That area is {km2:.1f} km²; the limit is {MAX_AREA_KM2} km². Draw a smaller one.")
        with self._lock:
            areas = self.areas()
            n = 1 + max([int(k.split("-")[1]) for k in areas] or [0])
            key = f"AREA-{n:04d}"
            areas[key] = {"code": key, "name": (name or key).strip()[:60], "geometry": mapping(geom), "created": datetime.now().isoformat(timespec="seconds")}
            self.areas_file.parent.mkdir(parents=True, exist_ok=True)
            self.areas_file.write_text(json.dumps(areas), encoding="utf-8")
        return {k: v for k, v in areas[key].items()} | {"area_ha": round(km2 * 100, 2)}

    def delete_area(self, key: str) -> None:
        with self._lock:
            areas = self.areas()
            if areas.pop(key, None) is None:
                raise KeyError(key)
            self.areas_file.write_text(json.dumps(areas), encoding="utf-8")
            self.summary().pop(key, None)
            self._write_summary()

    def target(self, key: str) -> tuple[geo.Farm, dict]:
        """A farm from the KML or a drawn area, as (geometry holder, extra info)."""
        if key.upper().startswith("AREA-"):
            a = self.areas().get(key.upper())
            if a is None:
                raise KeyError(key)
            geom = make_valid(shape(a["geometry"]))
            return geo.Farm(a["code"], "Custom area", a["name"], "Custom", geom), {"kind": "area", "name": a["name"]}
        return self.farm(key), {"kind": "farm"}

    # --- saved results ----------------------------------------------------------------
    def summary(self) -> dict[str, dict]:
        with self._lock:
            if self._summary is None:
                path = self.results_dir / "summary.json"
                self._summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            return self._summary

    def _write_summary(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "summary.json").write_text(json.dumps(self._summary), encoding="utf-8")

    def _save_result(self, result: dict) -> None:
        with self._lock:
            self.summary()[result["code"]] = {k: v for k, v in result.items() if k != "points"}
            self._write_summary()
            (self.results_dir / "points").mkdir(exist_ok=True)
            (self.results_dir / "points" / f"{result['code']}.json").write_text(json.dumps(result["points"]), encoding="utf-8")
            # Keep every run: the farm's capture (clean + with palms) and its detection, under runs/<code>/<time>/
            run = self.results_dir / "runs" / result["code"] / datetime.now().strftime("%Y%m%d-%H%M%S")
            run.mkdir(parents=True, exist_ok=True)
            for name, key in (("capture.jpg", "image_raw"), ("detection.jpg", "image")):
                src = config.RESULTS_DIR / result[key].removeprefix("/results/")
                if src.exists():
                    shutil.copy2(src, run / name)
            (run / "result.json").write_text(json.dumps(result), encoding="utf-8")
            result["history"] = len(list((self.results_dir / "runs" / result["code"]).iterdir()))

    def saved_points(self, code: str) -> list:
        path = self.results_dir / "points" / f"{code}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    # --- map data -----------------------------------------------------------------------
    def farms_geojson(self, region: str | None = None) -> dict:
        coverage, done = self.coverage(), self.summary()
        features = []
        for f in self.boundaries.farms.values():
            if region and f.region != region:
                continue
            saved = done.get(f.code)
            features.append({"type": "Feature", "geometry": mapping(f.geometry.simplify(0.000004)), "properties": {
                "code": f.code, "type": f.type, "status": f.status, "region": f.region, "area_ha": round(f.area_ha, 3),
                "coverage": round(coverage.get(f.code, 0), 1), "palms": saved["count"] if saved else None}})
        return {"type": "FeatureCollection", "features": features}

    def areas_geojson(self) -> dict:
        done = self.summary()
        return {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": a["geometry"], "properties": {
                "code": k, "name": a["name"], "area_ha": round(geo.area_m2(shape(a["geometry"])) / 1e4, 2),
                "palms": done[k]["count"] if k in done else None}} for k, a in self.areas().items()]}

    def regions_geojson(self) -> dict:
        return {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": mapping(g.simplify(0.00003)), "properties": {"name": name}}
            for name, g in self.boundaries.regions.items()]}

    def meta(self) -> dict:
        coverage, done = self.coverage(), self.summary()
        regions = {}
        for f in self.boundaries.farms.values():
            r = regions.setdefault(f.region, {"name": f.region, "farms": 0, "captured": 0, "analyzed": 0, "palms": 0})
            r["farms"] += 1
            r["captured"] += coverage.get(f.code, 0) >= 99.5
            if f.code in done:
                r["analyzed"] += 1
                r["palms"] += done[f.code]["count"]
        stored, live = self.captures("stored"), self.available_captures("live")
        farm_total = sum(r["palms"] for r in regions.values())
        return {"regions": sorted(regions.values(), key=lambda r: r["name"]), "farms": len(self.boundaries.farms),
                "captures_planned": len(stored), "captures_ready": sum(c.available for c in stored), "live_captures": len(live),
                "saved_polygons": len(self._saved_overlays()),
                "imagery_dates": sorted({c.imagery_date for c in stored if c.imagery_date}),
                "source_kml": self.boundaries.source, "total_palms": farm_total,
                "analyzed": sum(r["analyzed"] for r in regions.values()),
                "areas": len(self.areas())}

    def preview(self, name: str) -> Path:
        """Downscaled copy of a capture for the map (generated once)."""
        cap = next((c for c in self.captures() if c.name == name and c.available), None)
        if cap is None:
            raise KeyError(name)
        out = config.DATA_DIR / "previews" / f"{name}.jpg"
        if not out.exists() or out.stat().st_mtime < cap.path.stat().st_mtime:
            image = self._images.get(cap)
            grey_mean = float(cv2.cvtColor(cv2.resize(image, (256, 135)), cv2.COLOR_BGR2GRAY).mean())
            if grey_mean < 90:                                 # dark imagery would show as a black rectangle on the map
                gamma = float(np.clip(np.log(0.42) / np.log(max(grey_mean, 4) / 255), 0.35, 1.0))
                image = cv2.LUT(image, np.array([255 * ((i / 255) ** gamma) for i in range(256)], np.uint8))
            scale = config.PREVIEW_PX / image.shape[1]
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
        return out

    def capture_plan(self, codes: list[str]) -> dict:
        """Stored captures still missing for these farms (used when the imagery source is 'stored')."""
        coverage = self.coverage()
        missing = [c for c in codes if coverage.get(c, 0) < 99.5]
        needed = set()
        for code in missing:
            farm = self.farm(code)
            for cap in self.captures("stored"):
                if not cap.available and cap.box.intersects(farm.geometry):
                    needed.add(cap.name)
        return {"farms_missing_imagery": missing, "captures_to_take": sorted(needed)}

    # --- estimate ---------------------------------------------------------------------------
    def view_for(self, farm: geo.Farm) -> tuple[float, float, float, float]:
        """The padded bounding box around a polygon that gets cropped and counted."""
        west, south, east, north = farm.geometry.bounds
        per_lon, per_lat = geo.metres_per_degree((south + north) / 2)
        pad_lon = max((east - west) * config.FARM_MARGIN, config.FARM_MARGIN_MIN_M / per_lon)
        pad_lat = max((north - south) * config.FARM_MARGIN, config.FARM_MARGIN_MIN_M / per_lat)
        return west - pad_lon, south - pad_lat, east + pad_lon, north + pad_lat

    def view_of(self, keys: list[str]):
        return union_all([box(*self.view_for(self.target(k)[0])) for k in keys])

    def estimate(self, keys: list[str], source: str, refresh: bool) -> dict:
        if source == "saved":
            return {"source": "saved", "captures_needed": 0, "minutes": 0,
                    "farms_without_imagery": [k for k in keys if not self.saved_captures(k)]}
        return imagery.estimate(self.view_of(keys), source, refresh)

    # --- saved captures (one folder per farm / area, kept until deleted) ----------------------------
    def saved_captures(self, code: str) -> list[dict]:
        """Captures saved for a polygon, newest first."""
        folder = SAVED_DIR / code.upper()
        if not folder.exists():
            return []
        out = []
        for meta_file in folder.glob("*.json"):
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if (folder / meta["file"]).exists():
                out.append(meta)
        return sorted(out, key=lambda m: m["id"], reverse=True)

    def delete_saved_capture(self, code: str, capture_id: str) -> None:
        folder = SAVED_DIR / code.upper()
        meta_file = folder / f"{capture_id}.json"
        if not meta_file.exists():
            raise KeyError(capture_id)
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        for name in (meta["file"], meta.get("status_image") or "", meta_file.name):
            if name:
                (folder / name).unlink(missing_ok=True)
        (config.DATA_DIR / "previews" / f"{meta['name']}.jpg").unlink(missing_ok=True)

    def _save_capture(self, code: str, canvas: np.ndarray, out_geo: geo.GeoImage, caps: list[Capture], imagery_date: str) -> dict:
        """Keep the stitched capture of a polygon so it can be counted again later without Google Earth."""
        folder = SAVED_DIR / code.upper()
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now()
        cid = stamp.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(str(folder / f"{cid}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 93])
        status_cap = next((c for c in caps if c.status_image and (c.path.parent / c.status_image).exists()), None)
        status = ""
        if status_cap is not None:
            shutil.copy2(status_cap.path.parent / status_cap.status_image, folder / f"{cid}_status.png")
            status = f"{cid}_status.png"
        meta = {"id": cid, "name": f"{code.upper()}_{cid}", "file": f"{cid}.jpg", "status_image": status,
                "bounds": [out_geo.west, out_geo.south, out_geo.east, out_geo.north], "width": canvas.shape[1],
                "height": canvas.shape[0], "captured_at": stamp.isoformat(timespec="seconds"), "source": "Google Earth Pro",
                "m_per_pixel": round(out_geo.metres_per_pixel, 4), "size_mb": round((folder / f"{cid}.jpg").stat().st_size / 1e6, 1)}
        (folder / f"{cid}.json").write_text(json.dumps(meta), encoding="utf-8")
        return meta

    def _load_saved(self, code: str, capture_id: str) -> tuple[np.ndarray, geo.GeoImage, dict]:
        saved = self.saved_captures(code)
        if not saved:
            raise ValueError(f"No saved capture for {code} yet. Use “Capture new from Google Earth” once; it is saved for later.")
        meta = next((m for m in saved if m["id"] == capture_id), None) if capture_id else saved[0]
        if meta is None:
            raise ValueError(f"Saved capture {capture_id} was not found for {code}.")
        image = cv2.imread(str(SAVED_DIR / code.upper() / meta["file"]))
        if image is None:
            raise ValueError(f"The saved capture {meta['id']} for {code} could not be read.")
        return image, geo.GeoImage(*meta["bounds"], image.shape[1], image.shape[0]), meta

    # --- analysis -----------------------------------------------------------------------
    def analyze(self, key: str, detector: PalmDetector, model: str, confidence: float, source: str = "live",
                refresh: bool = False, progress=lambda d, t, m: None, cancelled=lambda: False,
                imagery_date: str = "", capture_id: str = "", caps_override: list[Capture] | None = None) -> dict:
        """source "live": capture new imagery from Google Earth and save it; "saved": count a capture saved earlier.

        `caps_override` supplies captures that were just taken for a whole region (the farm is cropped from them and
        the crop is saved for the farm, exactly as for a single-farm capture).
        """
        farm, info = self.target(key)
        view = self.view_for(farm)
        saved_meta: dict | None = None

        if source == "saved":
            canvas, out_geo, saved_meta = self._load_saved(farm.code, capture_id)
            caps = [Capture(saved_meta["name"], SAVED_DIR / farm.code.upper() / saved_meta["file"], tuple(saved_meta["bounds"]),
                            saved_meta["width"], saved_meta["height"], "", f"saved {saved_meta['captured_at'][:10]}",
                            saved_meta["captured_at"], "", "saved")]
        else:
            if caps_override is not None:
                caps = [c for c in caps_override if c.box.intersects(box(*view))]
            else:
                caps = imagery.ensure_live(box(*view), refresh, progress, cancelled, imagery_date)
            if not caps:
                raise ValueError("No imagery was available for this polygon.")
            canvas, out_geo = self._mosaic(view, caps, config.FARM_DETECT_MAX_PX)
            saved_meta = self._save_capture(farm.code, canvas, out_geo, caps, imagery_date)
        if cancelled():
            raise ValueError("Cancelled.")

        started = time.perf_counter()
        progress(0, 1, "Counting palms…")
        mpp = out_geo.metres_per_pixel
        brightness = self._brightness(canvas, out_geo, farm)
        quality = "good" if brightness >= DARK_BRIGHTNESS else ("dark" if brightness >= VERY_DARK_BRIGHTNESS else "very_dark")
        detect_on = _lift_shadows(canvas, brightness) if quality != "good" else canvas   # dark imagery hides palm crowns
        det = detector.detect(detect_on, confidence, "auto", model, palm_px=config.EXPECTED_PALM_M / mpp)
        b = det.boxes
        if len(b):
            lon, lat = out_geo.pixel_to_lonlat((b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2)
            inside = geo.points_in_geometry(farm.geometry, lon, lat)
            lon, lat, b = lon[inside], lat[inside], b[inside]
        else:
            lon = lat = np.array([])
        sizes_m = np.minimum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) * mpp if len(b) else np.array([])

        title = info.get("name") or farm.code
        self._write_images(farm, title, detect_on, out_geo, lon, lat, sizes_m, mpp)   # shown brightened when the imagery is dark
        area = farm.area_ha
        result = {
            "code": farm.code, "kind": info["kind"], "name": info.get("name"), "region": farm.region, "type": farm.type, "status": farm.status,
            "count": int(len(lon)), "area_ha": round(area, 3),
            "palms_per_ha": round(len(lon) / area, 1) if area > 0.01 else None,
            "mean_confidence": round(float(b[:, 4].mean()), 3) if len(b) else None,
            "coverage_pct": round(100 * min(1.0, farm.geometry.intersection(union_all([c.box for c in caps])).area / farm.geometry.area), 1) if farm.geometry.area else 0,
            "model": model, "confidence": confidence, "source": source,
            "imagery_brightness": round(brightness), "imagery_quality": quality,
            "warning": None if quality == "good" else (
                "The imagery for this polygon is very dark (palm crowns are hidden in shadow), so this count is a clear underestimate. "
                "Set Google Earth's Historical Imagery slider to a better-lit date and count again."
                if quality == "very_dark" else
                "The imagery for this polygon is dark, so some palms are probably missed. A better-lit Historical Imagery date in Google Earth will count more accurately."),
            "imagery": sorted({c.imagery_date for c in caps if c.imagery_date}),
            "captured_at": saved_meta["captured_at"], "capture_id": saved_meta["id"],
            "imagery_status_images": [f"/saved/{farm.code.upper()}/{saved_meta['status_image']}"] if saved_meta.get("status_image") else [],
            "m_per_pixel": round(mpp, 3), "window": det.window, "device": detector.device_name,
            "analyzed": datetime.now().isoformat(timespec="seconds"), "seconds": round(time.perf_counter() - started, 1),
            "image": f"/results/{RESULTS_SUBDIR}/images/{farm.code}.jpg", "image_raw": f"/results/{RESULTS_SUBDIR}/raw/{farm.code}.jpg",
            "points": [[round(float(x), 7), round(float(y), 7)] for x, y in zip(lon, lat)],
        }
        self._save_result(result)
        return result

    # kept for callers that only deal with farms
    analyze_farm = analyze

    @staticmethod
    def _brightness(canvas: np.ndarray, out_geo: geo.GeoImage, farm: geo.Farm) -> float:
        """Mean grey level (0-255) of the imagery inside the polygon."""
        h, w = canvas.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        for poly in [p for p in getattr(farm.geometry, "geoms", [farm.geometry]) if p.geom_type == "Polygon"]:
            x, y = out_geo.lonlat_to_pixel(*map(np.array, zip(*poly.exterior.coords)))
            cv2.fillPoly(mask, [np.column_stack([x, y]).astype(np.int32)], 255)
        grey = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        return float(grey[mask > 0].mean()) if mask.any() else float(grey.mean())

    def _mosaic(self, view: tuple, caps: list[Capture], max_px: int) -> tuple[np.ndarray, geo.GeoImage]:
        """Stitch the captures covering `view` into one north-up image (finest resolution, capped)."""
        per_lon, per_lat = geo.metres_per_degree((view[1] + view[3]) / 2)
        finest = min(c.geo.metres_per_pixel for c in caps)
        w_px = (view[2] - view[0]) * per_lon / finest
        h_px = (view[3] - view[1]) * per_lat / finest
        scale = min(1.0, max_px / max(w_px, h_px))
        out_w, out_h = max(2, int(w_px * scale)), max(2, int(h_px * scale))
        out_geo = geo.GeoImage(*view, out_w, out_h)
        canvas = np.full((out_h, out_w, 3), 40, np.uint8)
        for cap in sorted(caps, key=lambda c: (-c.geo.metres_per_pixel, c.captured_at)):  # finest / newest pasted last, so it wins
            image = self._images.get(cap)
            gi = cap.geo
            west, east = max(view[0], cap.bounds[0]), min(view[2], cap.bounds[2])
            south, north = max(view[1], cap.bounds[1]), min(view[3], cap.bounds[3])
            if west >= east or south >= north:
                continue
            sx0, sy1 = gi.lonlat_to_pixel(west, south)
            sx1, sy0 = gi.lonlat_to_pixel(east, north)
            dx0, dy1 = out_geo.lonlat_to_pixel(west, south)
            dx1, dy0 = out_geo.lonlat_to_pixel(east, north)
            sx0, sy0, sx1, sy1 = (int(round(float(v))) for v in (sx0, sy0, sx1, sy1))
            dx0, dy0, dx1, dy1 = (max(0, int(round(float(v)))) for v in (dx0, dy0, dx1, dy1))
            dx1, dy1 = min(dx1, out_w), min(dy1, out_h)
            if sx1 - sx0 < 2 or sy1 - sy0 < 2 or dx1 - dx0 < 2 or dy1 - dy0 < 2:
                continue
            piece = image[max(sy0, 0):sy1, max(sx0, 0):sx1]
            canvas[dy0:dy1, dx0:dx1] = cv2.resize(piece, (dx1 - dx0, dy1 - dy0), interpolation=cv2.INTER_AREA)
        return canvas, out_geo

    def _write_images(self, farm: geo.Farm, title: str, canvas: np.ndarray, out_geo: geo.GeoImage,
                      lon: np.ndarray, lat: np.ndarray, sizes_m: np.ndarray, mpp: float) -> None:
        scale = min(1.0, config.FARM_IMAGE_MAX_PX / max(canvas.shape[:2]))
        if scale < 1:
            canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = canvas.shape[:2]
        gi = geo.GeoImage(out_geo.west, out_geo.south, out_geo.east, out_geo.north, w, h)
        polygons = [p for p in getattr(farm.geometry, "geoms", [farm.geometry]) if p.geom_type == "Polygon"]
        mask = np.zeros((h, w), np.uint8)
        for poly in polygons:
            x, y = gi.lonlat_to_pixel(*map(np.array, zip(*poly.exterior.coords)))
            cv2.fillPoly(mask, [np.column_stack([x, y]).astype(np.int32)], 255)
            for hole in poly.interiors:
                hx, hy = gi.lonlat_to_pixel(*map(np.array, zip(*hole.coords)))
                cv2.fillPoly(mask, [np.column_stack([hx, hy]).astype(np.int32)], 0)
        shaded = canvas.copy()
        outside = mask == 0
        shaded[outside] = (canvas[outside] * OUTSIDE_DIM).astype(np.uint8)
        thickness = max(2, w // 450)

        def outline(img):
            for poly in polygons:
                x, y = gi.lonlat_to_pixel(*map(np.array, zip(*poly.exterior.coords)))
                cv2.polylines(img, [np.column_stack([x, y]).astype(np.int32)], True, OUTLINE_BGR, thickness, cv2.LINE_AA)

        for sub in ("images", "raw"):
            (self.results_dir / sub).mkdir(parents=True, exist_ok=True)
        raw = shaded.copy()
        outline(raw)
        cv2.imwrite(str(self.results_dir / "raw" / f"{farm.code}.jpg"), _banner(raw, f"{title}   {farm.area_ha:.2f} ha"), [cv2.IMWRITE_JPEG_QUALITY, 90])

        annotated = shaded
        outline(annotated)
        if len(lon):
            px, py = gi.lonlat_to_pixel(lon, lat)
            radius = int(np.clip(np.median(sizes_m) / (mpp / scale) * 0.1, 3, 16))
            for x, y in zip(px, py):
                cv2.circle(annotated, (int(x), int(y)), radius + max(1, radius // 4), (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(annotated, (int(x), int(y)), radius, DOT_BGR, -1, cv2.LINE_AA)
        cv2.imwrite(str(self.results_dir / "images" / f"{farm.code}.jpg"),
                    _banner(annotated, f"{title}   {len(lon):,} palms   {farm.area_ha:.2f} ha"), [cv2.IMWRITE_JPEG_QUALITY, 90])

    # --- export -------------------------------------------------------------------------
    def export_csv(self, region: str | None = None, kind: str = "farm") -> str:
        rows = [r for r in self.summary().values() if r.get("kind", "farm") == kind and (not region or r["region"] == region)]
        rows.sort(key=lambda r: r["code"])
        buffer = io.StringIO()
        fields = ["code", "name", "region", "type", "status", "count", "area_ha", "palms_per_ha", "mean_confidence",
                  "coverage_pct", "source", "captured_at", "model", "confidence", "analyzed"]
        writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        buffer.write(f"TOTAL,,,,,{sum(r['count'] for r in rows)},{round(sum(r['area_ha'] for r in rows), 3)}\n")
        return buffer.getvalue()


def _banner(image: np.ndarray, text: str) -> np.ndarray:
    """Add a dark title strip with the code/name and count above the image."""
    h, w = image.shape[:2]
    banner_h = max(44, min(110, w // 14))
    scale = banner_h / 70
    width = max(w, 420)
    banner = np.full((banner_h, width, 3), 28, np.uint8)
    cv2.putText(banner, text, (int(14 * scale), int(banner_h * 0.68)), cv2.FONT_HERSHEY_SIMPLEX,
                0.95 * scale, (255, 255, 255), max(1, int(2 * scale)), cv2.LINE_AA)
    body = image if w >= width else cv2.copyMakeBorder(image, 0, 0, 0, width - w, cv2.BORDER_CONSTANT, value=(28, 28, 28))
    return np.vstack([banner, body])


class JobManager:
    """Runs analyses in a background thread; one at a time (the GPU and Google Earth are shared)."""

    def __init__(self, service: FarmService) -> None:
        self.service, self.jobs, self._lock = service, {}, threading.Lock()
        self._cancel: dict[str, threading.Event] = {}

    def start(self, keys: list[str], detector: PalmDetector, model: str, confidence: float,
              source: str = "live", refresh: bool = False, imagery_date: str = "", capture_id: str = "") -> str:
        with self._lock:
            if any(j["status"] == "running" for j in self.jobs.values()):
                raise RuntimeError("An analysis is already running. Wait for it to finish (or cancel it).")
            job_id = f"{datetime.now():%H%M%S}-{uuid.uuid4().hex[:6]}"
            job = {"id": job_id, "status": "running", "done": 0, "total": len(keys), "message": "Starting…",
                   "results": [], "skipped": [], "error": None, "phase": "start", "source": source}
            self.jobs[job_id] = job
            self._cancel[job_id] = threading.Event()
        cancel = self._cancel[job_id]

        def run():
            try:
                for i, key in enumerate(sorted(keys)):
                    if cancel.is_set():
                        break
                    job.update(done=i, message=f"{key} ({i + 1}/{len(keys)})", phase="count")

                    def progress(done, total, message, _key=key):
                        job.update(message=f"{_key}: {message}", phase="capture" if "Google Earth" in message else "count")
                    try:
                        r = self.service.analyze(key, detector, model, confidence, source, refresh, progress, cancel.is_set, imagery_date, capture_id)
                        job["results"].append({k: r[k] for k in ("code", "count", "area_ha", "palms_per_ha", "coverage_pct", "kind")})
                    except ValueError as exc:
                        job["skipped"].append({"code": key, "reason": str(exc)})
                        if source == "live" and ("not linked" in str(exc) or "not running" in str(exc)):
                            break  # no point trying the remaining polygons
                job.update(status="cancelled" if cancel.is_set() else "done", done=len(keys), message="Cancelled" if cancel.is_set() else "Done", phase="done")
            except Exception:  # noqa: BLE001 - details go to the log, not the user
                log.exception("Analysis job failed")
                job.update(status="error", error="The analysis failed. Check the server log for details.")

        threading.Thread(target=run, daemon=True).start()
        return job_id

    def start_task(self, run, total: int = 0) -> str:
        """Run `run(job, cancelled)` in the background (one job at a time); used for region captures."""
        with self._lock:
            if any(j["status"] == "running" for j in self.jobs.values()):
                raise RuntimeError("An analysis is already running. Wait for it to finish (or cancel it).")
            job_id = f"{datetime.now():%H%M%S}-{uuid.uuid4().hex[:6]}"
            job = {"id": job_id, "status": "running", "done": 0, "total": total, "message": "Starting…", "results": [],
                   "skipped": [], "error": None, "phase": "start", "source": "region", "summary": None}
            self.jobs[job_id] = job
            self._cancel[job_id] = threading.Event()
        cancel = self._cancel[job_id]

        def worker():
            try:
                summary = run(job, cancel.is_set)
                job["summary"] = summary
                cancelled = cancel.is_set() or bool(summary and summary.get("cancelled"))
                job.update(status="cancelled" if cancelled else "done", done=job["total"], phase="done",
                           message="Cancelled" if cancelled else "Done")
            except ValueError as exc:
                job.update(status="error", error=str(exc))
            except Exception:  # noqa: BLE001
                log.exception("Region job failed")
                job.update(status="error", error="The region run failed. Check the server log for details.")

        threading.Thread(target=worker, daemon=True).start()
        return job_id

    def cancel(self, job_id: str) -> bool:
        event = self._cancel.get(job_id)
        if event is None:
            return False
        event.set()
        return True
