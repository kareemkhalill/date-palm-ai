"""Imagery sources: where the pixels for a polygon come from.

  stored  captures saved earlier under data/captures (instant; the date is whenever they were taken)
  live    fresh captures from Google Earth Pro on this PC, reused for a short time (config.LIVE_MAX_AGE_HOURS)

Both are plain lists of georeferenced `Capture`s, so the counting code does not care which one it got.
"""
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from shapely.geometry import box
from shapely.ops import unary_union

from . import capture as plan, config, geo

SECONDS_PER_CAPTURE = 100
EDGE_TRIM = 0.06     # a live capture only counts as covering the part away from its border (marker zone)


@dataclass
class Capture:
    name: str
    path: Path
    bounds: tuple[float, float, float, float]  # west, south, east, north
    width: int
    height: int
    region: str = ""
    imagery_date: str = ""
    captured_at: str = ""
    status_image: str = ""
    kind: str = "stored"

    @property
    def available(self) -> bool:
        return self.path.exists()

    @property
    def geo(self) -> geo.GeoImage:
        return geo.GeoImage(*self.bounds, self.width, self.height)

    @property
    def box(self):
        return box(*self.bounds)

    def trimmed_box(self):
        w, s, e, n = self.bounds
        mx, my = (e - w) * EDGE_TRIM, (n - s) * EDGE_TRIM
        return box(w + mx, s + my, e - mx, n - my)


def _read(manifest: Path, kind: str) -> list[Capture]:
    if not manifest.exists():
        return []
    return [Capture(e["name"], manifest.parent / e["file"], tuple(e["bounds"]), e["width"], e["height"], e.get("region", ""),
                    e.get("imagery_date", ""), e.get("captured_at", ""), e.get("status_image", ""), kind)
            for e in json.loads(manifest.read_text(encoding="utf-8"))["captures"]]


def stored_captures() -> list[Capture]:
    return _read(config.DATA_DIR / "captures" / "manifest.json", "stored")


def live_captures() -> list[Capture]:
    return [c for c in _read(config.DATA_DIR / "live" / "manifest.json", "live") if c.available]


def fresh_live_captures(max_age_hours: float | None = None) -> list[Capture]:
    limit = datetime.now() - timedelta(hours=config.LIVE_MAX_AGE_HOURS if max_age_hours is None else max_age_hours)
    return [c for c in live_captures() if c.captured_at and datetime.fromisoformat(c.captured_at) >= limit]


def discard_live_captures() -> None:
    """Delete earlier live captures (images, status strips, manifest) so every count starts from a new capture."""
    live = config.DATA_DIR / "live"
    if not live.exists():
        return
    for f in live.iterdir():
        if f.is_file():
            f.unlink(missing_ok=True)
    preview = config.DATA_DIR / "previews"
    for f in preview.glob("live_*.jpg") if preview.exists() else []:
        f.unlink(missing_ok=True)


def _fits_one_view(part, lon: float, lat: float, half_lon: float, half_lat: float) -> bool:
    """One capture centred on `part` suffices if it is inside the view and clear of the four corner markers."""
    from . import earth
    west, south, east, north = part.bounds
    if (east - west) > 2 * half_lon * 0.92 or (north - south) > 2 * half_lat * 0.92:
        return False
    per_lon, per_lat = geo.metres_per_degree(lat)
    pad_lon, pad_lat = (earth.MARKER_HALF_M + 4) / per_lon, (earth.MARKER_HALF_M + 4) / per_lat
    for sx in (-1, 1):
        for sy in (-1, 1):
            mx, my = lon + sx * earth.MARKER_REACH * half_lon, lat + sy * earth.MARKER_REACH * half_lat
            if part.intersects(box(mx - pad_lon, my - pad_lat, mx + pad_lon, my + pad_lat)):
                return False
    return True


def plan_live(view, refresh: bool) -> list[plan.Cell]:
    """Captures still needed so that `view` (a lon/lat geometry) is covered by fresh live imagery."""
    target = view.buffer(config.LIVE_MARGIN_DEG)   # live counts always capture anew; old captures are never reused
    if target.is_empty or target.area < 1e-10:
        return []
    stamp = f"live_{datetime.now():%Y%m%d%H%M%S}"
    cells: list[plan.Cell] = []
    for part in getattr(target, "geoms", [target]):
        if part.is_empty or part.area < 1e-10:
            continue
        west, south, east, north = part.bounds
        lat = (south + north) / 2
        half_lon, half_lat = plan.half_extent(lat, config.LIVE_RANGE_M)
        if _fits_one_view(part, (west + east) / 2, lat, half_lon, half_lat):
            cells.append(plan.Cell(f"{stamp}_s{len(cells):02d}", (west + east) / 2, lat, config.LIVE_RANGE_M))   # one capture is enough
        else:
            cells += [plan.Cell(f"{stamp}_g{len(cells) + i:02d}", c.lon, c.lat, c.range_m)
                      for i, c in enumerate(plan.plan_grid(part, config.LIVE_RANGE_M, config.LIVE_OVERLAP))]
    return cells


def estimate(view, source: str, refresh: bool) -> dict:
    if source != "live":
        return {"source": source, "captures_needed": 0, "minutes": 0}
    cells = plan_live(view, refresh)
    return {"source": "live", "captures_needed": len(cells), "minutes": round(len(cells) * SECONDS_PER_CAPTURE / 60, 1),
            "too_many": len(cells) > config.LIVE_MAX_CAPTURES, "limit": config.LIVE_MAX_CAPTURES}


def ensure_live(view, refresh: bool, progress=lambda done, total, msg: None, cancelled=lambda: False,
                imagery_date: str = "") -> list[Capture]:
    """Capture whatever is missing from Google Earth, and return the fresh live captures covering `view`."""
    from . import earth_capture
    cells = plan_live(view, refresh)
    cancelled = cancelled or (lambda: False)
    discard_live_captures()   # only the captures of the current run are kept
    if cells:
        progress(0, len(cells), "Preparing Google Earth: shading off, choosing a well-lit imagery date…")
        try:
            # "newest" forces Earth's very latest imagery (even if dark); anything else picks the newest well-lit date
            mode = None if imagery_date == "keep" else (bool(config.AUTO_IMAGERY_DATE) and imagery_date != "newest")
            earth_capture.prepare_earth(cells[0].lon, cells[0].lat, cells[0].range_m, mode, cancelled)
        except earth_capture.CaptureError as exc:
            raise ValueError(str(exc)) from exc
    if len(cells) > config.LIVE_MAX_CAPTURES:
        raise ValueError(f"This area needs {len(cells)} Google Earth captures (limit {config.LIVE_MAX_CAPTURES}). Select a smaller area.")
    for i, cell in enumerate(cells):
        progress(i, len(cells), f"Capturing from Google Earth {i + 1}/{len(cells)} — please don't use the mouse or keyboard")
        if cancelled():
            raise ValueError("Cancelled.")
        try:
            earth_capture.capture_view(cell.lon, cell.lat, cell.range_m, cell.name, cancelled, imagery_date)
        except earth_capture.CaptureError as exc:
            raise ValueError(str(exc)) from exc
    progress(len(cells), max(len(cells), 1), "Imagery ready")
    view_box = box(*view.bounds)
    return [c for c in fresh_live_captures() if c.box.intersects(view_box)]
