"""Plan top-down Google Earth Pro captures that tile a region, and know what ground each covers.

Calibration (measured with marker polygons at known coordinates, Earth Pro 8192x4320 Save Image,
top-down, north-up): at LookAt range 1500 m near Tayma (27.645 N) one image spans
0.017536 deg lon x 0.008211 deg lat, centred on the LookAt point (~0.211 m/pixel).
Ground size scales linearly with range.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

from shapely.geometry import box

from .geo import GeoImage

IMAGE_W, IMAGE_H = 8192, 4320
REF_RANGE_M = 1500.0
REF_LAT = 27.645
REF_HALF_LON = 0.017536 / 2   # degrees at REF_RANGE_M and REF_LAT
REF_HALF_LAT = 0.008211 / 2


@dataclass
class Cell:
    name: str
    lon: float      # LookAt centre
    lat: float
    range_m: float

    @property
    def geo(self) -> GeoImage:
        half_lon, half_lat = half_extent(self.lat, self.range_m)
        return GeoImage(self.lon - half_lon, self.lat - half_lat, self.lon + half_lon, self.lat + half_lat,
                        IMAGE_W, IMAGE_H)


def half_extent(lat: float, range_m: float) -> tuple[float, float]:
    scale = range_m / REF_RANGE_M
    return (REF_HALF_LON * scale * math.cos(math.radians(REF_LAT)) / math.cos(math.radians(lat)),
            REF_HALF_LAT * scale)


def plan_grid(region, range_m: float = 1500, overlap: float = 0.10, name_prefix: str = "c") -> list[Cell]:
    """Cells (row-major, north to south) covering `region`; cells missing the region are dropped."""
    min_lon, min_lat, max_lon, max_lat = region.bounds
    mid_lat = (min_lat + max_lat) / 2
    half_lon, half_lat = half_extent(mid_lat, range_m)
    step_lon, step_lat = 2 * half_lon * (1 - overlap), 2 * half_lat * (1 - overlap)
    cells = []
    row, lat = 0, max_lat - half_lat * (1 - overlap)
    while True:
        col, lon = 0, min_lon + half_lon * (1 - overlap)
        while True:
            cell = Cell(f"{name_prefix}_r{row:02d}_c{col:02d}", lon, lat, range_m)
            g = cell.geo
            if region.intersects(box(g.west, g.south, g.east, g.north)):
                cells.append(cell)
            if lon - half_lon * (1 - overlap) >= max_lon - 1e-9 or lon >= max_lon:
                break
            lon += step_lon
            col += 1
        if lat - half_lat * (1 - overlap) <= min_lat or lat <= min_lat:
            break
        lat -= step_lat
        row += 1
    return cells


def write_grid_kml(cells: list[Cell], path: Path, title: str) -> None:
    # Invisible icon/label so the capture points never show up in saved images.
    style = ('<Style id="hidden"><IconStyle><scale>0</scale></IconStyle>'
             '<LabelStyle><scale>0</scale></LabelStyle></Style>')
    marks = "\n".join(
        f'<Placemark><name>{c.name}</name><styleUrl>#hidden</styleUrl><LookAt><longitude>{c.lon:.7f}</longitude><latitude>{c.lat:.7f}</latitude>'
        f"<altitude>0</altitude><heading>0</heading><tilt>0</tilt><range>{c.range_m:.0f}</range>"
        f"<altitudeMode>relativeToGround</altitudeMode></LookAt>"
        f"<Point><coordinates>{c.lon:.7f},{c.lat:.7f},0</coordinates></Point></Placemark>"
        for c in cells)
    path.write_text(f'<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2">'
                    f"<Document><name>{title}</name>{style}\n{marks}\n</Document></kml>", encoding="utf-8")


def update_manifest(path: Path, cells: list[Cell], region: str, imagery_date: str = "") -> None:
    """Add/replace planned captures in manifest.json. A capture exists when <name>.jpg is next to it."""
    existing = {}
    if path.exists():
        existing = {c["name"]: c for c in json.loads(path.read_text(encoding="utf-8"))["captures"]}
    for c in cells:
        existing[c.name] = {
            "name": c.name, "file": f"{c.name}.jpg", "region": region, "range_m": c.range_m,
            "imagery_date": imagery_date, "center": [c.lon, c.lat],
            "bounds": [c.geo.west, c.geo.south, c.geo.east, c.geo.north], "width": IMAGE_W, "height": IMAGE_H}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"captures": sorted(existing.values(), key=lambda c: c["name"])}, indent=1), encoding="utf-8")
