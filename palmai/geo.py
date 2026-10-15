"""Geographic helpers: farm boundaries from a KML, and image georeferencing.

Coordinates are WGS84 (lon, lat). Captures are north-up, top-down images georeferenced by a bounding
box (west, south, east, north); over a few hundred metres the lon/lat -> pixel mapping is linear
enough that the pixel error stays far below one palm.
"""
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely import contains_xy, make_valid, union_all
from shapely.affinity import scale as affine_scale
from shapely.geometry import MultiPolygon, Polygon, mapping, shape

KML_NS = {"k": "http://www.opengis.net/kml/2.2"}
EARTH_RADIUS_M = 6_371_008.8


def _ring(el) -> list[tuple[float, float]]:
    return [tuple(map(float, c.split(",")[:2])) for c in el.find(".//k:coordinates", KML_NS).text.split()]


def _polygons(placemark):
    parts = []
    for p in placemark.findall(".//k:Polygon", KML_NS):
        outer = _ring(p.find("k:outerBoundaryIs", KML_NS))
        holes = [_ring(h) for h in p.findall("k:innerBoundaryIs", KML_NS)]
        parts.append(make_valid(Polygon(outer, holes)))
    return parts[0] if len(parts) == 1 else make_valid(MultiPolygon([g for g in parts if g.geom_type == "Polygon"]))


def _meta(placemark) -> dict[str, str]:
    return {d.get("name"): (d.find("k:value", KML_NS).text or "") for d in placemark.findall(".//k:Data", KML_NS)}


def extract_layers(kml_path: Path, code_prefix: str = "") -> dict:
    """Read every layer of a boundaries KML: region polygons and farm polygons.

    Farms are the placemarks that have a FarmCode; `code_prefix` optionally keeps only codes starting with it.
    """
    root = ET.parse(kml_path).getroot()
    regions, farms = [], []
    for folder in root.iter(f"{{{KML_NS['k']}}}Folder"):
        folder_name = folder.find("k:name", KML_NS).text
        for pm in folder.findall("k:Placemark", KML_NS):
            meta = _meta(pm)
            if meta.get("FarmCode"):
                if meta["FarmCode"].upper().startswith(code_prefix.upper()):
                    farms.append({"code": meta["FarmCode"], "type": meta.get("FarmType", ""),
                                  "status": meta.get("RegistrationStatus", ""), "region": meta.get("RegionName", ""),
                                  "geometry": mapping(_polygons(pm))})
            elif folder_name == "Regions":
                regions.append({"name": pm.find("k:name", KML_NS).text, "zone": meta.get("Zone", ""),
                                "geometry": mapping(_polygons(pm))})
    if not farms:
        raise ValueError(f"No farms found in {Path(kml_path).name}")
    return {"source_kml": Path(kml_path).name, "regions": regions, "farms": sorted(farms, key=lambda f: f["code"])}


@dataclass
class Farm:
    code: str
    type: str
    status: str
    region: str
    geometry: Polygon | MultiPolygon

    @property
    def area_ha(self) -> float:
        return area_m2(self.geometry) / 1e4


@dataclass
class Boundaries:
    farms: dict[str, Farm]
    regions: dict[str, Polygon | MultiPolygon]
    source: str = ""

    @classmethod
    def load(cls, path: Path) -> "Boundaries":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        farms = {f["code"]: Farm(f["code"], f["type"], f["status"], f.get("region", ""), make_valid(shape(f["geometry"])))
                 for f in data["farms"]}
        regions = {r["name"]: make_valid(shape(r["geometry"])) for r in data.get("regions", [])}
        return cls(farms, regions, data.get("source_kml", ""))

    def farms_in(self, region: str) -> list[Farm]:
        return [f for f in self.farms.values() if f.region == region]

    @staticmethod
    def union_of(farms) -> "Polygon | MultiPolygon":
        return union_all([f.geometry for f in farms])


def area_m2(geom) -> float:
    """Area of a lon/lat geometry using a local equirectangular projection."""
    lat0 = math.radians(geom.centroid.y)
    kx = math.pi / 180 * EARTH_RADIUS_M * math.cos(lat0)
    ky = math.pi / 180 * EARTH_RADIUS_M
    return affine_scale(geom, kx, ky, origin=(0, 0)).area


def metres_per_degree(lat: float) -> tuple[float, float]:
    """(metres per degree of longitude, metres per degree of latitude) at `lat`."""
    return math.pi / 180 * EARTH_RADIUS_M * math.cos(math.radians(lat)), math.pi / 180 * EARTH_RADIUS_M


@dataclass
class GeoImage:
    """A north-up image with known geographic bounds."""
    west: float
    south: float
    east: float
    north: float
    width: int
    height: int

    def pixel_to_lonlat(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        lon = self.west + (np.asarray(x) / self.width) * (self.east - self.west)
        lat = self.north - (np.asarray(y) / self.height) * (self.north - self.south)
        return lon, lat

    def lonlat_to_pixel(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        x = (np.asarray(lon) - self.west) / (self.east - self.west) * self.width
        y = (self.north - np.asarray(lat)) / (self.north - self.south) * self.height
        return x, y

    @property
    def metres_per_pixel(self) -> float:
        per_lon, per_lat = metres_per_degree((self.north + self.south) / 2)
        return float(((self.east - self.west) * per_lon / self.width + (self.north - self.south) * per_lat / self.height) / 2)


def points_in_geometry(geometry, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Boolean mask of the points that fall inside `geometry`."""
    return contains_xy(geometry, lon, lat)
