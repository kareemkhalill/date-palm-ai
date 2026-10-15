"""Google Earth Pro integration: KML that Earth loads from this app, and helpers to drive Earth.

Earth cannot hand its imagery to a program, but it can *pull KML from a URL*. A NetworkLink makes Earth
ask this app for farms every time the user stops moving the view, so farms, counts and result images
show up inside Earth, and balloon links call back into the app.
"""
import math
import os
import re
from xml.sax.saxutils import escape

from shapely.geometry import box, mapping

from . import config, geo
from .farms import FarmService

KML_MIME = "application/vnd.google-earth.kml+xml"
MAX_FARMS_IN_VIEW = 700
MAX_VIEW_SPAN_DEG = 0.25     # wider than this we only draw region outlines
EARTH_DIR = config.DATA_DIR / "earth"

# KML colours are aabbggrr
STYLES = {
    "missing": ("66a39d9a", "1a8c968f", 1.0),
    "imaged": ("ffb8a719", "33b8a719", 2.0),
    "d1": ("ff4a9a2f", "66a8d98e", 2.0),
    "d2": ("ff3f8a25", "73a0d07d", 2.0),
    "d3": ("ff347a1d", "80509d46", 2.0),
    "d4": ("ff2a6a17", "8c4a8a2a", 2.0),
    "d5": ("ff205a10", "99307a1e", 2.0),
}


def _style_xml() -> str:
    out = []
    for sid, (line, fill, width) in STYLES.items():
        out.append(f'<Style id="{sid}"><LineStyle><color>{line}</color><width>{width}</width></LineStyle>'
                   f'<PolyStyle><color>{fill}</color><fill>1</fill><outline>1</outline></PolyStyle>'
                   f'<BalloonStyle><text>$[description]</text></BalloonStyle></Style>')
    out.append('<Style id="palm"><IconStyle><scale>0.45</scale><color>ff2828eb</color><Icon><href>'
               'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon></IconStyle>'
               '<LabelStyle><scale>0</scale></LabelStyle></Style>')
    out.append('<Style id="region"><LineStyle><color>ccffffff</color><width>2</width></LineStyle>'
               '<PolyStyle><fill>0</fill></PolyStyle></Style>')
    return "".join(out)


def _ring(coords) -> str:
    return " ".join(f"{x:.7f},{y:.7f},0" for x, y, *_ in coords)


def _polygon_xml(geometry) -> str:
    polys = [p for p in getattr(geometry, "geoms", [geometry]) if p.geom_type == "Polygon"]
    parts = []
    for p in polys:
        inner = "".join(f"<innerBoundaryIs><LinearRing><coordinates>{_ring(h.coords)}</coordinates></LinearRing></innerBoundaryIs>"
                        for h in p.interiors)
        parts.append(f"<Polygon><tessellate>1</tessellate><altitudeMode>clampToGround</altitudeMode><outerBoundaryIs><LinearRing>"
                     f"<coordinates>{_ring(p.exterior.coords)}</coordinates></LinearRing></outerBoundaryIs>{inner}</Polygon>")
    return parts[0] if len(parts) == 1 else f"<MultiGeometry>{''.join(parts)}</MultiGeometry>"


def _doc(name: str, body: str, extra: str = "") -> str:
    return (f'<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
            f"<name>{escape(name)}</name>{_style_xml()}{extra}{body}</Document></kml>")


def entry_kml(base_url: str) -> str:
    """The small KML the user opens in Earth once.

    It holds two links: the farms of the current view, and the "camera" the app uses to make
    Earth fly to a place (with flyToView) when it needs to capture imagery.
    """
    base = base_url.rstrip("/")
    return ('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
            "<name>Date Palm AI</name><open>1</open>"
            "<NetworkLink><name>Farms</name><Link>"
            f"<href>{base}/earth/farms.kml</href><refreshMode>onInterval</refreshMode><refreshInterval>3</refreshInterval>"
            "<viewRefreshMode>onStop</viewRefreshMode><viewRefreshTime>1</viewRefreshTime>"
            "<viewFormat>BBOX=[bboxWest],[bboxSouth],[bboxEast],[bboxNorth]</viewFormat></Link></NetworkLink>"
            "<NetworkLink><name>Capture camera</name><flyToView>1</flyToView><Link>"
            f"<href>{base}/earth/camera.kml</href><refreshMode>onInterval</refreshMode><refreshInterval>2</refreshInterval>"
            "</Link></NetworkLink></Document></kml>")


# --- capture camera ---------------------------------------------------------------------------
# Marker squares in four corners of the view; their pixel positions in the saved image give the
# exact georeference of every capture (independent of Earth's window size).
MARKER_HALF_M = 3.5
MARKER_REACH = 0.78            # fraction of the expected half extent where the markers sit
MARKER_STYLE = ('<Style id="mk"><PolyStyle><color>ffff00ff</color><fill>1</fill><outline>0</outline></PolyStyle></Style>')


class Camera:
    """The place Earth should currently look at (served as /earth/camera.kml)."""

    def __init__(self) -> None:
        self.target: tuple[float, float, float] | None = None   # lon, lat, range_m
        self.date: str = ""                                      # YYYY-MM-DD, or "" for Earth's newest imagery
        self.polled_at = 0.0

    def set(self, lon: float, lat: float, range_m: float, date: str = "") -> None:
        self.target = (lon, lat, range_m)
        self.date = date

    def clear(self) -> None:
        self.target = None

    def marker_positions(self) -> list[tuple[str, float, float]]:
        from .capture import half_extent
        lon, lat, rng = self.target
        half_lon, half_lat = half_extent(lat, rng)
        return [(name, lon + sx * MARKER_REACH * half_lon, lat + sy * MARKER_REACH * half_lat)
                for name, sx, sy in (("NW", -1, 1), ("NE", 1, 1), ("SW", -1, -1), ("SE", 1, -1))]

    def kml(self) -> str:
        import time
        self.polled_at = time.time()
        if self.target is None:
            return '<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>idle</name></Document></kml>'
        lon, lat, rng = self.target
        per_lon, per_lat = geo.metres_per_degree(lat)
        d_lon, d_lat = MARKER_HALF_M / per_lon, MARKER_HALF_M / per_lat
        marks = "".join(
            f"<Placemark><name>{n}</name><styleUrl>#mk</styleUrl><Polygon><altitudeMode>clampToGround</altitudeMode><outerBoundaryIs><LinearRing><coordinates>"
            f"{x - d_lon:.7f},{y - d_lat:.7f},0 {x + d_lon:.7f},{y - d_lat:.7f},0 {x + d_lon:.7f},{y + d_lat:.7f},0 "
            f"{x - d_lon:.7f},{y + d_lat:.7f},0 {x - d_lon:.7f},{y - d_lat:.7f},0</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>"
            for n, x, y in self.marker_positions())
        when = ""   # Earth ignores a date in the view, so imagery date stays whatever Earth's Historical Imagery slider shows
        look = (f"<LookAt><longitude>{lon:.7f}</longitude><latitude>{lat:.7f}</latitude><altitude>0</altitude><heading>0</heading>"
                f"<tilt>0</tilt><range>{rng:.0f}</range>{when}<altitudeMode>relativeToGround</altitudeMode></LookAt>")
        return (f'<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2"><Document><name>capture</name>'
                f"{MARKER_STYLE}{look}{marks}</Document></kml>")


camera = Camera()


def _style_for(farm: geo.Farm, palms: int | None, coverage: float) -> str:
    if palms is not None:
        density = palms / farm.area_ha if farm.area_ha > 0.01 else 0
        return f"d{min(5, 1 + int(density // 35))}"
    return "imaged" if coverage >= 99.5 else "missing"


def _balloon(base: str, farm: geo.Farm, saved: dict | None, coverage: float) -> str:
    status = escape(farm.status or farm.type or "")
    rows = (f"<tr><td>Region</td><td>{escape(farm.region)}</td></tr><tr><td>Status</td><td>{status}</td></tr>"
            f"<tr><td>Area</td><td>{farm.area_ha:.2f} ha</td></tr><tr><td>Imagery stored</td><td>{coverage:.0f}%</td></tr>")
    if saved:
        head = f'<p style="font-size:30px;margin:4px 0"><b>{saved["count"]:,}</b> palms</p>'
        img = f'<p><img src="{base}{saved["image"]}" width="420"></p>'
        links = (f'<a href="{base}/earth/count/{farm.code}">Count again</a> · '
                 f'<a href="{base}/earth/farm/{farm.code}.kml">Show palm points in Earth</a> · '
                 f'<a href="{base}/farms?code={farm.code}">Open in app</a>')
        rows += f"<tr><td>Counted</td><td>{escape(saved['analyzed'])} · {escape(str(saved['model']).replace('.pt', ''))}</td></tr>"
    else:
        head = '<p style="color:#666">Not counted yet.</p>'
        img = ""
        links = f'<a href="{base}/earth/count/{farm.code}"><b>Count palms in this farm</b></a> · <a href="{base}/farms?code={farm.code}">Open in app</a>'
    return (f'<div style="font-family:Segoe UI,Arial;width:430px"><h3 style="margin:0 0 4px">{farm.code}</h3>{head}'
            f'<table style="font-size:13px;border-spacing:10px 2px">{rows}</table>{img}<p>{links}</p></div>')


def farms_in_view(service: FarmService, base_url: str, bbox: tuple[float, float, float, float] | None) -> str:
    """KML for the farms inside the user's current Earth view (called by the NetworkLink)."""
    base = base_url.rstrip("/")
    if camera.target is not None:
        # A capture is in progress: show Earth's plain imagery, otherwise the polygons get saved into the image.
        return _doc("Date Palm AI (capturing)", "")
    boundaries = service.boundaries
    if bbox is None:
        view = None
    else:
        west, south, east, north = bbox
        view = box(west, south, east, north)
    too_wide = bbox is None or (bbox[2] - bbox[0]) > MAX_VIEW_SPAN_DEG
    if too_wide:
        regions = "".join(
            f'<Placemark><name>{escape(name)}</name><styleUrl>#region</styleUrl>{_polygon_xml(g.simplify(0.0002))}</Placemark>'
            for name, g in boundaries.regions.items() if view is None or g.intersects(view))
        note = '<Placemark><name>Zoom in to see farms</name></Placemark>' if view is not None else ""
        return _doc("Date Palm AI", regions + note)

    coverage, done = service.coverage(), service.summary()
    marks, n = [], 0
    for farm in boundaries.farms.values():
        if not farm.geometry.intersects(view):
            continue
        n += 1
        if n > MAX_FARMS_IN_VIEW:
            break
        saved = done.get(farm.code)
        label = f"{farm.code} · {saved['count']:,} palms" if saved else farm.code
        marks.append(f"<Placemark><name>{escape(label)}</name><styleUrl>#{_style_for(farm, saved['count'] if saved else None, coverage.get(farm.code, 0))}</styleUrl>"
                     f"<description><![CDATA[{_balloon(base, farm, saved, coverage.get(farm.code, 0))}]]></description>"
                     f"{_polygon_xml(farm.geometry.simplify(0.000003))}</Placemark>")
    return _doc("Date Palm AI", "".join(marks))


def farm_kml(service: FarmService, code: str) -> str:
    """One farm's polygon plus every detected palm as a point (for checking over Earth's imagery)."""
    farm = service.farm(code)
    saved = service.summary().get(farm.code)
    if not saved:
        raise ValueError(f"{farm.code} has not been counted yet.")
    points = "".join(f"<Placemark><styleUrl>#palm</styleUrl><Point><coordinates>{x:.7f},{y:.7f},0</coordinates></Point></Placemark>"
                     for x, y in service.saved_points(farm.code))
    minx, miny, maxx, maxy = farm.geometry.bounds
    look = _look_at((minx + maxx) / 2, (miny + maxy) / 2, farm)
    body = (f"<Placemark><name>{farm.code} · {saved['count']:,} palms</name><styleUrl>#d3</styleUrl>{_polygon_xml(farm.geometry)}</Placemark>"
            f"<Folder><name>Palms ({saved['count']:,})</name>{points}</Folder>")
    return _doc(f"{farm.code} – {saved['count']:,} palms", body, look)


def _look_at(lon: float, lat: float, farm: geo.Farm) -> str:
    west, south, east, north = farm.geometry.bounds
    per_lon, per_lat = geo.metres_per_degree(lat)
    extent = max((east - west) * per_lon, (north - south) * per_lat)
    rng = int(min(2500, max(180, extent * 1.7)))
    return (f"<LookAt><longitude>{lon:.7f}</longitude><latitude>{lat:.7f}</latitude><altitude>0</altitude><heading>0</heading>"
            f"<tilt>0</tilt><range>{rng}</range><altitudeMode>relativeToGround</altitudeMode></LookAt>")


def fly_kml(service: FarmService, code: str) -> str:
    """A tiny KML that makes Earth fly (top-down) to a farm and outline it."""
    farm = service.farm(code)
    c = farm.geometry.centroid
    saved = service.summary().get(farm.code)
    name = f"{farm.code} · {saved['count']:,} palms" if saved else farm.code
    return _doc(name, f"<Placemark><name>{escape(name)}</name><styleUrl>#d3</styleUrl>{_polygon_xml(farm.geometry)}</Placemark>",
                _look_at(c.x, c.y, farm))


def open_in_earth(kml: str, filename: str) -> str:
    """Write the KML and open it with the default handler (Google Earth Pro). Windows only."""
    EARTH_DIR.mkdir(parents=True, exist_ok=True)
    path = EARTH_DIR / re.sub(r"[^A-Za-z0-9_.-]", "_", filename)
    path.write_text(kml, encoding="utf-8")
    if not hasattr(os, "startfile"):
        raise OSError("Opening Google Earth from the app is only supported on Windows.")
    os.startfile(str(path))
    return str(path)
