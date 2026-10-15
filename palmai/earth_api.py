"""HTTP endpoints Google Earth Pro talks to (KML network link, balloon links)."""
import html
import logging
from typing import Callable

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, Response

from . import config, earth
from .detector import PalmDetector
from .farms import FarmService

log = logging.getLogger("palmai.earth_api")


def _base(request: Request) -> str:
    return f"{request.url.scheme}://{request.url.netloc}"


def _local_only(request: Request) -> None:
    """Endpoints that start programs on this PC must only be callable from this PC."""
    if request.client is None or request.client.host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(403, "This action is only available on the computer running the app.")


def _kml(text: str, name: str = "palm-ai.kml") -> Response:
    return Response(text, media_type=earth.KML_MIME, headers={"Content-Disposition": f'inline; filename="{name}"'})


def build_router(service: FarmService, get_detector: Callable[[], PalmDetector | None]) -> APIRouter:
    router = APIRouter(prefix="/earth", tags=["google-earth"])

    @router.get("/palm-ai.kml")
    def entry(request: Request):
        """Open this in Google Earth Pro once: it then pulls farms from the app as you move around."""
        return _kml(earth.entry_kml(_base(request)), "Date Palm AI.kml")

    @router.get("/farms.kml")
    def farms(request: Request, BBOX: str | None = None):
        bbox = None
        if BBOX:
            try:
                west, south, east, north = (float(v) for v in BBOX.split(","))
                bbox = (west, south, east, north)
            except ValueError:
                raise HTTPException(400, "BBOX must be west,south,east,north")
        try:
            return _kml(earth.farms_in_view(service, _base(request), bbox), "farms.kml")
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @router.get("/camera.kml")
    def camera_kml():
        """Polled by Earth every couple of seconds; carries the view to fly to while a capture is pending."""
        return _kml(earth.camera.kml(), "camera.kml")

    @router.post("/camera")
    def set_camera(request: Request, lon: float | None = None, lat: float | None = None, range_m: float = 600):
        """Point Earth's view at a place (omit lon/lat to release it). Local only."""
        _local_only(request)
        if lon is None or lat is None:
            earth.camera.clear()
        else:
            earth.camera.set(lon, lat, range_m)
        return {"target": earth.camera.target}

    @router.post("/best-date")
    def best_date(request: Request, lon: float, lat: float, range_m: float = config.LIVE_RANGE_M):
        """Step through Earth's Historical Imagery at a place and leave Earth on the newest well-lit date. Local only."""
        _local_only(request)
        from . import earth_capture
        try:
            return earth_capture.choose_imagery_date(lon, lat, range_m, log=lambda m: log.info(m))
        except earth_capture.CaptureError as exc:
            raise HTTPException(409, str(exc))

    @router.get("/status")
    def status():
        """Is Earth polling the capture camera (i.e. is the live link loaded in Earth)?"""
        import time
        age = time.time() - earth.camera.polled_at if earth.camera.polled_at else None
        return {"linked": age is not None and age < 10, "seconds_since_poll": age}

    @router.get("/farm/{code}.kml")
    def farm(code: str):
        try:
            return _kml(earth.farm_kml(service, code), f"{code}.kml")
        except KeyError:
            raise HTTPException(404, f"Unknown farm: {code}")
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @router.get("/count/{code}", response_class=HTMLResponse)
    def count(code: str, request: Request):
        """Balloon link: count one farm now and show the result as a small page."""
        base = _base(request)
        detector = get_detector()
        try:
            farm = service.farm(code)
        except KeyError:
            return HTMLResponse(_page(f"Unknown farm {html.escape(code)}", "This code is not in the loaded KML."), 404)
        if detector is None:
            return HTMLResponse(_page(farm.code, "The AI model is still loading. Try again in a minute."), 503)
        try:
            job_id = service.jobs.start([farm.code], detector, detector.default_model, config.DEFAULT_CONFIDENCE, "live", False)
        except RuntimeError as exc:
            return HTMLResponse(_page(farm.code, html.escape(str(exc))), 409)
        # The page polls the job (a capture takes about a minute per Google Earth view) and shows the result.
        script = (
            "<p id=msg>Starting…</p><div id=out></div><script>"
            f"const base={base!r}, job={job_id!r}, code={farm.code!r};"
            "async function tick(){const j=await (await fetch(base+'/api/farms/jobs/'+job)).json();"
            "document.getElementById('msg').textContent=j.message||'';"
            "if(j.status==='running'){return setTimeout(tick,2000)}"
            "if(j.skipped&&j.skipped.length){document.getElementById('msg').textContent=j.skipped[0].reason;return}"
            "if(j.status!=='done'){document.getElementById('msg').textContent=j.error||'Stopped.';return}"
            "const r=await (await fetch(base+'/api/farms/'+code)).json();"
            "document.getElementById('msg').innerHTML='';"
            "document.getElementById('out').innerHTML='<p class=big>'+r.count.toLocaleString()+' palms</p>'"
            "+'<p>'+r.area_ha+' ha · '+(r.palms_per_ha||'—')+' palms/ha</p>'"
            "+'<p><img src=\"'+base+r.image+'\" style=\"max-width:100%\"></p>'"
            "+'<p><a href=\"'+base+'/earth/farm/'+code+'.kml\">Show palm points in Earth</a> · <a href=\"'+base+'/farms?code='+code+'\">Open in app</a></p>';}"
            "tick();</script>")
        return HTMLResponse(_page(farm.code, "<p>Capturing current imagery from Google Earth, then counting. "
                                             "Please don't use the mouse or keyboard until it finishes.</p>" + script))

    @router.post("/fly/{code}")
    def fly(code: str, request: Request):
        """Make Google Earth Pro fly to a farm (opens a KML with the default program)."""
        _local_only(request)
        try:
            earth.open_in_earth(earth.fly_kml(service, code), f"fly_{code}.kml")
        except KeyError:
            raise HTTPException(404, f"Unknown farm: {code}")
        except OSError as exc:
            raise HTTPException(500, str(exc))
        return {"opened": code}

    @router.post("/open-link")
    def open_link(request: Request):
        """Open the live farm link in Google Earth Pro."""
        _local_only(request)
        try:
            earth.open_in_earth(earth.entry_kml(_base(request)), "Date Palm AI.kml")
        except OSError as exc:
            raise HTTPException(500, str(exc))
        answered = 0
        try:  # Earth asks for confirmation each time; answer for the user
            from . import earth_capture
            answered = earth_capture.accept_earth_prompts()
        except Exception:  # noqa: BLE001 - the user can still click Open themselves
            log.warning("Could not answer Earth's prompt automatically", exc_info=True)
        return {"opened": True, "prompts_answered": answered}

    return router


def _page(title: str, body: str) -> str:
    return ("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            "<style>body{font:15px Segoe UI,Arial,sans-serif;margin:0;padding:18px;color:#1d2620}h2{margin:0 0 8px}"
            ".big{font-size:34px;font-weight:700;color:#2f5d3f;margin:4px 0}a{color:#2f5d3f}</style>"
            f"<h2>{html.escape(title)}</h2>{body}")
