# Date Palm AI

Custom-trained **YOLO26** detector that counts date palms in aerial and satellite imagery, per image or per farm polygon on a map.

**Live demo:** https://date-palm-ai.vercel.app · https://kareemkhalill.github.io/date-palm-ai

> Model weights (`*.pt`), farm boundaries (KML) and imagery are not included in this repository. Put your own trained weights in the project folder (see *Models* below).

**Full documentation (HTML): [docs/index.html](docs/index.html)** — overview, how to use it, how it works, configuration, API and troubleshooting.

Count date palms with your trained YOLO26 models, two ways:

- **Farm map** (`/farms`): all farm polygons from your KML on a map. Click farms (or type a farm code), press *Count palms*, and get the count plus a labelled image of each farm. Works offline of Google Earth once its imagery is stored (see below).
- **Upload image** (`/`): drop any aerial JPG/PNG and get the count + annotated image.

## Run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # or omit the index for CPU
pip install -r requirements.txt
python app.py        # http://127.0.0.1:8000  (or double-click start.bat)
```
Always use http://127.0.0.1:8000 (opening the HTML files directly cannot reach the AI backend). The model needs up to a minute to load.

## Farm map: live counting from Google Earth Pro (recommended)
Open `/farms`, make sure **Google Earth is linked** (button *Link Google Earth*; Earth Pro must be running), then either
- **select farms** (click polygons, search a code, or *Select all farms* for a region), or
- **draw an area** (*Draw an area*, click the corners) and save it,

choose one of the two **Imagery** options and press *Count palms*:
- **Capture new from Google Earth (and save it)** — takes a new capture and saves it under the polygon's code in `data/saved/<code>/` (a farm-sized JPEG, 1–10 MB, with its coordinates and Earth's imagery-date strip).
- **Use a saved capture (no Earth needed)** — counts from a capture saved earlier (pick which one when a single polygon is selected, or the newest for each polygon in a multi-selection), in a few seconds. You can delete saved captures from the same picker.

The first option works like this: The app flies Earth to the polygon (top-down, 600 m camera range, so a palm is ~90 px wide), waits for the imagery to load, saves Earth's 8192×4320 image, georeferences it from four corner markers it places in the view, stitches several captures for big polygons, counts the palms whose centre lies inside the polygon, and shows the labelled image and count on the map. Imagery is whatever Earth currently shows, so a count a year from now uses the new imagery; each result records the capture time and shows Earth's status bar (the imagery date). Captures are reused for 24 h (`PALM_LIVE_MAX_AGE_H`) unless you tick *Capture fresh imagery*.

Notes: about 100 s per Earth view (a 1 ha farm = 1–2 views; the app tells you the number and asks before it starts; limit 40 views per job). **Don't touch the mouse or keyboard while it captures**, and keep Earth's *Save Image* resolution at 8192×4320 and its window visible. The imagery date is whatever Earth's own **Historical Imagery slider** shows (the app cannot change it; Earth resets it to the newest after a restart/crash), so set the date you want in Earth before counting — in Tayma the newest (10/2025) imagery is dark and hazy and finds far fewer palms than 3/29/2023. The app measures how bright the imagery is inside the polygon: dark imagery is brightened before detection and the result shows a warning ("dark" / "very dark" — the count is then an underestimate; on the same Tayma farm the newest dark imagery gave 4 palms and a clear date gave 375). A new-capture count always takes a new capture (a recent Earth capture is never silently reused); to reuse one on purpose, pick *Use a saved capture*. Every run is kept under `results/farms/runs/<code>/<time>/` (`capture.jpg`, `detection.jpg`, `result.json`). Tested on Tayma. Results are saved in `results/farms/`; *Export CSV* lists counts per farm (and `?kind=area` per custom area).

## Farm map: stored captures (no Earth needed)
```
your KML ──import──> data/farms.json            all 16 regions + 7,124 farms
Earth Pro ─capture─> data/captures/*.jpg        stored once per area (palms ~90 px wide)
click farm ─────────> crop farm from captures -> YOLO -> palms inside polygon -> count + image
```
1. **Import the KML** (once, or when it changes): `python tools/import_kml.py "boundaries.kml"` (add `--prefix TYM` to keep one code prefix).
2. **Plan the imagery you need**: `python tools/plan_captures.py --list` shows captures per region; `python tools/plan_captures.py --region Tayma --imagery-date 3/29/2023` writes `data/grids/tayma_N.kml` (camera positions, no farm shapes) and `data/captures/manifest.json`.
3. **Capture once in Google Earth Pro**: open a `grids/*.kml`, double-click each position, *File → Save → Save Image* at **8192×4320** (top-down, north up), saved as `data/captures/<position name>.jpg`. Best imagery found for Tayma: *View → Historical Imagery* → **3/29/2023** (clear, green; the 10/2025 imagery is dark and hazy). Range **600 m** gives palms ~90 px wide, which tested best.
4. **Use the map**: farms turn teal when their imagery is stored, green when counted (darker = denser). Select farms → *Count palms in selection*; outputs go to `results/farms/` (`images/<code>.jpg` labelled, `raw/<code>.jpg` clean, `summary.json`). *Export CSV* gives per-farm counts and the total. Farms without imagery tell you which captures are needed.

Per region, a full grid takes about 90 s per capture in Earth (`--list` shows the numbers: Tayma is 62 captures; the other regions are sparse and need hundreds). You can also capture only the area around the farms you care about.

## Models
Every `*.pt` in the project folder appears in the model dropdown (`PALM_MODELS_DIR`, default `PALM_DEFAULT_MODEL` in `palmai/config.py`). Default: `yolov26l-mixed-data-70epoch.pt`. Multi-class models are filtered so only palm classes are counted; models with two palm classes are de-duplicated.

## How inference works
- **Automatic scale.** Counts depend strongly on how large palms look to the model. The app measures median palm width and tiles the image so palms appear ~24 px at the model's 640 px input (`PALM_TARGET_PX`). Single-pass and fixed-tile modes are in *Advanced Settings* on the upload page.
- Count = palm detections above the confidence threshold (default 0.15). `max_det` is raised to 20,000. On the farm map only palms whose centre lies inside the farm polygon count.
- If another program (e.g. Google Earth) takes the GPU memory, inference falls back to the CPU automatically (slower).

## Structure
```
app.py                 FastAPI app (/, /farms, /predict, /api/farms/*)
palmai/config.py       all settings, env-overridable
palmai/detector.py     models, automatic scale, tiling, GPU->CPU fallback
palmai/farms.py        farm/area service: crops, counting, images, jobs, CSV
palmai/imagery.py      imagery sources (live Earth captures / stored captures)
palmai/earth_capture.py drives Google Earth Pro: fly, wait, Save Image, georeference (Windows)
palmai/earth.py, earth_api.py  Earth link: KML network link, camera, balloons
palmai/farms_api.py    /api/farms endpoints
palmai/geo.py          KML parsing, georeferencing
palmai/capture.py      Earth Pro capture planning and calibration
tools/                 import_kml.py, plan_captures.py
static/                upload page (index.html) and farm map (farms.html, farms.js, farms.css)
data/                  farms.json, captures/, grids/ (generated; not in source control)
results/               upload results and results/farms/
```

## Limitations
- Imagery is a fixed snapshot (e.g. March 2023); palms planted later are not visible, and counts are model estimates. Validate on a farm you have counted by hand before trusting totals; counts move about ±15% with the confidence slider.
- Young palms in sparse plots and palms in deep shadow are the most commonly missed.
- The farm map's background tiles (OpenStreetMap / Esri) are for orientation only and need internet.
- Check Google's terms for your use of Earth imagery; Earth Pro desktop is scheduled to be discontinued (June 2027), but stored captures keep working.
