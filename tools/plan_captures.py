"""Plan the Google Earth Pro captures that cover farms.

Usage:
  python tools/plan_captures.py --list                       # captures needed per region
  python tools/plan_captures.py --region Tayma [--region "Al Hijr"] [--imagery-date 3/29/2023]
  python tools/plan_captures.py --all

Writes under data/:
  grids/<region>_N.kml     camera positions to fly to in Earth Pro (invisible pins, no farm shapes)
  captures/manifest.json   every planned capture and the ground area it covers (merged across runs)
Capture each position with Earth's Save Image at 8192x4320 (top-down, north up), saving it as
captures/<name>.jpg; after that the app needs nothing else from Earth.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shapely.ops import unary_union  # noqa: E402

from palmai import capture, config, geo  # noqa: E402


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def plan_region(boundaries: geo.Boundaries, region: str, range_m: float, margin_deg: float):
    farms = boundaries.farms_in(region)
    area = unary_union([f.geometry.buffer(margin_deg) for f in farms])
    return capture.plan_grid(area, range_m, 0.10, name_prefix=slug(region))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--range", type=float, default=600,
                    help="Earth LookAt range in metres. 600 puts a mature palm at ~90 px (tested best).")
    ap.add_argument("--per-kml", type=int, default=24)
    ap.add_argument("--margin-deg", type=float, default=0.0002, help="margin around farms (~20 m)")
    ap.add_argument("--imagery-date", default="", help="Earth imagery date used, recorded in the manifest")
    args = ap.parse_args()

    boundaries = geo.Boundaries.load(config.DATA_DIR / "farms.json")
    regions = sorted({f.region for f in boundaries.farms.values()}) if args.all or args.list else args.region
    if not regions:
        ap.error("give --region NAME, --all or --list")

    total = 0
    for region in regions:
        cells = plan_region(boundaries, region, args.range, args.margin_deg)
        total += len(cells)
        print(f"{region:<22} {len(boundaries.farms_in(region)):>5} farms -> {len(cells):>4} captures (~{len(cells) * 85 / 60:.0f} min in Earth)")
        if args.list:
            continue
        grids = config.DATA_DIR / "grids"
        grids.mkdir(parents=True, exist_ok=True)
        for old in grids.glob(f"{slug(region)}_*.kml"):
            old.unlink()
        for n, start in enumerate(range(0, len(cells), args.per_kml), 1):
            capture.write_grid_kml(cells[start:start + args.per_kml], grids / f"{slug(region)}_{n}.kml", f"{slug(region)}_{n}")
        capture.update_manifest(config.DATA_DIR / "captures" / "manifest.json", cells, region, args.imagery_date)
    print(f"total: {total} captures")


if __name__ == "__main__":
    main()
