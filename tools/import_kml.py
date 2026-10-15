"""Import every layer (regions and farms) of a boundaries KML into the project data folder.

Usage: python tools/import_kml.py "<boundaries>.kml" [--prefix TYM]

Writes data/farms.json. Without --prefix all farms of all regions are kept.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from palmai import config, geo  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kml")
    ap.add_argument("--prefix", default="", help="keep only FarmCodes starting with this, e.g. TYM")
    args = ap.parse_args()

    data = geo.extract_layers(Path(args.kml), args.prefix)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = config.DATA_DIR / "farms.json"
    out.write_text(json.dumps(data), encoding="utf-8")
    by_region = collections.Counter(f["region"] for f in data["farms"])
    prefixes = {r: collections.Counter(f["code"].split("-")[0] for f in data["farms"] if f["region"] == r).most_common(1)[0][0]
                for r in by_region}
    print(f"{len(data['farms'])} farms, {len(data['regions'])} regions -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    for region, n in by_region.most_common():
        print(f"  {region:<22} {n:>5} farms   code prefix {prefixes[region]}")


if __name__ == "__main__":
    main()
