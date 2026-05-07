#!/usr/bin/env python3
"""Build before/after RGB GIFs from existing predictions for the README hero.

Picks the top-5 most demo-worthy rows from predictions.db (clear cloud
cover, high signal, geographic diversity), then for the headline three —
Acre, Madre de Dios, Borneo Kalimantan — combines `rgb_before.png` +
`rgb_after.png` into a 2-frame animated GIF saved to `assets/`.

Usage:
    uv run python scripts/make_demo_gifs.py
    uv run python scripts/make_demo_gifs.py --top-only       # just print rankings
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "predictions.db"
ASSETS = REPO_ROOT / "assets"

# Three GIFs the README hero references — region_id → output filename slug.
# Top-5 demo GIFs — drives the README hero + ARCHITECTURE.md examples.
HEADLINE_REGIONS = {
    "madagascar_east":    "madagascar",   # rank 1: 90 % area, 2/3 % cloud, slash-and-burn
    "congo_equateur":     "congo",        # rank 2: 90 % area, shifting cultivation
    "amazon_rondonia":    "rondonia",     # rank 3: 45 % area, 0 % cloud, fishbone settlement
    "sumatra_riau":       "sumatra",      # rank 4: 65 % area, peatland pulp plantations
    "borneo_kalimantan":  "kalimantan",   # rank 5: 65 % area, oil-palm conversion
}


def _load_recent_demo_rows(conn: sqlite3.Connection) -> list[dict]:
    """Return all source='demo' rows, joined to most-recent per region_id."""
    rows = list(conn.execute("""
        SELECT id, region_id, change_class, severity, area_pct, confidence,
               before_cloud_cover, after_cloud_cover, panels_dir,
               rgb_before_path, rgb_after_path, before_timestamp, after_timestamp
        FROM predictions
        WHERE source IN ('demo', 'backfill') AND panels_dir IS NOT NULL
        ORDER BY id DESC
    """))
    seen_regions: set[str] = set()
    out: list[dict] = []
    for r in rows:
        if r["region_id"] in seen_regions:
            continue
        seen_regions.add(r["region_id"])
        out.append(dict(r))
    return out


def _score(row: dict) -> float:
    """Heuristic — bigger = better demo material."""
    area = row.get("area_pct") or 0.0
    cc_b = row.get("before_cloud_cover") or 0.0
    cc_a = row.get("after_cloud_cover") or 0.0
    cloud_penalty = (cc_b + cc_a) / 2
    conf = row.get("confidence") or 0.0
    # Prefer rows where the model committed (deforestation/fire/afforestation),
    # not stable_forest or ambiguous, since those make less compelling visuals.
    cls_bonus = {
        "deforestation":     20,
        "fire_disturbance":  18,
        "afforestation":     15,
    }.get(row.get("change_class") or "", 0)
    return area + cls_bonus + conf * 10 - cloud_penalty


def _make_gif(before_png: Path, after_png: Path, out_path: Path,
              caption_before: str, caption_after: str, size: int = 320) -> None:
    """Two-frame animated GIF cycling before → after with labels."""
    a = Image.open(before_png).convert("RGB").resize((size, size), Image.BILINEAR)
    b = Image.open(after_png).convert("RGB").resize((size, size), Image.BILINEAR)

    def annotate(img: Image.Image, label: str) -> Image.Image:
        canvas = Image.new("RGB", (size, size + 28), (16, 22, 32))
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype(
                "/System/Library/Fonts/Helvetica.ttc", 13
            )
        except (OSError, IOError):
            font = ImageFont.load_default()
        draw.text((8, size + 6), label, fill=(220, 230, 245), font=font)
        return canvas

    frames = [annotate(a, caption_before), annotate(b, caption_after)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        str(out_path),
        save_all=True, append_images=frames[1:],
        duration=900, loop=0, optimize=True,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--top-only", action="store_true",
                   help="Print top-5 ranking, don't build GIFs.")
    args = p.parse_args()

    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = _load_recent_demo_rows(conn)
    if not rows:
        print("ERROR: no demo/backfill rows in predictions.db", file=sys.stderr)
        sys.exit(1)

    ranked = sorted(rows, key=_score, reverse=True)
    print(f"\n=== Top 5 demo candidates (of {len(rows)} regions) ===")
    print(f"{'rank':>4}  {'region':<22} {'class':<18} {'sev':<8} {'area':>5}  {'cc_b':>5} {'cc_a':>5}  score")
    for i, r in enumerate(ranked[:5], 1):
        ap = f"{r['area_pct']:.0f}" if r["area_pct"] is not None else "-"
        cc_b = f"{r['before_cloud_cover']:.0f}" if r["before_cloud_cover"] is not None else "-"
        cc_a = f"{r['after_cloud_cover']:.0f}" if r["after_cloud_cover"] is not None else "-"
        print(f"{i:>4}  {r['region_id']:<22} {r['change_class'] or '-':<18} {r['severity'] or '-':<8} {ap:>5}  {cc_b:>5} {cc_a:>5}  {_score(r):.1f}")

    if args.top_only:
        return

    # Build the three headline GIFs (Acre / Madre Dios / Kalimantan)
    print(f"\n=== Building 3 headline GIFs in {ASSETS}/ ===")
    by_region = {r["region_id"]: r for r in rows}
    for region, slug in HEADLINE_REGIONS.items():
        r = by_region.get(region)
        if r is None:
            print(f"  SKIP {region}: no row in DB. Run scripts/demo_backfill.py first.")
            continue
        rgb_before = REPO_ROOT / (r["rgb_before_path"] or "")
        rgb_after = REPO_ROOT / (r["rgb_after_path"] or "")
        if not rgb_before.exists():
            # Fall back to the panel inside panels_dir
            rgb_before = REPO_ROOT / r["panels_dir"] / "rgb_before.png"
        if not rgb_after.exists():
            rgb_after = REPO_ROOT / r["panels_dir"] / "rgb_after.png"
        if not rgb_before.exists() or not rgb_after.exists():
            print(f"  SKIP {region}: panel files not found "
                  f"({rgb_before}, {rgb_after})")
            continue

        out = ASSETS / f"before_after_{slug}.gif"
        before_year = (r["before_timestamp"] or "")[:4] or "before"
        after_year = (r["after_timestamp"] or "")[:4] or "after"
        _make_gif(
            rgb_before, rgb_after, out,
            caption_before=f"{before_year}  ·  cloud {r['before_cloud_cover']:.0f}%"
                           if r["before_cloud_cover"] is not None else before_year,
            caption_after=f"{after_year}  ·  cloud {r['after_cloud_cover']:.0f}%"
                          if r["after_cloud_cover"] is not None else after_year,
        )
        cls = r["change_class"] or "-"
        sev = r["severity"] or "-"
        print(f"  -> {out.relative_to(REPO_ROOT)}  ({cls}, severity={sev}, area={r['area_pct'] or 0:.0f}%)")

    print("\nDone. Drop them into the README hero (already wired in the markdown).")


if __name__ == "__main__":
    main()
