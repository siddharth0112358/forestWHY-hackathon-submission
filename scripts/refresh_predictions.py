#!/usr/bin/env python3
"""Re-extract dashboard fields from existing raw_response without re-inferring.

The fine-tune emits prose with a 10-step reasoning protocol. The normalizer
in `forestwhy.evaluator` extracts dashboard fields (change_class, severity,
driver_hypothesis, area_pct) by searching for synonyms in that prose. If the
synonym table changes (e.g. we add new patterns), this script re-applies the
new normalizer to every row's stored raw_response and updates the
predictions.db in place — no SimSat fetch, no model call.

Usage:
    uv run python scripts/refresh_predictions.py
    uv run python scripts/refresh_predictions.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from forestwhy.evaluator import _normalize  # noqa: E402

DB_PATH = REPO_ROOT / "predictions.db"
log = logging.getLogger("forestwhy.refresh_predictions")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(DB_PATH))
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would change without writing.")
    args = p.parse_args()

    if not Path(args.db).exists():
        log.error("DB not found: %s", args.db)
        sys.exit(1)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = list(conn.execute(
        "SELECT id, change_class, severity, driver_hypothesis, area_pct, "
        "       confidence, reasoning, raw_response "
        "FROM predictions ORDER BY id"
    ))
    log.info("Refreshing %d rows ...", len(rows))

    n_changed = 0
    n_filled = 0
    for r in rows:
        raw = r["raw_response"]
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("  row %d: raw_response is not JSON, skipping", r["id"])
            continue
        # The DB stores `prediction` (a dict), not `payload`. Re-normalise it.
        # If raw_response itself contains a "raw" key, we already normalised
        # it once — re-extract from the original raw payload.
        source = payload.get("raw", payload)
        norm = _normalize(source)

        # Diff
        before = {
            "change_class": r["change_class"],
            "severity": r["severity"],
            "driver_hypothesis": r["driver_hypothesis"],
            "area_pct": r["area_pct"],
            "confidence": r["confidence"],
        }
        after = {k: norm.get(k) for k in before}
        if before == after:
            continue

        diffs = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
        before_filled = sum(1 for v in before.values() if v not in (None, ""))
        after_filled = sum(1 for v in after.values() if v not in (None, ""))
        if after_filled > before_filled:
            n_filled += 1
        n_changed += 1

        log.info("  row %d: %s", r["id"],
                 ", ".join(f"{k}={d[0]!r}→{d[1]!r}" for k, d in diffs.items()))

        if args.dry_run:
            continue
        conn.execute(
            "UPDATE predictions SET change_class=?, severity=?, "
            "driver_hypothesis=?, area_pct=?, confidence=?, "
            "reasoning=COALESCE(?, reasoning) "
            "WHERE id=?",
            (
                norm.get("change_class"), norm.get("severity"),
                norm.get("driver_hypothesis"), norm.get("area_pct"),
                norm.get("confidence"),
                norm.get("reasoning") if not r["reasoning"] else None,
                r["id"],
            ),
        )

    if not args.dry_run:
        conn.commit()
    conn.close()
    log.info("Done. %d rows changed, %d had previously-None fields filled. (%s)",
             n_changed, n_filled, "DRY RUN" if args.dry_run else "WRITTEN")


if __name__ == "__main__":
    main()
