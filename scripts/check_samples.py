#!/usr/bin/env python3
"""Validate a directory of samples produced by `generate_samples.py`.

Checks: every sample dir has all 14 panels present, metadata.json + truth.json
exist, truth.json conforms to JSON_SCHEMA. Returns non-zero on any failure.

Usage:
    uv run python scripts/check_samples.py samples/20260506T123456
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import jsonschema

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from forestwhy.prompts import JSON_SCHEMA, PANEL_ORDER  # noqa: E402

log = logging.getLogger("forestwhy.check_samples")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("directory", help="Sample directory to validate.")
    args = p.parse_args()

    root = Path(args.directory)
    if not root.is_dir():
        log.error("Not a directory: %s", root)
        sys.exit(1)

    failures = 0
    samples = [p for p in root.iterdir() if p.is_dir()]
    log.info("Checking %d sample dirs in %s ...", len(samples), root)

    for d in samples:
        for name in PANEL_ORDER:
            if not (d / f"{name}.png").exists():
                log.error("  %s: missing panel %s.png", d.name, name)
                failures += 1
        meta = d / "metadata.json"
        truth = d / "truth.json"
        if not meta.exists():
            log.error("  %s: missing metadata.json", d.name)
            failures += 1
        if not truth.exists():
            log.error("  %s: missing truth.json", d.name)
            failures += 1
            continue
        try:
            payload = json.loads(truth.read_text())
            jsonschema.validate(payload, JSON_SCHEMA)
        except Exception as exc:
            log.error("  %s: truth.json invalid (%s)", d.name, exc)
            failures += 1

    if failures:
        log.error("FAILED: %d issues across %d samples", failures, len(samples))
        sys.exit(1)
    log.info("OK: all %d samples passed.", len(samples))


if __name__ == "__main__":
    main()
