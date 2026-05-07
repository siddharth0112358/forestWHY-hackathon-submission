#!/usr/bin/env python3
"""Re-run the normaliser + recompute metrics over an existing results.json.

Use this when you've changed `forestwhy.evaluator._normalize` (synonym table,
negation rules, etc.) and want updated dashboard fields + accuracy numbers
WITHOUT re-running the 25-min model inferences. The base/finetuned model
outputs (under each sample's `base.raw` and `finetuned.raw` keys) are
preserved verbatim from the original run.

Usage:
    uv run python scripts/rescore_results.py evals/<timestamp>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from forestwhy.evaluator import _normalize  # noqa: E402
from evaluate import (  # noqa: E402
    PRED_TO_TRUTH_LABEL,
    _accuracy,
    _binary_deforestation_pred,
    _mae,
    _mapped_change_class,
)

log = logging.getLogger("forestwhy.rescore")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("eval_dir", type=Path)
    args = p.parse_args()

    src = args.eval_dir / "results.json"
    if not src.exists():
        log.error("Not found: %s", src)
        sys.exit(1)

    with open(src) as f:
        data = json.load(f)
    records = data["samples"]
    log.info("Re-normalising %d samples ...", len(records))

    # Re-apply the current _normalize() to each side's raw payload.
    for r in records:
        for side in ("base", "finetuned"):
            payload = r.get(side) or {}
            raw = payload.get("raw")
            if not isinstance(raw, dict):
                continue
            new = _normalize(raw)
            r[side] = new

    # Recompute metrics
    truths = [r["truth"] for r in records]

    def col(side, k):
        return [r[side].get(k) if isinstance(r[side], dict) else None for r in records]

    summary = {
        "n_samples": len(records),
        "n_deforestation_truth": sum(1 for t in truths if t["deforestation"]),
        "n_other_truth": sum(1 for t in truths if not t["deforestation"]),
    }

    metrics = [
        ("change_class accuracy (mapped)",
         lambda side: _accuracy(
             [_mapped_change_class(r[side]) if isinstance(r[side], dict) else None
              for r in records],
             [t["change_class"] for t in truths])),
        ("binary deforestation acc.",
         lambda side: _accuracy(
             [_binary_deforestation_pred(r[side]) if isinstance(r[side], dict) else None
              for r in records],
             [t["deforestation"] for t in truths])),
        ("driver accuracy",
         lambda side: _accuracy(col(side, "driver_hypothesis"),
                                [t["driver_hypothesis"] for t in truths])),
        ("area_pct MAE (lower=better)",
         lambda side: _mae(col(side, "area_pct"), [t["area_pct"] for t in truths])),
    ]
    for label, fn in metrics:
        summary[label] = {"base": fn("base"), "finetuned": fn("finetuned")}

    # Write new artefacts
    out_dir = args.eval_dir
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = {"summary": summary, "samples": records}
    (out_dir / f"results_rescored_{suffix}.json").write_text(json.dumps(out, indent=2, default=str))

    # Markdown report
    lines = [
        f"# forestWHY evaluation — {out_dir.name} (re-scored {suffix})",
        "",
        f"- Samples scored: **{summary['n_samples']}**",
        f"- Sampling: 50/50 deforestation / other "
        f"({summary['n_deforestation_truth']}/{summary['n_other_truth']} actual)",
        f"- Re-scored from existing model outputs with the current `_normalize` synonym table.",
        "",
        "| Metric | Base | Fine-tuned | Δ |",
        "|---|---|---|---|",
    ]
    for label, _ in metrics:
        b = summary[label]["base"]
        f_ = summary[label]["finetuned"]
        b_s = "—" if b is None else f"{b:.3f}"
        f_s = "—" if f_ is None else f"{f_:.3f}"
        d_s = ""
        if isinstance(b, (int, float)) and isinstance(f_, (int, float)):
            d_s = f"{f_ - b:+.3f}"
        lines.append(f"| {label} | {b_s} | {f_s} | {d_s} |")
    (out_dir / f"report_rescored_{suffix}.md").write_text("\n".join(lines) + "\n")
    log.info("Wrote %s", out_dir / f"report_rescored_{suffix}.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
