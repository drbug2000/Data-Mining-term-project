"""
k_sweep_eval.py — neighborhood-size (K) selection for the boosted SKNCP model.

Selection protocol (leak-free):
    * The only quantities consulted to *choose* K are computed on the sorted
      SearchID 80/20 *internal* split (internal-train fits the model, internal-val
      scores it). External `click_validation` labels are NEVER read to pick K.
    * K is selected by the argmax of the internal-validation Cohen's d of the
      boosted decision score. Cohen's d is preferred over top-k F1 as the
      selection criterion because top-k F1 over only ~229 external positives (and
      a comparably thin internal-val positive set) is a high-variance estimator,
      whereas Cohen's d is a function of full-sample means and spread and is far
      more stable for hyper-parameter selection. F1 is *reported*, never tuned on.
    * External F1 / AUC / bootstrap appear in the output ONLY as the downstream
      result of the already-selected K; they do not participate in selection.

This script re-runs the full boosted pipeline (`task1_skncp_boosted.main`) once
per K, reading the per-K diagnostics it writes (internal/external Cohen's d were
added to that script's `diagnostics` block for exactly this sweep).

Prerequisite (content-score cache):
    python -X utf8 models/m04_gated/experiments/exp_content_strong.py

Run:
    python -X utf8 models/m06_skncp/experiments/k_sweep_eval.py
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.m06_skncp.experiments import task1_skncp_boosted as boosted

# Neighborhood sizes swept (step 50). The grid brackets the plateau on both
# sides so the internal-val d argmax is an interior point, not a grid edge.
K_GRID = list(range(50, 401, 50))

OUT_JSON = boosted.OUT_DIR / "k_sweep_results.json"
TMP_DIR = Path("/tmp")


def _run_one(k: int) -> dict:
    """Run the boosted pipeline at neighborhood size k; return its result dict."""
    boosted.K_SKNN = k
    boosted.OUT_JSON = TMP_DIR / f"k_sweep_boosted_K{k}.json"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):  # silence the per-run JSON dump
        boosted.main()
    with boosted.OUT_JSON.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    rows = []
    for k in K_GRID:
        r = _run_one(k)
        diag = r["diagnostics"]
        rows.append(
            {
                "k": k,
                # ---- selection signal (internal split only) ----
                "internal_val_cohens_d": diag["boosted_d_internal_val"],
                "internal_val_oracle_f1": r["internal"]["oracle_f1"],
                # ---- reported result (NOT used for selection) ----
                "external_f1_prevalence": r["external"]["f1_prevalence"]["f1"],
                "external_auc": r["external"]["auc"],
                "external_bootstrap_prevalence_mean": r["external"][
                    "bootstrap_prevalence"
                ].get("mean"),
                "external_cohens_d": diag["boosted_d_external"],
                "skncp_d_external": diag["skncp_d_external"],
            }
        )

    # Selection rule: argmax internal-validation Cohen's d. Leak-free.
    selected = max(rows, key=lambda x: x["internal_val_cohens_d"])

    out = {
        "selection_rule": (
            "argmax internal-validation Cohen's d of the boosted decision score; "
            "external labels are not used to select K"
        ),
        "k_grid": K_GRID,
        "selected_k": selected["k"],
        "selected": selected,
        "sweep": rows,
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Human-readable table.
    print(
        f"{'K':>4} {'intval_d':>9} {'intval_F1':>10} "
        f"{'ext_F1':>8} {'ext_AUC':>8} {'ext_d':>7} {'boot_mean':>9}"
    )
    for x in rows:
        mark = "  <- selected" if x["k"] == selected["k"] else ""
        print(
            f"{x['k']:>4} {x['internal_val_cohens_d']:>9.4f} "
            f"{x['internal_val_oracle_f1']:>10.4f} "
            f"{x['external_f1_prevalence']:>8.4f} {x['external_auc']:>8.4f} "
            f"{x['external_cohens_d']:>7.4f} "
            f"{x['external_bootstrap_prevalence_mean']:>9.4f}{mark}"
        )
    print(
        f"\nSelected K = {selected['k']} "
        f"(internal-val Cohen's d = {selected['internal_val_cohens_d']:.4f}); "
        f"reported external F1 = {selected['external_f1_prevalence']:.4f}"
    )
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
