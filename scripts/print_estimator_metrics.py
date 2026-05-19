"""Print estimator metrics from a rollout_estimator_eval metrics.json.

Usage:
    python scripts/print_estimator_metrics.py <path_to_metrics.json> [--basket 30]
"""

import argparse
import json
import sys

parser = argparse.ArgumentParser()
parser.add_argument("metrics", type=str, help="Path to metrics.json")
parser.add_argument("--basket", type=str, default=None,
                    help="Basket key to print (e.g. '30'). Default: all baskets.")
args = parser.parse_args()

with open(args.metrics) as f:
    data = json.load(f)

baskets = data.get("baskets", data)
if args.basket:
    keys = [k for k in baskets if str(args.basket) in k]
    if not keys:
        print(f"Basket '{args.basket}' not found. Available: {list(baskets.keys())}")
        sys.exit(1)
else:
    keys = list(baskets.keys())

for key in keys:
    m = baskets[key]
    print(f"\n{'='*55}")
    print(f"  Basket: {key}  (range {m.get('force_range', '?')} N)")
    print(f"{'='*55}")

    def row(label, val, std=None, unit=""):
        std_str = f" ± {std:.3f}" if std is not None else ""
        print(f"  {label:<30s} {val:>7.3f}{std_str} {unit}")

    row("MAE total",       m["mae"],       m.get("mae_std"),       "N")
    row("Median AE",       m["median_ae"], None,                   "N")

    pam  = m.get("per_axis_mae", {})
    pams = m.get("per_axis_mae_std", {})
    for axis_label, val in pam.items():
        unit = "Nm" if "τ" in axis_label else "N"
        row(f"  MAE {axis_label}", val, pams.get(axis_label), unit)

    row("Force MAE",       m["force_mae"],    m.get("force_mae_std"),    "N")
    if m.get("torque_mae") is not None:
        row("Torque MAE",  m["torque_mae"],   m.get("torque_mae_std"),   "Nm")

    row("Ang err mean (XY)",   m.get("angular_err_xy_deg_mean",   0.0), m.get("angular_err_xy_deg_mean_std"),   "deg")
    row("Ang err median (XY)", m.get("angular_err_xy_deg_median", 0.0), m.get("angular_err_xy_deg_median_std"), "deg")
    row("Rel err",             m.get("relative_err_pct", 0.0), None, "%")

print()
