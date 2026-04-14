"""Compare rollout estimator evaluations across multiple runs.

Interactive terminal picker to select runs, then generates a PDF with:
  1. Per-basket overlaid force time series (envs 0-4) across runs
  2. Per-basket overlaid torque time series (envs 0-4) across runs
  3. Per-basket grouped bar chart comparing metrics across runs
  4. Combined summary table across all baskets and runs

Usage:
    python scripts/rsl_rl/eval/compare_rollout_estimators.py
    python scripts/rsl_rl/eval/compare_rollout_estimators.py --scan_dir data/eval
    python scripts/rsl_rl/eval/compare_rollout_estimators.py --runs path/to/run1 path/to/run2
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUN_COLORS = [
    "tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
]

DIM_LABELS = {
    2: ["Fx", "Fy"],
    3: ["Fx", "Fy", "Fz"],
    4: ["Fx", "Fy", "Fz", "τ_yaw"],
    6: ["Fx", "Fy", "Fz", "τ_roll", "τ_pitch", "τ_yaw"],
}


def find_rollout_runs(scan_dir):
    """Find all directories containing metrics.json from rollout_estimator evals."""
    runs = []
    for root, dirs, files in os.walk(scan_dir):
        if "metrics.json" in files and "rollout_estimator" in root:
            runs.append(root)
    runs.sort()
    return runs


def pick_runs(runs):
    """Interactive terminal picker for selecting runs."""
    try:
        from pick import pick
    except ImportError:
        print("Install pick: pip install pick")
        sys.exit(1)

    labels = []
    for r in runs:
        # Extract a short label from the path
        parts = r.split(os.sep)
        # Typically: data/eval/<experiment>/<rollout_estimator_timestamp_suffix>
        label = os.sep.join(parts[-2:]) if len(parts) >= 2 else r
        labels.append(label)

    title = "Select runs to compare (SPACE to select, ENTER to confirm):"
    selected = pick(labels, title, multiselect=True, min_selection_count=2)
    return [runs[idx] for _, idx in selected]


def load_run(run_dir):
    """Load metrics and sample data from a run directory."""
    with open(os.path.join(run_dir, "metrics.json")) as f:
        metrics = json.load(f)

    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)

    sample_data = {}
    for fname in os.listdir(run_dir):
        if fname.startswith("sample_envs_") and fname.endswith(".json"):
            basket_key = fname.replace("sample_envs_", "").replace(".json", "")
            with open(os.path.join(run_dir, fname)) as f:
                sample_data[basket_key] = json.load(f)

    # Short label for legend
    parts = run_dir.rstrip(os.sep).split(os.sep)
    short_label = parts[-1] if parts else run_dir
    task = metrics.get("task", config.get("task", "unknown"))

    return {
        "dir": run_dir,
        "metrics": metrics,
        "config": config,
        "sample_data": sample_data,
        "label": short_label,
        "task": task,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare rollout estimator evaluations.")
    parser.add_argument("--scan_dir", type=str, default="data/eval",
                        help="Directory to scan for rollout_estimator runs.")
    parser.add_argument("--runs", type=str, nargs="+", default=None,
                        help="Explicit run directories (skip picker).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PDF path (default: auto in data/eval/comparisons/).")
    parser.add_argument("--sample_env", type=int, default=0,
                        help="Which sample env to show in time series (0-4, default: 0).")
    args = parser.parse_args()

    if args.runs:
        run_dirs = args.runs
    else:
        all_runs = find_rollout_runs(args.scan_dir)
        if len(all_runs) < 2:
            print(f"Found {len(all_runs)} runs in {args.scan_dir}. Need at least 2.")
            sys.exit(1)
        print(f"Found {len(all_runs)} rollout_estimator runs.\n")
        run_dirs = pick_runs(all_runs)

    runs = [load_run(d) for d in run_dirs]
    n_runs = len(runs)
    print(f"\nComparing {n_runs} runs:")
    for i, r in enumerate(runs):
        print(f"  [{i}] {r['label']}  ({r['task']})")

    # Normalize old (v1, no baskets) metrics into the basket format
    for r in runs:
        m = r["metrics"]
        if "baskets" not in m:
            basket_key = "default"
            m["baskets"] = {basket_key: {
                "mae": m["mae"],
                "median_ae": m["median_ae"],
                "relative_err_pct": m["relative_err_pct"],
                "angular_err_xy_deg_mean": m["angular_err_xy_deg_mean"],
                "angular_err_xy_deg_median": m.get("angular_err_xy_deg_median", 0.0),
                "per_axis_mae": m["per_axis_mae"],
                "force_mae": m.get("force_mae", 0.0),
                "torque_mae": m.get("torque_mae"),
            }}

    # Find common baskets
    all_baskets = set()
    for r in runs:
        all_baskets.update(r["metrics"]["baskets"].keys())
    common_baskets = sorted(all_baskets, key=lambda b: float(b.replace("N", "")) if b != "default" else 0)

    force_dim = runs[0]["metrics"]["force_dim"]
    dim_labels = DIM_LABELS.get(force_dim, [f"d{i}" for i in range(force_dim)])
    dt = runs[0]["metrics"]["dt"]
    n_force_axes = min(force_dim, 3)
    torque_indices = list(range(3, force_dim))
    force_colors_ax = ["tab:blue", "tab:red", "tab:green"]
    torque_colors_ax = ["tab:purple", "tab:brown", "tab:orange"]

    figures = []
    env_i_str = str(args.sample_env)

    # ═══════════════════════════════════════════════════════════════════
    # Per-basket: overlaid time series for sample env
    # ═══════════════════════════════════════════════════════════════════

    for basket_key in common_baskets:
        # Check which runs have this basket's sample data
        available = [r for r in runs if basket_key in r["sample_data"]
                     and env_i_str in r["sample_data"][basket_key]]
        if not available:
            continue

        # ── Force time series ───────────────────────────────────────────
        fig, axes_grid = plt.subplots(n_force_axes, 1, figsize=(14, 3.5 * n_force_axes))
        if n_force_axes == 1:
            axes_grid = [axes_grid]
        fig.suptitle(f"Force Estimate Comparison — {basket_key} — Env {args.sample_env}",
                     fontsize=14, fontweight="bold")

        for col in range(n_force_axes):
            ax = axes_grid[col]
            # Plot GT from first run (they should all be identical)
            r0 = available[0]
            gt = np.array(r0["sample_data"][basket_key][env_i_str]["gt"])
            time_s = np.arange(len(gt)) * dt
            ax.plot(time_s, gt[:, col], color="black", linewidth=1.5, alpha=0.8,
                    label="GT", zorder=10)

            # Overlay estimates from each run
            for ri, r in enumerate(available):
                est = np.array(r["sample_data"][basket_key][env_i_str]["est"])
                ax.plot(time_s, est[:, col], color=RUN_COLORS[ri % len(RUN_COLORS)],
                        linewidth=1.0, alpha=0.7, linestyle="--", label=r["label"])

            ax.set_ylabel(f"{dim_labels[col]} (N)", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            if col == 0:
                ax.legend(fontsize=7, ncol=min(n_runs + 1, 4), loc="upper right")
            if col == n_force_axes - 1:
                ax.set_xlabel("Time (s)", fontsize=10)
        plt.tight_layout()
        figures.append(fig)

        # ── Torque time series ──────────────────────────────────────────
        if torque_indices:
            n_tq = len(torque_indices)
            fig, axes_grid = plt.subplots(n_tq, 1, figsize=(14, 3.5 * n_tq))
            if n_tq == 1:
                axes_grid = [axes_grid]
            fig.suptitle(f"Torque Estimate Comparison — {basket_key} — Env {args.sample_env}",
                         fontsize=14, fontweight="bold")

            for col_i, d in enumerate(torque_indices):
                ax = axes_grid[col_i]
                r0 = available[0]
                gt = np.array(r0["sample_data"][basket_key][env_i_str]["gt"])
                time_s = np.arange(len(gt)) * dt
                ax.plot(time_s, gt[:, d], color="black", linewidth=1.5, alpha=0.8,
                        label="GT", zorder=10)

                for ri, r in enumerate(available):
                    est = np.array(r["sample_data"][basket_key][env_i_str]["est"])
                    ax.plot(time_s, est[:, d], color=RUN_COLORS[ri % len(RUN_COLORS)],
                            linewidth=1.0, alpha=0.7, linestyle="--", label=r["label"])

                ax.set_ylabel(f"{dim_labels[d]} (Nm)", fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)
                if col_i == 0:
                    ax.legend(fontsize=7, ncol=min(n_runs + 1, 4), loc="upper right")
                if col_i == n_tq - 1:
                    ax.set_xlabel("Time (s)", fontsize=10)
            plt.tight_layout()
            figures.append(fig)

    # ═══════════════════════════════════════════════════════════════════
    # Grouped bar charts: metrics comparison per basket
    # ═══════════════════════════════════════════════════════════════════

    metric_keys = [
        ("mae", "MAE (N)"),
        ("median_ae", "Median AE (N)"),
        ("relative_err_pct", "Rel. Error (%)"),
        ("angular_err_xy_deg_mean", "Ang. Error XY (°)"),
    ]

    for basket_key in common_baskets:
        available = [r for r in runs if basket_key in r["metrics"]["baskets"]]
        if not available:
            continue

        n_metrics = len(metric_keys)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 5))
        fig.suptitle(f"Metrics Comparison — {basket_key}", fontsize=14, fontweight="bold")

        for mi, (mkey, mlabel) in enumerate(metric_keys):
            ax = axes[mi]
            vals = []
            labels = []
            colors = []
            for ri, r in enumerate(available):
                bm = r["metrics"]["baskets"][basket_key]
                vals.append(bm[mkey])
                labels.append(r["label"][:25])
                colors.append(RUN_COLORS[ri % len(RUN_COLORS)])

            x = np.arange(len(vals))
            ax.bar(x, vals, color=colors, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel(mlabel, fontsize=10)
            ax.set_title(mlabel, fontsize=11)
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        figures.append(fig)

    # ── Per-axis MAE grouped bar chart per basket ───────────────────────
    for basket_key in common_baskets:
        available = [r for r in runs if basket_key in r["metrics"]["baskets"]]
        if not available:
            continue

        n_axes = force_dim
        x = np.arange(n_axes)
        width = 0.8 / len(available)

        fig, ax = plt.subplots(figsize=(max(8, 2 * n_axes), 5))
        fig.suptitle(f"Per-Axis MAE — {basket_key}", fontsize=14, fontweight="bold")

        for ri, r in enumerate(available):
            bm = r["metrics"]["baskets"][basket_key]
            vals = [bm["per_axis_mae"].get(dim_labels[d], 0.0) for d in range(force_dim)]
            offset = (ri - len(available) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, color=RUN_COLORS[ri % len(RUN_COLORS)],
                   alpha=0.8, label=r["label"][:30])

        ax.set_xticks(x)
        ax.set_xticklabels(dim_labels, fontsize=10)
        ax.set_ylabel("MAE", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        figures.append(fig)

    # ═══════════════════════════════════════════════════════════════════
    # Summary table: all baskets × all runs
    # ═══════════════════════════════════════════════════════════════════

    fig_height = 2 + len(common_baskets) * (3 + force_dim) * 0.35
    fig, ax = plt.subplots(figsize=(5 + 3 * n_runs, min(fig_height, 30)))
    ax.axis("off")

    headers = ["Basket", "Metric"] + [r["label"][:25] for r in runs]
    table_rows = []
    for basket_key in common_baskets:
        first_in_basket = True
        for mkey, mlabel in metric_keys:
            row = [basket_key if first_in_basket else "", mlabel]
            for r in runs:
                bm = r["metrics"]["baskets"].get(basket_key, {})
                val = bm.get(mkey, None)
                row.append(f"{val:.2f}" if val is not None else "—")
            table_rows.append(row)
            first_in_basket = False

        for d in range(force_dim):
            unit = "Nm" if d >= 3 else "N"
            row = ["", f"MAE {dim_labels[d]} ({unit})"]
            for r in runs:
                bm = r["metrics"]["baskets"].get(basket_key, {})
                pam = bm.get("per_axis_mae", {})
                val = pam.get(dim_labels[d], None)
                row.append(f"{val:.3f}" if val is not None else "—")
            table_rows.append(row)

    table = ax.table(cellText=table_rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    fig.suptitle("Rollout Estimator Comparison — Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    figures.append(fig)

    # ── Save ────────────────────────────────────────────────────────────
    if args.output:
        out_path = args.output
    else:
        os.makedirs("data/eval/comparisons", exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = f"data/eval/comparisons/rollout_comparison_{ts}.pdf"

    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out_path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\n[compare] PDF saved: {out_path}")
    print(f"[compare] Pages: {len(figures)}")


if __name__ == "__main__":
    main()
