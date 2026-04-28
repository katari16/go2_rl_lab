"""Interactive viewer for real-robot force recordings — localhost:6008.

Browse the CSV/JSON logs saved by the B-button recording in the deploy scripts.
Toggle ablations, choose which channel to plot (magnitude, Fx, Fy, Fz, tau_yaw,
tau_roll, tau_pitch), pick EMA vs raw, show/hide GT, set time window.

Usage:
    python deploy/deploy_real/logs/recordings/viewer.py
    # then open http://localhost:6008
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

RECORDINGS_DIR = Path(__file__).resolve().parent

LABELS = {
    # Group 1 — history / net / TCN / PD gains
    "go2_ablation_p1":  "P1 (4D, h=10, baseline)",
    "go2_ablation_p2":  "P2 (4D, h=20, baseline)",
    "go2_ablation_p3":  "P3 (4D, h=30, baseline)",
    "go2_ablation_p4":  "P4 (4D, h=40, baseline)",
    "go2_ablation_p5":  "P5 (4D, h=30, enc=[64,32])",
    "go2_ablation_p6":  "P6 (4D, h=30, enc=[256,128])",
    "go2_ablation_p12": "P12 (4D, h=30, TCN encoder)",
    "go2_ablation_p20": "P20 (4D, h=30, Kp=25 Kd=0.5)",
    "go2_ablation_j3":  "J3 (4D, h=40, est_acc w=50)",
    "go2_ablation_j5":  "J5 (4D, h=40, TCN + est_acc w=50)",
    "go2_ablation_p9":  "P9 (4D, h=30, compliance w=1)",
    "go2_ablation_p10": "P10 (4D, h=30, compliance w=5)",
    "go2_ablation_p17": "P17 (6D, h=30, enc=[256,128])",
    "go2_ablation_p18": "P18 (6D, payload 0-4kg)",
    "go2_payload_3kg":  "Payload 3kg (3D, h=20)",
    "go2_ablation_6dctrl_total50":            "6Dctrl-T50 (no est-acc)",
    "go2_ablation_6dctrl_total50_estacc_w10": "6Dctrl-T50 (est-acc w=10)",
    "go2_ablation_6dctrl_total50_estacc_w25": "6Dctrl-T50 (est-acc w=25)",
    "go2_ablation_6dctrl_total50_estacc_w50": "6Dctrl-T50 (est-acc w=50)",
}

CHANNELS = [
    ("force_mag", "|F| magnitude"),
    ("Fx",        "Fx"),
    ("Fy",        "Fy"),
    ("Fz",        "Fz"),
    ("tau_roll",  "tau_roll (6D only)"),
    ("tau_pitch", "tau_pitch (6D only)"),
    ("tau_yaw",   "tau_yaw (4D/6D)"),
]

# Stable-ish color mapping (palette cycled by stem order once scanned)
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#4472C4", "#ED7D31", "#70AD47", "#7030A0", "#FFC000",
    "#00B0F0", "#00ACC1", "#AB47BC", "#FF7043", "#66BB6A",
]


def scan_recordings(rec_dir: Path) -> dict[str, list[tuple[str, Path, Path | None]]]:
    """Return {stem: [(timestamp, csv_path, json_path_or_None), ...]} sorted oldest->newest."""
    out: dict[str, list[tuple[str, Path, Path | None]]] = defaultdict(list)
    for f in rec_dir.glob("*.csv"):
        m = re.match(r"(go2_\w+?)_rec\d+_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.csv", f.name)
        if not m:
            continue
        stem, ts = m.group(1), m.group(2)
        json_path = f.with_suffix(".json")
        out[stem].append((ts, f, json_path if json_path.exists() else None))
    for stem in out:
        out[stem].sort()
    return dict(out)


def _torque_series(json_path: Path | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (tau_roll, tau_pitch) for 6D runs, else (None, None). Uses JSON force_hat."""
    if json_path is None or not json_path.exists():
        return None, None
    with open(json_path) as f:
        data = json.load(f)
    if data.get("force_dim", 0) < 6:
        return None, None
    steps = data["steps"]
    fh = np.array([s["force_hat"] for s in steps], dtype=np.float32)
    # Channels 3, 4, 5 = tau_roll, tau_pitch, tau_yaw
    return fh[:, 3], fh[:, 4]


def load_series(csv_path: Path, json_path: Path | None, channel: str, use_ema: bool,
                t_min: float) -> tuple[np.ndarray, np.ndarray] | None:
    df = pd.read_csv(csv_path)
    t = df["t"].values.astype(np.float32)
    mask = t >= t_min
    if mask.sum() == 0:
        return None
    t = t[mask]
    t = t - t[0]

    if channel == "force_mag":
        col = "force_mag_ema" if use_ema else "force_mag"
        y = df[col].values[mask]
    elif channel == "Fx":
        y = df["Fx_hat"].values[mask] if "Fx_hat" in df.columns else None
    elif channel == "Fy":
        y = df["Fy_hat"].values[mask] if "Fy_hat" in df.columns else None
    elif channel == "Fz":
        y = df["Fz_hat"].values[mask] if "Fz_hat" in df.columns else None
    elif channel == "tau_yaw":
        col = "tau_yaw_ema" if use_ema else "tau_yaw_hat"
        y = df[col].values[mask] if col in df.columns else None
    elif channel in ("tau_roll", "tau_pitch"):
        roll, pitch = _torque_series(json_path)
        src = roll if channel == "tau_roll" else pitch
        if src is None:
            return None
        y = src[mask]
    else:
        return None

    if y is None:
        return None
    return t, np.asarray(y, dtype=np.float32)


# ── Dash app ──────────────────────────────────────────────────────────────────

RECORDINGS = scan_recordings(RECORDINGS_DIR)
STEMS = sorted(RECORDINGS.keys())
STEM_COLOR = {stem: _PALETTE[i % len(_PALETTE)] for i, stem in enumerate(STEMS)}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Force Recording Viewer"


def _ablation_checklist():
    return dcc.Checklist(
        id="chk-ablations",
        options=[{"label": " " + LABELS.get(s, s), "value": s} for s in STEMS],
        value=STEMS[:],
        labelStyle={"display": "block", "fontSize": "12px", "marginBottom": "3px"},
        inputStyle={"marginRight": "5px"},
    )


sidebar = html.Div([
    html.H5("Channel", style={"fontWeight": "700", "marginTop": "6px"}),
    dcc.RadioItems(
        id="channel",
        options=[{"label": " " + lbl, "value": ch} for ch, lbl in CHANNELS],
        value="force_mag",
        labelStyle={"display": "block", "fontSize": "13px", "marginBottom": "3px"},
        inputStyle={"marginRight": "5px"},
    ),

    html.Hr(),
    dbc.Checklist(
        id="toggles",
        options=[
            {"label": " Use EMA-smoothed", "value": "ema"},
            {"label": " Show GT line (|F|)",  "value": "gt"},
        ],
        value=["gt"],
        style={"fontSize": "13px"},
    ),

    html.Hr(),
    html.Div([
        html.Label("GT magnitude (N)", style={"fontSize": "12px"}),
        dcc.Input(id="gt", type="number", value=20.50, step=0.01, debounce=True,
                  style={"width": "100%", "fontSize": "13px"}),
    ]),
    html.Div([
        html.Label("Cut first N seconds", style={"fontSize": "12px", "marginTop": "6px"}),
        dcc.Input(id="tcut", type="number", value=5.0, step=0.5, debounce=True,
                  style={"width": "100%", "fontSize": "13px"}),
    ]),
    html.Div([
        html.Label("Recordings per run", style={"fontSize": "12px", "marginTop": "6px"}),
        dcc.Slider(id="n_per_run", min=1, max=5, step=1, value=2,
                   marks={i: str(i) for i in range(1, 6)}),
    ]),

    html.Hr(),
    html.H6("Ablations", style={"fontWeight": "700"}),
    dbc.Row([
        dbc.Col(dbc.Button("All", id="btn-all", size="sm", outline=True,
                           color="secondary", style={"fontSize": "11px", "width": "100%"})),
        dbc.Col(dbc.Button("None", id="btn-none", size="sm", outline=True,
                           color="secondary", style={"fontSize": "11px", "width": "100%"})),
    ], className="mb-2"),
    _ablation_checklist(),
], style={
    "width": "280px", "minWidth": "280px", "padding": "16px 12px",
    "background": "#f8f9fa", "borderRight": "1px solid #dee2e6",
    "overflowY": "auto", "height": "calc(100vh - 56px)",
})


app.layout = html.Div([
    html.Div([
        html.Span("Force Recording Viewer",
                  style={"fontSize": "18px", "fontWeight": "700", "color": "white"}),
        html.Span(f" — {sum(len(v) for v in RECORDINGS.values())} recordings, "
                  f"{len(STEMS)} ablations",
                  style={"fontSize": "13px", "color": "#cce", "marginLeft": "8px"}),
    ], style={"background": "#2c3e50", "padding": "10px 20px",
              "display": "flex", "alignItems": "center", "height": "56px"}),
    html.Div([
        sidebar,
        html.Div([
            dcc.Graph(id="main-graph", style={"height": "calc(100vh - 260px)"}),
            html.Div(id="mae-table", style={"padding": "10px", "fontSize": "12px"}),
        ], style={"flex": "1", "overflow": "hidden"}),
    ], style={"display": "flex"}),
])


@app.callback(Output("chk-ablations", "value"),
              Input("btn-all", "n_clicks"),
              Input("btn-none", "n_clicks"),
              prevent_initial_call=True)
def _toggle_all(n_all, n_none):
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    if trig == "btn-all":
        return STEMS[:]
    return []


@app.callback(
    Output("main-graph", "figure"),
    Output("mae-table", "children"),
    Input("chk-ablations", "value"),
    Input("channel", "value"),
    Input("toggles", "value"),
    Input("gt", "value"),
    Input("tcut", "value"),
    Input("n_per_run", "value"),
)
def update(selected, channel, toggles, gt, tcut, n_per_run):
    use_ema = "ema" in (toggles or [])
    show_gt = "gt" in (toggles or [])
    tcut = float(tcut if tcut is not None else 5.0)
    gt = float(gt if gt is not None else 20.50)

    fig = go.Figure()
    rows = []
    for stem in selected or []:
        recs = RECORDINGS.get(stem, [])[-int(n_per_run):]
        k = len(recs)
        base_label = LABELS.get(stem, stem)
        for j, (ts, csv_p, json_p) in enumerate(recs):
            res = load_series(csv_p, json_p, channel, use_ema, tcut)
            if res is None:
                continue
            t, y = res
            dash_style = "solid" if j == k - 1 else ("dash" if j == k - 2 else "dot")
            tag = f"run {j + 1}/{k}"
            # Compute MAE against GT only for magnitude
            if channel == "force_mag":
                mae = float(np.mean(np.abs(y - gt)))
                label = f"{base_label} ({tag}) | MAE={mae:.2f}N"
                rows.append((base_label, tag, len(y), mae,
                             float(np.mean(y)), float(np.std(y))))
            else:
                label = f"{base_label} ({tag}) | mean={np.mean(y):+.2f}"
                rows.append((base_label, tag, len(y), None,
                             float(np.mean(y)), float(np.std(y))))
            fig.add_trace(go.Scatter(
                x=t, y=y, mode="lines", name=label,
                line=dict(color=STEM_COLOR[stem], width=1.5, dash=dash_style),
                opacity=0.9,
                hovertemplate=f"<b>{base_label}</b> ({tag})<br>t=%{{x:.2f}}s<br>y=%{{y:.3f}}<extra></extra>",
            ))

    if show_gt and channel == "force_mag":
        fig.add_hline(y=gt, line=dict(color="black", width=2, dash="dash"),
                      annotation_text=f"GT={gt:.2f}N", annotation_position="top right")

    channel_lbl = dict(CHANNELS).get(channel, channel)
    fig.update_layout(
        title=f"{channel_lbl} (cut t<{tcut:.1f}s)",
        xaxis_title="Time (s)",
        yaxis_title=channel_lbl + (" (EMA)" if use_ema and channel in ("force_mag", "tau_yaw") else ""),
        margin=dict(l=40, r=20, t=40, b=40),
        hovermode="x unified",
        legend=dict(font=dict(size=10), bgcolor="rgba(255,255,255,0.9)"),
        xaxis=dict(rangeslider=dict(visible=True, thickness=0.05)),
    )
    if channel == "force_mag":
        fig.update_yaxes(rangemode="tozero")

    # ── Summary table ────────────────────────────────────────────────────
    if not rows:
        table = html.Div("No recordings for the current selection.",
                         style={"color": "#888"})
    else:
        headers = ["Ablation", "Run", "Steps", "MAE (N)", "Mean", "Std"]
        thead = html.Tr([html.Th(h, style={"borderBottom": "1px solid #ccc",
                                           "padding": "4px 8px", "textAlign": "left"})
                         for h in headers])
        tbody = []
        for label, tag, n, mae, mean_v, std_v in rows:
            cells = [label, tag, str(n),
                     f"{mae:.2f}" if mae is not None else "—",
                     f"{mean_v:+.2f}", f"{std_v:.2f}"]
            tbody.append(html.Tr([html.Td(c, style={"padding": "3px 8px",
                                                    "borderBottom": "1px solid #eee"})
                                  for c in cells]))
        table = html.Table([html.Thead(thead), html.Tbody(tbody)],
                           style={"width": "100%", "borderCollapse": "collapse",
                                  "fontFamily": "monospace"})

    return fig, table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=6008)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"Recordings dir: {RECORDINGS_DIR}")
    print(f"Scanned: {sum(len(v) for v in RECORDINGS.values())} files across "
          f"{len(STEMS)} ablations")
    print(f"Open → http://{args.host}:{args.port}")
    app.run(debug=False, host=args.host, port=args.port)
