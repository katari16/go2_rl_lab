"""Dash app: force-estimator ablation comparison — 3 tabs.

  Tab 1  Sim eval (rollout / OU)        — bar charts with ± std error bars
  Tab 2  Static-eval timeseries metrics — bar charts (noise / settling / integral / ...)
  Tab 3  Static-eval timeseries plots   — overlay GT vs estimate curves for selected runs

Reads:
  report_ablation_metrics.json                       (tab 1)
  data/static_eval_ablations/static_eval_timeseries_metrics.json   (tab 2)
  data/static_eval_ablations/<run>/static_eval_data_*.json         (tab 3)

Run:  python scripts/dashboard_ablations.py [--port 8050] [--host 127.0.0.1]
"""

import argparse
import glob
import json
import os

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

ROOT = "/home/ubuntu/go2_rl_lab"
STATIC_DIR = f"{ROOT}/data/static_eval_ablations"

_ap = argparse.ArgumentParser()
_ap.add_argument("--port", type=int, default=8050)
_ap.add_argument("--host", type=str, default="127.0.0.1")
_ap.add_argument("--sim_metrics", type=str, default=f"{ROOT}/report_ablation_metrics.json")
_ap.add_argument("--ts_metrics", type=str, default=f"{STATIC_DIR}/static_eval_timeseries_metrics.json")
_ap.add_argument("--realworld", type=str, default=f"{ROOT}/data/recordings/realworld_recordings_story.json")
_args = _ap.parse_args()

SIM = json.load(open(_args.sim_metrics))
TS  = json.load(open(_args.ts_metrics)) if os.path.exists(_args.ts_metrics) else None
RW  = json.load(open(_args.realworld)) if os.path.exists(_args.realworld) else None

# ── discover static_360 direction_metrics.json files ──────────────────────────
import re as _re
DIR_METRICS = {}   # label -> {"path":..., "exp":..., "data":...}
for _p in sorted(glob.glob(f"{ROOT}/data/eval/*/static_360_*/direction_metrics.json")):
    _exp = os.path.basename(os.path.dirname(os.path.dirname(_p)))      # ablation_<...>
    _folder = os.path.basename(os.path.dirname(_p))                    # static_360_<date>_<RID>_<map>_<N>N
    _rid = _exp.replace("ablation_", "").split("_")[0].upper()
    _m = _re.search(r"_(\d{4}-\d{2}-\d{2})_.*?_(\d+)N$", _folder)
    if _m:
        _date, _fn = _m.group(1), _m.group(2)
        _lbl = f"{_rid} · {_fn}N · {_date}"
    else:
        _lbl = f"{_rid} · {_folder[:24]}"
    # newest first when multiple share a label-prefix
    _key = _lbl
    i = 2
    while _key in DIR_METRICS:
        _key = f"{_lbl} ({i})"; i += 1
    DIR_METRICS[_key] = {"path": _p, "exp": _exp, "folder": _folder, "data": json.load(open(_p))}
# order: newest folder first
DIR_METRICS = dict(sorted(DIR_METRICS.items(), key=lambda kv: kv[1]["folder"], reverse=True))

GROUPS   = SIM["ablation_groups"]
SIM_RUNS = SIM["runs"]
TS_RUNS  = (TS or {}).get("runs", {})
RUN_IDS  = list(SIM_RUNS.keys())

COLORS = ["#1d3557", "#457b9d", "#2a6496", "#74c2e1", "#a8dadc", "#e76f51", "#f4a261", "#9b5de5",
          "#264653", "#2a9d8f", "#e9c46a", "#f4845f", "#52796f", "#84a98c", "#cad2c5"]

# ── Tab 1 metric options (sim eval) ───────────────────────────────────────────
SIM_METRIC_OPTIONS = [
    ("mae", "Overall MAE (per-dim)"),
    ("median_ae", "Median AE (rollout only)"),
    ("force_mae", "Force MAE"),
    ("torque_mae", "Torque MAE"),
    ("per_axis_mae:Fx", "MAE Fx"),
    ("per_axis_mae:Fy", "MAE Fy"),
    ("per_axis_mae:Fz", "MAE Fz"),
    ("per_axis_mae:τ_yaw", "MAE τ_yaw"),
    ("angular_err_xy_deg_mean", "Angular err XY (mean, deg)"),
    ("angular_err_xy_deg_median", "Angular err XY (median, deg)"),
    ("relative_err_pct", "Relative err (%)"),
]
SIM_SOURCES = [("rollout_eval", "Rollout (training regime)"), ("ou_eval", "OU disturbance")]

# ── Tab 2 metric options (static-eval timeseries) ─────────────────────────────
TS_AGG_METRICS = [
    ("agg:mae", "Aggregate MAE (per-dim)"),
    ("agg:force_mae", "Force MAE"),
    ("agg:torque_mae", "Torque MAE"),
    ("agg:angular_err_xy_deg_mean", "Angular err XY (mean, deg)"),
    ("agg:angular_err_xy_deg_median", "Angular err XY (median, deg)"),
    ("agg:mean_ss_noise_std_force", "Mean steady-state noise std — force chans"),
    ("agg:mean_settling_time_s_force", "Mean settling time (s) — force chans"),
    ("agg:mean_rel_integral_err_force", "Mean rel. integral err — force chans"),
]
TS_CHAN_METRICS = [
    ("mae", "MAE"),
    ("bias", "Bias (signed)"),
    ("rmse", "RMSE"),
    ("ss_mae", "Steady-state MAE"),
    ("ss_noise_std", "Steady-state noise std"),
    ("step_noise_std", "High-freq (step) noise std"),
    ("transient_mae", "Transient MAE (first 0.5 s)"),
    ("settling_time_s", "Settling time (s)"),
    ("overshoot_pct", "Overshoot (%)"),
    ("integral_abs_err", "∫|est−gt| dt (N·s)"),
    ("integral_signed_err", "∫(est−gt) dt (N·s)"),
    ("rel_integral_err", "∫|est−gt| / ∫|gt|"),
]
TS_CHANNELS = ["Fx", "Fy", "Fz", "τ_roll", "τ_pitch", "τ_yaw"]

# ── Tab 3: raw timeseries data per run ────────────────────────────────────────
_RAW_TS_CACHE = {}
def _load_raw_ts(run_id):
    if run_id in _RAW_TS_CACHE:
        return _RAW_TS_CACHE[run_id]
    hits = sorted(glob.glob(f"{STATIC_DIR}/{run_id}/static_eval_data_*.json"))
    data = json.load(open(hits[-1])) if hits else None
    _RAW_TS_CACHE[run_id] = data
    return data

TS_PLOT_CHANNELS = [
    ("Fx", "gt_force_x", "est_force_x", "N"),
    ("Fy", "gt_force_y", "est_force_y", "N"),
    ("Fz", "gt_force_z", "est_force_z", "N"),
    ("τ_roll", "gt_torque_roll", "est_torque_roll", "Nm"),
    ("τ_pitch", "gt_torque_pitch", "est_torque_pitch", "Nm"),
    ("τ_yaw", "gt_torque_yaw", "est_torque_yaw", "Nm"),
]


def _sim_val(entry, key):
    if entry is None:
        return None, None
    if ":" in key:
        base, axis = key.split(":", 1)
        return (entry.get(base) or {}).get(axis), (entry.get(base + "_std") or {}).get(axis)
    return entry.get(key), entry.get(key + "_std")


# ════════════════════════════════════════════════════════════════════════════
app = dash.Dash(__name__)
app.title = "Estimator Ablations"

_dropdown = lambda **k: dcc.Dropdown(clearable=False, **k)


def _group_dd(idd):
    return _dropdown(id=idd, options=[{"label": g["name"], "value": g["name"]} for g in GROUPS],
                     value=GROUPS[0]["name"], style={"width": "260px"})


tab1 = html.Div([
    html.P(SIM["description"], style={"color": "#555", "fontSize": "13px"}),
    html.Div(style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "marginBottom": "12px"}, children=[
        html.Div([html.Label("Ablation group", style={"fontWeight": "bold"}), _group_dd("s_group")]),
        html.Div([html.Label("Metric", style={"fontWeight": "bold"}),
                  _dropdown(id="s_metric", options=[{"label": l, "value": k} for k, l in SIM_METRIC_OPTIONS],
                            value="per_axis_mae:Fx", style={"width": "300px"})]),
        html.Div([html.Label("Eval source", style={"fontWeight": "bold"}),
                  _dropdown(id="s_source", options=[{"label": l, "value": k} for k, l in SIM_SOURCES],
                            value="rollout_eval", style={"width": "240px"})]),
    ]),
    html.Div(id="s_desc", style={"background": "#f5f5f5", "padding": "12px", "borderRadius": "6px",
                                 "fontSize": "13px", "marginBottom": "12px"}),
    dcc.Graph(id="s_bar"),
    html.H4("Both eval sources side-by-side"),
    dcc.Graph(id="s_bar_both"),
])

tab2 = html.Div([
    html.P((TS or {}).get("description", "static_eval_timeseries_metrics.json not found"),
           style={"color": "#555", "fontSize": "13px"}),
    html.Div(style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "marginBottom": "12px"}, children=[
        html.Div([html.Label("Ablation group", style={"fontWeight": "bold"}), _group_dd("t_group")]),
        html.Div([html.Label("Scope", style={"fontWeight": "bold"}),
                  _dropdown(id="t_scope", options=[{"label": "Aggregate", "value": "agg"}]
                            + [{"label": f"Channel {c}", "value": c} for c in TS_CHANNELS],
                            value="agg", style={"width": "200px"})]),
        html.Div([html.Label("Metric", style={"fontWeight": "bold"}),
                  _dropdown(id="t_metric", options=[], value=None, style={"width": "320px"})]),
    ]),
    html.Div(id="t_desc", style={"background": "#f5f5f5", "padding": "12px", "borderRadius": "6px",
                                 "fontSize": "13px", "marginBottom": "12px"}),
    dcc.Graph(id="t_bar"),
])

tab3 = html.Div([
    html.P("Overlay GT (step) vs estimated (line) for selected runs. All runs share the same "
           "force profile (seed 42, |F|∈[10,30] N/axis, re-randomised every 1–3 s over 20 s).",
           style={"color": "#555", "fontSize": "13px"}),
    html.Div(style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "marginBottom": "12px"}, children=[
        html.Div([html.Label("Runs to overlay", style={"fontWeight": "bold"}),
                  dcc.Dropdown(id="p_runs", options=[{"label": r, "value": r} for r in RUN_IDS],
                               value=["P3", "J5"], multi=True, style={"width": "420px"})]),
        html.Div([html.Label("Channel", style={"fontWeight": "bold"}),
                  _dropdown(id="p_chan", options=[{"label": c[0], "value": c[0]} for c in TS_PLOT_CHANNELS],
                            value="Fx", style={"width": "160px"})]),
        html.Div([html.Label("Time window (s)", style={"fontWeight": "bold"}),
                  dcc.RangeSlider(id="p_window", min=0, max=20, step=0.5, value=[0, 20],
                                  marks={i: str(i) for i in range(0, 21, 5)},
                                  tooltip={"placement": "bottom"})], style={"width": "320px"}),
    ]),
    dcc.Graph(id="p_curves"),
    dcc.Graph(id="p_err"),
])

# ── Tab 4: real-world recordings ──────────────────────────────────────────────
if RW:
    _rw_recs = RW["recordings"]
    _rw_opts = [{"label": f"{r['config']} — {r['id']}", "value": r["id"]} for r in _rw_recs]
    _rw_default = [r["id"] for r in _rw_recs]
    tab4 = html.Div([
        html.P(RW["story"], style={"color": "#444", "fontSize": "13px", "lineHeight": "1.5"}),
        html.Div([
            html.B("Selection rationale: "), RW["selection_rationale"], html.Br(), html.Br(),
            html.B("Known issues: "), html.Ul([html.Li(x) for x in RW["known_issues"]]),
            html.B("Summary: "),
            html.Span(f"{RW['summary']['n_recordings']} recordings across {RW['summary']['n_configs']} configs; "
                      f"mean of means = {RW['summary']['mean_of_means_n']} N vs GT {RW['ground_truth_n']} N "
                      f"(spread {RW['summary']['spread_of_means_n']} N); mean oscillation std "
                      f"= {RW['summary']['mean_oscillation_std_n']} N; mean MAE vs GT = {RW['summary']['mean_mae_vs_gt_n']} N "
                      f"(sim force MAE for comparison: {RW['summary']['sim_force_mae_n_for_comparison']})."),
        ], style={"background": "#f5f5f5", "padding": "12px", "borderRadius": "6px", "fontSize": "13px",
                  "marginBottom": "12px"}),
        html.Div(style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "marginBottom": "12px"}, children=[
            html.Div([html.Label("Recordings to overlay", style={"fontWeight": "bold"}),
                      dcc.Dropdown(id="rw_recs", options=_rw_opts, value=_rw_default, multi=True,
                                   style={"width": "520px"})]),
            html.Div([html.Label("Trace", style={"fontWeight": "bold"}),
                      _dropdown(id="rw_trace", options=[{"label": "EMA-filtered magnitude", "value": "force_mag_ema"},
                                                        {"label": "Raw magnitude", "value": "force_mag_raw"}],
                                value="force_mag_ema", style={"width": "240px"})]),
        ]),
        dcc.Graph(id="rw_curves"),
        dcc.Graph(id="rw_bars"),
    ])
else:
    tab4 = html.Div([html.P("realworld_recordings_story.json not found — run scripts/build_realworld_recordings_json.py")])

# ── Tab 5: directional spider plot (static-360) ───────────────────────────────
DIR_METRIC_OPTS = [
    ("force_mae", "Force MAE (N)"),
    ("mae", "Overall per-dim MAE"),
    ("per_axis_mae:Fx", "MAE Fx (N)"),
    ("per_axis_mae:Fy", "MAE Fy (N)"),
    ("per_axis_mae:Fz", "MAE Fz (N) — GT Fz=0 here ⇒ standing offset"),
    ("per_axis_mae:τ_yaw", "MAE τ_yaw (Nm) — GT τ_yaw=0 here"),
    ("per_axis_bias:Fx", "Bias Fx (N, signed est−gt) — magnitude under-estimate"),
    ("per_axis_bias:Fy", "Bias Fy (N, signed est−gt)"),
    ("force_noise_std", "Within-trial noise std — force (N, the jitter)"),
    ("per_axis_noise_std:Fx", "Within-trial noise std Fx (N)"),
    ("per_axis_noise_std:Fy", "Within-trial noise std Fy (N)"),
    ("angular_err_xy_deg_mean", "Angular err XY (mean, deg)"),
    ("angular_err_xy_deg_median", "Angular err XY (median, deg)"),
]
# which std field to shade as the ± band for each metric (None ⇒ no sensible band)
def _band_for(metric_key):
    if metric_key in ("force_mae", "mae"):
        return "force_noise_std"          # the jitter you see, magnitude
    if metric_key.startswith("per_axis_mae:") or metric_key.startswith("per_axis_bias:"):
        ax = metric_key.split(":", 1)[1]
        return f"per_axis_noise_std:{ax}"  # within-trial jitter on that axis
    if metric_key == "force_noise_std" or metric_key.startswith("per_axis_noise_std:"):
        return None                        # already a spread metric
    if metric_key.startswith("angular_err_xy_deg_"):
        return metric_key + "_std"         # between-trial spread (reproducibility)
    return None
if DIR_METRICS:
    tab5 = html.Div([
        html.P("Directional spider plot from the static-360 sweep (fixed-magnitude pull, only the "
               "azimuth rotates). Angle = pull direction in the robot's body frame: 0° = toward the "
               "head (+Fx), 90° = left (+Fy), 180° = tail, 270° = right. Radius (`r`) = the chosen "
               "metric. The shaded band is the ± std appropriate for that metric: for MAE / bias it is "
               "the WITHIN-trial estimate noise (the jitter you actually see during a hold); for "
               "noise-std metrics there is no band; for angular error it is the between-trial spread "
               "(reproducibility). Tip: pick 'Bias Fx/Fy (signed)' to see the directional magnitude "
               "under-estimate; pick 'Within-trial noise std' to see the jitter.",
               style={"color": "#555", "fontSize": "13px"}),
        html.Div(style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "marginBottom": "12px"}, children=[
            html.Div([html.Label("Run", style={"fontWeight": "bold"}),
                      _dropdown(id="d_run", options=[{"label": k, "value": k} for k in DIR_METRICS],
                                value=list(DIR_METRICS)[0], style={"width": "220px"})]),
            html.Div([html.Label("Metric", style={"fontWeight": "bold"}),
                      _dropdown(id="d_metric", options=[{"label": l, "value": k} for k, l in DIR_METRIC_OPTS],
                                value="force_mae", style={"width": "420px"})]),
            html.Div([html.Label("Show ± std band", style={"fontWeight": "bold"}),
                      dcc.Checklist(id="d_band", options=[{"label": " yes", "value": "y"}], value=["y"])]),
        ]),
        html.Div(id="d_desc", style={"background": "#f5f5f5", "padding": "10px", "borderRadius": "6px",
                                     "fontSize": "13px", "marginBottom": "10px"}),
        dcc.Graph(id="d_spider"),
    ])
else:
    tab5 = html.Div([html.P("No static-360 direction_metrics.json found — run scripts/rsl_rl/run_static360_r1.sh")])

_tabs = [
    dcc.Tab(label="Sim eval (rollout / OU)", value="t1", children=[tab1]),
    dcc.Tab(label="Static-eval timeseries metrics", value="t2", children=[tab2]),
    dcc.Tab(label="Static-eval timeseries plots", value="t3", children=[tab3]),
    dcc.Tab(label="Real-world recordings", value="t4", children=[tab4]),
    dcc.Tab(label="Directional (static-360)", value="t5", children=[tab5]),
]

app.layout = html.Div(style={"fontFamily": "Liberation Sans, Arial, sans-serif", "maxWidth": "1200px",
                              "margin": "0 auto", "padding": "20px"}, children=[
    html.H2("Force-Estimator Ablation Dashboard"),
    dcc.Tabs(id="tabs", value="t1", children=_tabs),
])


# ── Tab 1 callbacks ───────────────────────────────────────────────────────────
@app.callback(Output("s_desc", "children"), Output("s_bar", "figure"), Output("s_bar_both", "figure"),
              Input("s_group", "value"), Input("s_metric", "value"), Input("s_source", "value"))
def _sim_update(group_name, metric_key, source_key):
    grp = next(g for g in GROUPS if g["name"] == group_name)
    runs, av = grp["runs"], grp.get("axis_values", {})
    label = dict(SIM_METRIC_OPTIONS)[metric_key]
    desc = [html.B("Studies: "), grp["studies"], html.Br(), html.Br(),
            html.B("Baseline: "), grp["baseline"], html.Br(),
            html.B("Runs: "), ", ".join(f"{r} ({av.get(r, r)})" for r in runs)]

    xs, ys, es, txt = [], [], [], []
    for r in runs:
        v, s = _sim_val(SIM_RUNS[r][source_key], metric_key)
        xs.append(f"{r}\n{av.get(r, '')}"); ys.append(v or 0.0); es.append(s or 0.0)
        txt.append(f"{v:.2f}±{s:.2f}" if v is not None and s is not None else (f"{v:.2f}" if v is not None else "n/a"))
    fig = go.Figure(go.Bar(x=xs, y=ys, error_y=dict(type="data", array=es, color="#333", thickness=1.5, width=6),
                           marker_color=[COLORS[i % len(COLORS)] for i in range(len(runs))],
                           text=txt, textposition="outside"))
    fig.update_layout(title=f"{group_name} — {label} — {dict(SIM_SOURCES)[source_key]}",
                      yaxis_title=label, template="plotly_white", height=480, font=dict(size=14),
                      margin=dict(t=60, b=80))

    fig2 = go.Figure()
    for j, (sk, sl) in enumerate(SIM_SOURCES):
        ys2, es2 = [], []
        for r in runs:
            v, s = _sim_val(SIM_RUNS[r][sk], metric_key); ys2.append(v or 0.0); es2.append(s or 0.0)
        fig2.add_trace(go.Bar(name=sl, x=[f"{r} ({av.get(r,'')})" for r in runs], y=ys2,
                              error_y=dict(type="data", array=es2, color="#333", thickness=1.2, width=5),
                              marker_color=COLORS[j]))
    fig2.update_layout(barmode="group", title=f"{group_name} — {label} — rollout vs OU",
                       yaxis_title=label, template="plotly_white", height=460, font=dict(size=14),
                       margin=dict(t=60, b=100))
    return desc, fig, fig2


# ── Tab 2 callbacks ───────────────────────────────────────────────────────────
@app.callback(Output("t_metric", "options"), Output("t_metric", "value"),
              Input("t_scope", "value"), Input("t_metric", "value"))
def _ts_metric_opts(scope, cur):
    opts = TS_AGG_METRICS if scope == "agg" else [(f"chan:{k}", l) for k, l in TS_CHAN_METRICS]
    options = [{"label": l, "value": k} for k, l in opts]
    keys = [k for k, _ in opts]
    val = cur if cur in keys else keys[0]
    return options, val


@app.callback(Output("t_desc", "children"), Output("t_bar", "figure"),
              Input("t_group", "value"), Input("t_scope", "value"), Input("t_metric", "value"))
def _ts_update(group_name, scope, metric_key):
    grp = next(g for g in GROUPS if g["name"] == group_name)
    runs, av = grp["runs"], grp.get("axis_values", {})
    if scope == "agg":
        label = dict(TS_AGG_METRICS).get(metric_key, metric_key); akey = metric_key.split(":", 1)[1]
        getv = lambda r: (TS_RUNS.get(r) or {}).get("aggregate", {}).get(akey)
    else:
        label = f"{scope} — " + dict(TS_CHAN_METRICS).get(metric_key.split(':',1)[1], metric_key)
        ckey = metric_key.split(":", 1)[1]
        getv = lambda r: ((TS_RUNS.get(r) or {}).get("channels", {}).get(scope) or {}).get(ckey)
    desc = [html.B("Studies: "), grp["studies"], html.Br(), html.Br(),
            html.B("Baseline: "), grp["baseline"], html.Br(),
            html.B("Runs: "), ", ".join(f"{r} ({av.get(r, r)})" for r in runs),
            html.Br(), html.I("Single env-0 trajectory — no error bars (cross-env std lives in tab 1).")]
    xs, ys, txt = [], [], []
    for r in runs:
        v = getv(r)
        xs.append(f"{r}\n{av.get(r, '')}"); ys.append(v if v is not None else 0.0)
        txt.append(f"{v:.3f}" if v is not None else "n/a")
    fig = go.Figure(go.Bar(x=xs, y=ys, marker_color=[COLORS[i % len(COLORS)] for i in range(len(runs))],
                           text=txt, textposition="outside"))
    fig.update_layout(title=f"{group_name} — {label}  (static-eval timeseries)",
                      yaxis_title=label, template="plotly_white", height=480, font=dict(size=14),
                      margin=dict(t=60, b=80))
    return desc, fig


# ── Tab 3 callbacks ───────────────────────────────────────────────────────────
@app.callback(Output("p_curves", "figure"), Output("p_err", "figure"),
              Input("p_runs", "value"), Input("p_chan", "value"), Input("p_window", "value"))
def _plot_update(run_ids, chan_label, window):
    spec = next(c for c in TS_PLOT_CHANNELS if c[0] == chan_label)
    _, gk, ek, unit = spec
    run_ids = run_ids or []
    t0, t1 = (window or [0, 20])

    fig = go.Figure()
    fig_e = go.Figure()
    gt_drawn = False
    for i, r in enumerate(run_ids):
        d = _load_raw_ts(r)
        if d is None or gk not in d:
            continue
        import numpy as np
        t = np.asarray(d["time_s"]); m = (t >= t0) & (t <= t1)
        gt = np.asarray(d[gk])[m]; est = np.asarray(d[ek])[m]; tt = t[m]
        if not gt_drawn:
            fig.add_trace(go.Scatter(x=tt, y=gt, mode="lines", line=dict(color="#222", width=2.5, shape="hv"),
                                     name="GT"))
            gt_drawn = True
        c = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(x=tt, y=est, mode="lines", line=dict(color=c, width=1.8), name=f"{r} est"))
        fig_e.add_trace(go.Scatter(x=tt, y=est - gt, mode="lines", line=dict(color=c, width=1.6), name=f"{r}"))
    # mark force changes
    if run_ids:
        d0 = _load_raw_ts(run_ids[0])
        if d0 is not None:
            tarr = np.asarray(d0["time_s"])
            for s in d0.get("rerandom_steps", []):
                si = int(s)
                if si < len(tarr) and t0 <= tarr[si] <= t1:
                    for f in (fig, fig_e):
                        f.add_vline(x=float(tarr[si]), line=dict(color="gray", width=0.6, dash="dash"))
    fig.update_layout(title=f"{chan_label}: GT vs estimate", xaxis_title="Time (s)",
                      yaxis_title=f"{chan_label} ({unit})", template="plotly_white", height=420, font=dict(size=14))
    fig_e.add_hline(y=0, line=dict(color="gray", width=0.6))
    fig_e.update_layout(title=f"{chan_label}: estimation error (est − GT)", xaxis_title="Time (s)",
                        yaxis_title=f"err ({unit})", template="plotly_white", height=340, font=dict(size=14))
    return fig, fig_e


# ── Tab 4 callbacks ───────────────────────────────────────────────────────────
if RW:
    _RW_BY_ID = {r["id"]: r for r in RW["recordings"]}

    @app.callback(Output("rw_curves", "figure"), Output("rw_bars", "figure"),
                  Input("rw_recs", "value"), Input("rw_trace", "value"))
    def _rw_update(rec_ids, trace_key):
        rec_ids = rec_ids or []
        gt = RW["ground_truth_n"]
        fig = go.Figure()
        for i, rid in enumerate(rec_ids):
            r = _RW_BY_ID.get(rid)
            if r is None:
                continue
            c = COLORS[i % len(COLORS)]
            fig.add_trace(go.Scatter(x=r["time_s"], y=r[trace_key], mode="lines",
                                     line=dict(color=c, width=1.6),
                                     name=f"{r['config']} ({r['stats']['mean_mag_ema']:.1f}±{r['stats']['std_mag_ema']:.1f} N)"))
        fig.add_hline(y=gt, line=dict(color="#222", width=2, dash="dash"),
                      annotation_text=f"GT = {gt} N", annotation_position="top left")
        fig.update_layout(title="Real-world static-pull: estimated force magnitude vs ground truth",
                          xaxis_title="Time (s)", yaxis_title="Force magnitude (N)",
                          template="plotly_white", height=460, font=dict(size=14))

        # bar: mean ± std per recording vs GT
        fig2 = go.Figure()
        xs = [f"{_RW_BY_ID[rid]['config']}\n{rid[-8:]}" for rid in rec_ids if rid in _RW_BY_ID]
        means = [_RW_BY_ID[rid]["stats"]["mean_mag_ema"] for rid in rec_ids if rid in _RW_BY_ID]
        stds  = [_RW_BY_ID[rid]["stats"]["std_mag_ema"]  for rid in rec_ids if rid in _RW_BY_ID]
        txt   = [f"{m:.1f}±{s:.1f}" for m, s in zip(means, stds)]
        fig2.add_trace(go.Bar(x=xs, y=means,
                              error_y=dict(type="data", array=stds, color="#333", thickness=1.5, width=6),
                              marker_color=[COLORS[i % len(COLORS)] for i in range(len(means))],
                              text=txt, textposition="outside"))
        fig2.add_hline(y=gt, line=dict(color="#222", width=2, dash="dash"),
                       annotation_text=f"GT = {gt} N", annotation_position="top right")
        fig2.update_layout(title="Mean ± std of estimated magnitude per recording (band = pulley-swing oscillation)",
                           yaxis_title="Force magnitude (N)", template="plotly_white", height=420, font=dict(size=14),
                           margin=dict(t=60, b=80))
        return fig, fig2


# ── Tab 5 callbacks (directional spider) ──────────────────────────────────────
if DIR_METRICS:
    def _get_field(entry_dict, key):
        """key may be 'a' or 'a:Fx'."""
        if ":" in key:
            base, ax = key.split(":", 1)
            return (entry_dict.get(base) or {}).get(ax)
        return entry_dict.get(key)

    @app.callback(Output("d_desc", "children"), Output("d_spider", "figure"),
                  Input("d_run", "value"), Input("d_metric", "value"), Input("d_band", "value"))
    def _spider_update(run_key, metric_key, band):
        if run_key not in DIR_METRICS:
            run_key = next(iter(DIR_METRICS))
        entry = DIR_METRICS[run_key]
        dm = entry["data"]
        dirs = dm["directions"]
        mag_range = dm.get("mag_range", ["?", "?"])
        degs = sorted(float(k) for k in dirs)
        label = dict(DIR_METRIC_OPTS).get(metric_key, metric_key)
        band_key = _band_for(metric_key)

        vals, stds = [], []
        for dg in degs:
            e = dirs[str(dg)]
            v = _get_field(e, metric_key)
            s = _get_field(e, band_key) if band_key else None
            vals.append(v if v is not None else 0.0)
            stds.append(s if s is not None else 0.0)
        signed = metric_key.startswith("per_axis_bias:")  # bias can be negative
        theta_c = degs + [degs[0]]
        vals_c = vals + [vals[0]]
        stds_c = stds + [stds[0]]
        upper = [v + s for v, s in zip(vals_c, stds_c)]
        lower = [(v - s) if signed else max(0.0, v - s) for v, s in zip(vals_c, stds_c)]

        fig = go.Figure()
        if band and "y" in band and band_key:
            fig.add_trace(go.Scatterpolar(r=upper, theta=theta_c, mode="lines",
                                          line=dict(color="rgba(29,53,87,0)"), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatterpolar(r=lower, theta=theta_c, mode="lines", fill="tonext",
                                          fillcolor="rgba(231,111,81,0.22)", line=dict(color="rgba(29,53,87,0)"),
                                          name=f"± {band_key.split(':')[0]}", hoverinfo="skip"))
        if signed:
            fig.add_trace(go.Scatterpolar(r=[0] * len(theta_c), theta=theta_c, mode="lines",
                                          line=dict(color="gray", width=1, dash="dot"),
                                          name="zero (unbiased)", hoverinfo="skip"))
        fig.add_trace(go.Scatterpolar(r=vals_c, theta=theta_c, mode="lines+markers",
                                      line=dict(color="#1d3557", width=2.5), marker=dict(size=6), name=label))
        band_note = (f"  band = ± {band_key.split(':')[0].replace('_', ' ')}"
                     + ("  (within-trial jitter)" if "noise" in (band_key or "") else
                        "  (between-trial spread)" if "std" in (band_key or "") else "")) if band_key else "  (no band)"
        fig.update_layout(
            title=f"{run_key} — {label} vs pull direction  (|F| {mag_range[0]}–{mag_range[1]} N){band_note}",
            template="plotly_white", height=560, font=dict(size=14),
            polar=dict(angularaxis=dict(rotation=0, direction="counterclockwise",
                                        tickmode="array", tickvals=[0, 90, 180, 270],
                                        ticktext=["0° head", "90° left", "180° tail", "270° right"]),
                       radialaxis=dict(angle=90, tickangle=90)),
        )
        n_tr = dirs[str(degs[0])].get("n_trials", "?")
        sort_key = (lambda dg: abs(vals[degs.index(dg)])) if signed else (lambda dg: vals[degs.index(dg)])
        worst = max(degs, key=sort_key); best = min(degs, key=sort_key)
        desc = [html.B("Source: "), entry["folder"], f"  ({n_tr} trials/direction, elevation {dm.get('elevation', 0)}°)",
                html.Br(),
                html.B("Best (smallest): "), f"{best:.0f}°  ({vals[degs.index(best)]:+.2f})    ",
                html.B("Worst (largest): "), f"{worst:.0f}°  ({vals[degs.index(worst)]:+.2f})",
                html.Br(), html.I("0°=toward head (+Fx) · 90°=left (+Fy) · 180°=toward tail (−Fx) · 270°=right (−Fy).")]
        return desc, fig


if __name__ == "__main__":
    print(f"sim metrics: {_args.sim_metrics}")
    print(f"ts  metrics: {_args.ts_metrics}  ({'found' if TS else 'MISSING'})")
    print(f"realworld  : {_args.realworld}  ({'found' if RW else 'MISSING'})")
    print(f"direction  : {len(DIR_METRICS)} static-360 metric file(s) found")
    print(f"Open http://{_args.host}:{_args.port}")
    app.run(debug=False, host=_args.host, port=_args.port)
