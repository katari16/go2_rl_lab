"""Rollout estimator comparison dashboard — localhost:6007

Usage:
    python scripts/rsl_rl/dashboard.py
    # then open http://localhost:6007
"""

import base64
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback_context, dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Load data ─────────────────────────────────────────────────────────────────

MASTER_PATH = Path("data/eval/master_metrics.json")
with open(MASTER_PATH) as f:
    MASTER = json.load(f)

ALL_RUN_IDS = list(MASTER.keys())   # preserves insertion order


# ── Colors ────────────────────────────────────────────────────────────────────

_P_LIST = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,
           23,24,25,26,27,28,29,30,31,32,33,34,35]

def _p_hex(run_id: str) -> str:
    import re
    m = re.match(r"[A-Za-z]+(\d+)", run_id)
    p_num = int(m.group(1)) if m else 0
    idx   = _P_LIST.index(p_num) if p_num in _P_LIST else 0
    t     = idx / max(len(_P_LIST) - 1, 1)
    r, g, b, _ = plt.cm.Blues(0.35 + 0.55 * t)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

SERIES_COLOR = {"A": "#4472C4", "B": "#70AD47", "J": "#FF7043", "Q": "#AB47BC"}

def run_color(rid: str) -> str:
    if rid[0] in SERIES_COLOR:
        return SERIES_COLOR[rid[0]]
    return _p_hex(rid)

def run_series(rid: str) -> str:
    return rid[0] if rid[0] in ("A", "B", "J", "Q") else "P"


# ── Metric extraction ─────────────────────────────────────────────────────────

def extract(rid: str) -> dict:
    run    = MASTER[rid]
    basket = run["metrics"].get("baskets", {}).get("training", {})
    tr     = run["metrics"].get("training_regime", {})
    pam    = basket.get("per_axis_mae", {})
    max_f  = tr.get("max_force", 30.0)
    fmae   = basket.get("force_mae")
    return {
        "Fx":             pam.get("Fx"),
        "Fy":             pam.get("Fy"),
        "Fz":             pam.get("Fz"),
        "tau_yaw":        pam.get("\u03c4_yaw"),
        "force_mae":      fmae,
        "range_norm_pct": (fmae / max_f * 100) if fmae else None,
        "ang_err_median": basket.get("angular_err_xy_deg_median"),
        "rel_err_pct":    basket.get("relative_err_no_mask_pct"),
        "max_force":      max_f,
        "description":    run.get("description", ""),
    }

METRICS = {rid: extract(rid) for rid in ALL_RUN_IDS}

PANELS = [
    ("Fx",             "Fx MAE",               "N"),
    ("Fy",             "Fy MAE",               "N"),
    ("Fz",             "Fz MAE",               "N"),
    ("tau_yaw",        "τ_yaw MAE",            "Nm"),
    ("force_mae",      "Force MAE (total)",    "N"),
    ("range_norm_pct", "Range-Norm MAE",       "%"),
    ("ang_err_median", "Median Angular Error", "°"),
    ("rel_err_pct",    "Relative Error",       "%"),
]


# ── Image helpers ─────────────────────────────────────────────────────────────

def _checkpoint_dir(rid: str) -> Path | None:
    run  = MASTER[rid]
    cfg_path = (Path("data/eval")
                / run["experiment_folder"]
                / run["rollout_eval_dir"]
                / "config.json")
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        cfg = json.load(f)
    cp = cfg.get("checkpoint")
    return Path(cp).parent if cp else None


def _latest_png(directory: Path) -> Path | None:
    if not directory or not directory.exists():
        return None
    imgs = sorted(p for p in directory.glob("*.png") if "_metrics" not in p.name)
    return imgs[-1] if imgs else None


def _img_src(path: Path) -> str:
    with open(path, "rb") as f:
        enc = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{enc}"


# ── Figure builder ────────────────────────────────────────────────────────────

def build_figure(selected: list[str]) -> go.Figure:
    n = len(selected)
    row_h = max(320, n * 22 + 60)

    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=[p[1] for p in PANELS],
        horizontal_spacing=0.07,
        vertical_spacing=0.14,
    )

    positions = [(1,1),(1,2),(1,3),(1,4),(2,1),(2,2),(2,3),(2,4)]

    for (key, title, unit), (row, col) in zip(PANELS, positions):
        xs, ys, clrs, hover = [], [], [], []
        for rid in selected:
            m   = METRICS[rid]
            val = m[key]
            xs.append(val if val is not None else 0.0)
            ys.append(rid)
            clrs.append(run_color(rid) if val is not None else "#e0e0e0")
            hover.append(
                f"<b>{rid}</b><br>{m['description']}<br>"
                + (f"{key}: {val:.3f} {unit}" if val is not None else "Not estimated")
            )

        fig.add_trace(
            go.Bar(
                x=xs, y=ys,
                orientation="h",
                marker_color=clrs,
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover,
                showlegend=False,
                text=[f"{v:.2f}" if v else "N/A" for v in
                      [METRICS[r][key] for r in selected]],
                textposition="outside",
                textfont=dict(size=9),
                cliponaxis=False,
            ),
            row=row, col=col,
        )
        fig.update_yaxes(autorange="reversed", row=row, col=col,
                         tickfont=dict(size=10))
        fig.update_xaxes(title_text=unit, title_font=dict(size=10),
                         row=row, col=col)

    fig.update_layout(
        height=row_h * 2 + 80,
        margin=dict(l=20, r=30, t=60, b=20),
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=11),
    )
    return fig


# ── App layout ────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Rollout Eval Dashboard",
)

# sidebar: series toggle buttons + per-run checklist
def _series_block(series: str, run_ids: list[str], color: str,
                  initial_values: list[str] | None = None) -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col(html.Span(
                f"● {series}-series",
                style={"fontWeight": "700", "color": color, "fontSize": "13px"},
            ), width=7),
            dbc.Col(dbc.Button(
                "All", id=f"btn-{series}", size="sm", n_clicks=0,
                color="light",
                style={"fontSize": "11px", "padding": "1px 8px"},
            ), width=5, className="text-end"),
        ], className="align-items-center mb-1"),
        dcc.Checklist(
            id=f"chk-{series}",
            options=[
                {"label": html.Span(
                    rid,
                    style={"background": run_color(rid),
                           "color": "white",
                           "borderRadius": "4px",
                           "padding": "1px 6px",
                           "fontSize": "12px",
                           "fontWeight": "600",
                           "marginLeft": "4px"},
                ), "value": rid}
                for rid in run_ids
            ],
            value=initial_values if initial_values is not None else run_ids,
            labelStyle={"display": "flex", "alignItems": "center",
                        "marginBottom": "3px"},
            inputStyle={"marginRight": "4px"},
        ),
    ], style={"marginBottom": "16px"})


series_groups = {s: [] for s in ("A", "B", "J", "P", "Q")}
for rid in ALL_RUN_IDS:
    series_groups[run_series(rid)].append(rid)

P23Q2_RUNS = {r for r in ALL_RUN_IDS
              if (r.startswith("P") and len(r) > 1 and
                  any(r.startswith(f"P{n}") for n in
                      [23,24,25,26,27,28,29,30,31,"32a","32b","32c","32d",33,34,35]))
              or r.startswith("Q")}

sidebar = html.Div([
    html.H6("Force range", style={"fontWeight": "700", "marginBottom": "6px"}),
    dcc.RadioItems(
        id="force-filter",
        options=[
            {"label": " All", "value": "all"},
            {"label": " 20 N", "value": "20"},
            {"label": " 30 N", "value": "30"},
        ],
        value="30",
        labelStyle={"display": "block", "marginBottom": "4px", "fontSize": "13px"},
        inputStyle={"marginRight": "6px"},
    ),
    html.Hr(style={"margin": "12px 0"}),
    dbc.Row([
        dbc.Col(dbc.Button("Select all", id="btn-all", size="sm", n_clicks=0,
                           color="secondary", outline=True,
                           style={"fontSize": "11px"}), width=6),
        dbc.Col(dbc.Button("Clear all",  id="btn-none", size="sm", n_clicks=0,
                           color="secondary", outline=True,
                           style={"fontSize": "11px"}), width=6),
    ], className="mb-2"),
    dbc.Button("Toggle P23–Q2", id="btn-p23q2", size="sm", n_clicks=0,
               color="warning", outline=True,
               style={"fontSize": "11px", "width": "100%", "marginBottom": "12px"}),
    _series_block("A", series_groups["A"], SERIES_COLOR["A"]),
    _series_block("B", series_groups["B"], SERIES_COLOR["B"]),
    _series_block("J", series_groups["J"], SERIES_COLOR["J"]),
    _series_block("P", series_groups["P"], "#4682B4",
                  initial_values=[r for r in series_groups["P"] if r not in P23Q2_RUNS]),
    _series_block("Q", series_groups["Q"], SERIES_COLOR["Q"],
                  initial_values=[]),
], style={
    "width": "200px", "minWidth": "200px",
    "padding": "16px 12px",
    "background": "#f8f9fa",
    "borderRight": "1px solid #dee2e6",
    "overflowY": "auto",
    "height": "calc(100vh - 56px)",
})

app.layout = html.Div([
    # ── Header ──────────────────────────────────────────────────────────────
    html.Div([
        html.Span("🔬", style={"fontSize": "20px", "marginRight": "8px"}),
        html.Span("Rollout Estimator Comparison",
                  style={"fontSize": "18px", "fontWeight": "700",
                         "color": "white"}),
        html.Span(" — training regime, 128 envs, 10 s",
                  style={"fontSize": "13px", "color": "#cce", "marginLeft": "8px"}),
    ], style={
        "background": "#2c3e50", "padding": "10px 20px",
        "display": "flex", "alignItems": "center",
        "height": "56px",
    }),

    # ── Body ────────────────────────────────────────────────────────────────
    html.Div([
        sidebar,
        html.Div([
            dcc.Graph(
                id="main-chart",
                config={"displayModeBar": False},
                style={"width": "100%"},
            ),
            html.Div(id="image-panel", style={
                "padding": "16px 20px",
                "borderTop": "1px solid #dee2e6",
                "background": "#fff",
            }),
            html.Div([
                dbc.Button(
                    "▶ Run descriptions",
                    id="btn-desc",
                    size="sm",
                    color="secondary",
                    outline=True,
                    n_clicks=0,
                    style={"fontSize": "12px", "margin": "8px 20px"},
                ),
                dbc.Collapse(
                    html.Div(id="desc-panel", style={
                        "padding": "0 20px 16px",
                        "background": "#fafafa",
                        "borderTop": "1px solid #eee",
                    }),
                    id="desc-collapse",
                    is_open=False,
                ),
            ]),
        ], style={"flex": "1", "overflowY": "auto", "overflowX": "hidden"}),
    ], style={"display": "flex", "height": "calc(100vh - 56px)"}),
])


# ── Callbacks ─────────────────────────────────────────────────────────────────

def _merge_checklists(a, b, j, p, q=None):
    return (a or []) + (b or []) + (j or []) + (p or []) + (q or [])


@app.callback(
    Output("chk-A",  "value"),
    Output("chk-B",  "value"),
    Output("chk-J",  "value"),
    Output("chk-P",  "value"),
    Output("chk-Q",  "value"),
    Input("btn-all",  "n_clicks"),
    Input("btn-none", "n_clicks"),
    Input("btn-A",      "n_clicks"),
    Input("btn-B",      "n_clicks"),
    Input("btn-J",      "n_clicks"),
    Input("btn-P",      "n_clicks"),
    Input("btn-Q",      "n_clicks"),
    Input("btn-p23q2",  "n_clicks"),
    State("chk-A", "value"), State("chk-B", "value"),
    State("chk-J", "value"), State("chk-P", "value"),
    State("chk-Q", "value"),
    prevent_initial_call=True,
)
def toggle_series(n_all, n_none, nA, nB, nJ, nP, nQ, n23q2, cA, cB, cJ, cP, cQ):
    triggered = callback_context.triggered[0]["prop_id"].split(".")[0]
    if triggered == "btn-all":
        return (series_groups["A"], series_groups["B"],
                series_groups["J"], series_groups["P"], series_groups["Q"])
    if triggered == "btn-none":
        return [], [], [], [], []
    if triggered == "btn-p23q2":
        cur_p = set(cP or [])
        cur_q = set(cQ or [])
        # if all P23-Q2 runs are already hidden, show them; otherwise hide them
        p23q2_p = P23Q2_RUNS & set(series_groups["P"])
        p23q2_q = P23Q2_RUNS & set(series_groups["Q"])
        all_hidden = not (p23q2_p & cur_p) and not (p23q2_q & cur_q)
        if all_hidden:
            new_p = list(cur_p | p23q2_p)
            new_q = list(cur_q | p23q2_q)
        else:
            new_p = list(cur_p - p23q2_p)
            new_q = list(cur_q - p23q2_q)
        return list(cA or []), list(cB or []), list(cJ or []), new_p, new_q

    series_map = {"btn-A": "A", "btn-B": "B", "btn-J": "J", "btn-P": "P", "btn-Q": "Q"}
    chk_map    = {"A": cA, "B": cB, "J": cJ, "P": cP, "Q": cQ}
    target     = series_map.get(triggered)

    result = {"A": list(cA or []), "B": list(cB or []),
              "J": list(cJ or []), "P": list(cP or []), "Q": list(cQ or [])}
    if target:
        all_s = series_groups[target]
        if set(all_s).issubset(set(chk_map[target] or [])):
            result[target] = []
        else:
            result[target] = all_s

    return result["A"], result["B"], result["J"], result["P"], result["Q"]


@app.callback(
    Output("main-chart", "figure"),
    Input("chk-A",       "value"),
    Input("chk-B",       "value"),
    Input("chk-J",       "value"),
    Input("chk-P",       "value"),
    Input("chk-Q",       "value"),
    Input("force-filter","value"),
)
def update_chart(cA, cB, cJ, cP, cQ, force_filter):
    selected = _merge_checklists(cA, cB, cJ, cP, cQ)
    # preserve master order
    selected = [r for r in ALL_RUN_IDS if r in set(selected)]

    if force_filter == "20":
        selected = [r for r in selected if METRICS[r]["max_force"] == 20.0]
    elif force_filter == "30":
        selected = [r for r in selected if METRICS[r]["max_force"] == 30.0]

    if not selected:
        fig = go.Figure()
        fig.update_layout(
            height=200,
            annotations=[dict(text="No runs selected", showarrow=False,
                              font=dict(size=16, color="#888"),
                              xref="paper", yref="paper", x=0.5, y=0.5)],
        )
        return fig

    return build_figure(selected)


@app.callback(
    Output("image-panel", "children"),
    Input("main-chart",   "clickData"),
)
def show_images(click_data):
    if not click_data:
        return html.P(
            "↑ Click any bar to see the static eval and OU force eval images for that run.",
            style={"color": "#888", "fontStyle": "italic", "margin": "0"},
        )

    rid = click_data["points"][0]["y"]
    m   = METRICS[rid]
    cp_dir = _checkpoint_dir(rid)

    static_path = _latest_png(cp_dir / "force_eval")   if cp_dir else None
    ou_path     = _latest_png(cp_dir / "ou_force_eval") if cp_dir else None

    def _img_card(title: str, path: Path | None) -> dbc.Col:
        if path:
            return dbc.Col([
                html.P(title, style={"fontWeight": "600", "marginBottom": "6px",
                                     "fontSize": "13px", "color": "#555"}),
                html.Img(src=_img_src(path),
                         style={"maxWidth": "100%", "borderRadius": "6px",
                                "border": "1px solid #ddd"}),
                html.P(str(path.name),
                       style={"fontSize": "10px", "color": "#aaa", "marginTop": "4px"}),
            ])
        else:
            return dbc.Col(
                html.P(f"{title}: no image found", style={"color": "#bbb"}),
            )

    return html.Div([
        dbc.Row([
            dbc.Col(html.Div([
                html.Span(rid, style={
                    "background": run_color(rid), "color": "white",
                    "borderRadius": "5px", "padding": "3px 10px",
                    "fontWeight": "700", "fontSize": "15px", "marginRight": "10px",
                }),
                html.Span(m["description"],
                          style={"fontSize": "13px", "color": "#444"}),
            ]), width=12, className="mb-3"),
        ]),
        dbc.Row([
            _img_card("Static force eval (latest)", static_path),
            _img_card("OU force eval (latest)",     ou_path),
        ]),
    ])


@app.callback(
    Output("desc-collapse", "is_open"),
    Output("btn-desc",      "children"),
    Input("btn-desc", "n_clicks"),
    State("desc-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_desc(n, is_open):
    new_open = not is_open
    label = "▼ Run descriptions" if new_open else "▶ Run descriptions"
    return new_open, label


@app.callback(
    Output("desc-panel", "children"),
    Input("desc-collapse", "is_open"),
    State("chk-A", "value"), State("chk-B", "value"),
    State("chk-J", "value"), State("chk-P", "value"),
    State("chk-Q", "value"),
)
def populate_desc(is_open, cA, cB, cJ, cP, cQ):
    if not is_open:
        return []
    selected = set(_merge_checklists(cA, cB, cJ, cP, cQ))
    rows = []
    for rid in ALL_RUN_IDS:
        m = METRICS[rid]
        checked = rid in selected
        color = run_color(rid)
        rows.append(html.Tr([
            html.Td(
                dcc.Checklist(
                    id={"type": "desc-chk", "rid": rid},
                    options=[{"label": "", "value": rid}],
                    value=[rid] if checked else [],
                    style={"display": "inline"},
                    inputStyle={"marginRight": "4px"},
                ),
                style={"width": "30px", "verticalAlign": "middle"},
            ),
            html.Td(
                html.Span(rid, style={
                    "background": color, "color": "white",
                    "borderRadius": "4px", "padding": "2px 7px",
                    "fontWeight": "700", "fontSize": "12px",
                }),
                style={"width": "60px", "verticalAlign": "middle", "paddingRight": "8px"},
            ),
            html.Td(
                m["description"],
                style={"fontSize": "12px", "color": "#444", "verticalAlign": "middle"},
            ),
        ], style={"borderBottom": "1px solid #eee"}))

    return html.Div([
        html.P("Check/uncheck runs to show or hide them in the chart.",
               style={"fontSize": "12px", "color": "#888", "margin": "10px 0 6px"}),
        html.Table(rows, style={"width": "100%", "borderCollapse": "collapse"}),
    ])


@app.callback(
    Output("chk-A",  "value", allow_duplicate=True),
    Output("chk-B",  "value", allow_duplicate=True),
    Output("chk-J",  "value", allow_duplicate=True),
    Output("chk-P",  "value", allow_duplicate=True),
    Output("chk-Q",  "value", allow_duplicate=True),
    Input({"type": "desc-chk", "rid": dash.ALL}, "value"),
    State("chk-A", "value"), State("chk-B", "value"),
    State("chk-J", "value"), State("chk-P", "value"),
    State("chk-Q", "value"),
    prevent_initial_call=True,
)
def desc_toggle_run(all_vals, cA, cB, cJ, cP, cQ):
    ctx = callback_context.triggered[0]["prop_id"]
    import json as _json
    rid = _json.loads(ctx.split(".")[0])["rid"]
    checked = bool(all_vals[0]) if all_vals else False
    series = run_series(rid)
    buckets = {"A": list(cA or []), "B": list(cB or []),
               "J": list(cJ or []), "P": list(cP or []), "Q": list(cQ or [])}
    if checked and rid not in buckets[series]:
        buckets[series].append(rid)
    elif not checked and rid in buckets[series]:
        buckets[series].remove(rid)
    return buckets["A"], buckets["B"], buckets["J"], buckets["P"], buckets["Q"]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=6007)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    print(f"\n  Dashboard: http://localhost:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False)
