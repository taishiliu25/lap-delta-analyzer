"""Plotly figures for the dashboard: synced traces, delta, sectors and track maps.

All figures take the ``delta_df`` produced by :func:`lap_delta.align.compute_delta`
(channels suffixed ``_cmp`` / ``_ref``) plus the two driver codes and an ``imperial`` flag.
The delta convention is fixed and stated in every label: ``delta = comparison - reference``,
so a positive/rising delta means the *comparison* driver is slower / losing time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import units

REF_COLOR = "#2E8BFF"   # blue  (reference)
CMP_COLOR = "#E10600"   # F1 red (comparison)
GAIN_COLOR = "#22C55E"  # green (comparison gaining time)
LOSS_COLOR = "#E10600"  # red   (comparison losing time)

# --- shared "pit-wall" dark theme -------------------------------------------------
# Applied to every figure so the FastF1 and iRacing pages are visually identical; the
# canvas is transparent so figures sit on the Streamlit panel, and only the neutral
# chrome is themed (driver/team line colors keep their meaning).
_PAPER_BG = "rgba(0,0,0,0)"
_GRID = "#222834"
_AXIS = "#39414F"
_TEXT = "#D7DCE3"
_MUTED = "#8A93A3"
_HOVER_BG = "#141821"
_FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
_SF_LIGHT = "#E7EBF0"   # start/finish + corner labels, light for dark background


def _lighten(hex_color: str, amount: float) -> str:
    """Blend a hex color toward white by ``amount`` (0..1); used to split teammate colors."""
    try:
        h = hex_color.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    except (ValueError, IndexError):
        return hex_color
    r, g, b = (round(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02X}{g:02X}{b:02X}"


def _colors(ref_color: str | None, cmp_color: str | None) -> tuple[str, str]:
    """Resolve ref/cmp line colors.

    When both laps carry the *same* color (teammates on one team color), the comparison is
    lightened to a brighter shade of the same color so the two still read apart at a glance.
    """
    rc = ref_color or REF_COLOR
    cc = cmp_color or CMP_COLOR
    if rc.lower() == cc.lower():
        cc = _lighten(cc, 0.5)
    return rc, cc


def _apply_theme(fig: go.Figure) -> go.Figure:
    """Restyle a figure onto the shared dark theme (transparent canvas, subtle grid)."""
    fig.update_layout(
        paper_bgcolor=_PAPER_BG, plot_bgcolor=_PAPER_BG,
        font=dict(family=_FONT, color=_TEXT, size=12),
        title_font=dict(family=_FONT, color=_TEXT, size=15),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_MUTED, size=11)),
        hoverlabel=dict(bgcolor=_HOVER_BG, bordercolor=_AXIS,
                        font=dict(color=_TEXT, family=_FONT, size=12)),
    )
    axis = dict(gridcolor=_GRID, zerolinecolor=_AXIS, linecolor=_AXIS,
                tickfont=dict(color=_MUTED, size=11), title_font=dict(color=_MUTED, size=12))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    for ann in fig.layout.annotations or ():   # subplot titles + inline labels -> light
        ann.font.color = _TEXT
    return fig


def traces_figure(
    delta_df: pd.DataFrame,
    ref_label: str = "Reference",
    cmp_label: str = "Comparison",
    *,
    imperial: bool = False,
    ref_color: str | None = None,
    cmp_color: str | None = None,
) -> go.Figure:
    """Stacked, distance-aligned Speed / Throttle / Brake / Delta traces (shared x)."""
    rc, cc = _colors(ref_color, cmp_color)
    d = units.distance(delta_df["distance_m"].to_numpy(), imperial)
    du, su = units.distance_unit(imperial), units.speed_unit(imperial)

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.34, 0.19, 0.19, 0.28],
        subplot_titles=(f"Speed ({su})", "Throttle (%)", "Brake (%)",
                        f"Delta (s) — {cmp_label.split()[0]} minus {ref_label.split()[0]}"),
    )

    fig.add_trace(go.Scatter(x=d, y=units.speed(delta_df["speed_kph_ref"], imperial),
                             name=ref_label, line=dict(color=rc)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d, y=units.speed(delta_df["speed_kph_cmp"], imperial),
                             name=cmp_label, line=dict(color=cc)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d, y=delta_df["throttle_ref"] * 100, name=ref_label,
                             line=dict(color=rc), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=d, y=delta_df["throttle_cmp"] * 100, name=cmp_label,
                             line=dict(color=cc), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=d, y=delta_df["brake_ref"] * 100, name=ref_label,
                             line=dict(color=rc), showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=d, y=delta_df["brake_cmp"] * 100, name=cmp_label,
                             line=dict(color=cc), showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=d, y=delta_df["delta_s"].to_numpy(), name="delta",
                             fill="tozeroy", line=dict(color=_MUTED),
                             fillcolor="rgba(138,147,163,0.15)",
                             showlegend=False), row=4, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color=_AXIS, row=4, col=1)

    fig.update_xaxes(title_text=f"Distance ({du})", row=4, col=1)
    fig.update_layout(
        height=760, hovermode="x unified", margin=dict(t=40, b=40, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return _apply_theme(fig)


def delta_gain_loss_figure(
    delta_df: pd.DataFrame,
    ref_code: str = "REF",
    cmp_code: str = "CMP",
    *,
    imperial: bool = False,
) -> go.Figure:
    """Cumulative delta shaded green where the comparison gains and red where it loses.

    ``delta = cmp - ref``; the local slope tells the story: falling ⇒ ``cmp`` is gaining
    (green), rising ⇒ ``cmp`` is losing (red). Makes "where the driver did well" explicit.
    """
    d = units.distance(delta_df["distance_m"].to_numpy(), imperial)
    du = units.distance_unit(imperial)
    delta = delta_df["delta_s"].to_numpy(dtype=float)
    if len(delta) > 1 and np.ptp(d) > 0:
        slope = np.gradient(delta, d)
    else:
        slope = np.zeros_like(delta)

    gaining = np.where(slope < 0, delta, np.nan)  # cmp catching up
    losing = np.where(slope >= 0, delta, np.nan)  # cmp dropping back

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d, y=delta, line=dict(color=_MUTED, width=1),
                             name="delta", hovertemplate="%{y:.3f} s<extra></extra>"))
    fig.add_trace(go.Scatter(x=d, y=losing, line=dict(color=LOSS_COLOR, width=3),
                             name=f"{cmp_code} losing", connectgaps=False))
    fig.add_trace(go.Scatter(x=d, y=gaining, line=dict(color=GAIN_COLOR, width=3),
                             name=f"{cmp_code} gaining", connectgaps=False))
    fig.add_hline(y=0, line_dash="dot", line_color=_AXIS)
    fig.update_layout(
        title=f"Time delta — {cmp_code} minus {ref_code} "
              f"(below 0 → {cmp_code} ahead; green → {cmp_code} gaining)",
        height=340, hovermode="x unified", margin=dict(t=50, b=40, l=60, r=20),
        xaxis_title=f"Distance ({du})", yaxis_title="Delta (s)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return _apply_theme(fig)


def gear_rpm_figure(
    delta_df: pd.DataFrame,
    ref_label: str = "Reference",
    cmp_label: str = "Comparison",
    *,
    ref_color: str | None = None,
    cmp_color: str | None = None,
    imperial: bool = False,
) -> go.Figure:
    """Gear and RPM vs distance for both laps (shared x)."""
    rc, cc = _colors(ref_color, cmp_color)
    d = units.distance(delta_df["distance_m"].to_numpy(), imperial)
    du = units.distance_unit(imperial)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.5, 0.5], subplot_titles=("Gear", "RPM"))
    if "gear_ref" in delta_df:
        fig.add_trace(go.Scatter(x=d, y=delta_df["gear_ref"], name=ref_label,
                                 line=dict(color=rc, shape="hv")), row=1, col=1)
        fig.add_trace(go.Scatter(x=d, y=delta_df["gear_cmp"], name=cmp_label,
                                 line=dict(color=cc, shape="hv")), row=1, col=1)
    if "rpm_ref" in delta_df:
        fig.add_trace(go.Scatter(x=d, y=delta_df["rpm_ref"], name=ref_label,
                                 line=dict(color=rc), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=d, y=delta_df["rpm_cmp"], name=cmp_label,
                                 line=dict(color=cc), showlegend=False), row=2, col=1)
    fig.update_xaxes(title_text=f"Distance ({du})", row=2, col=1)
    fig.update_layout(height=480, hovermode="x unified", margin=dict(t=40, b=40, l=60, r=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="right", x=1))
    return _apply_theme(fig)




def _has_position(delta_df: pd.DataFrame) -> bool:
    x = delta_df.get("x_cmp")
    return x is not None and not np.isnan(np.asarray(x, dtype=float)).all()


def _interp_xy(delta_df: pd.DataFrame, distances) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the plotted lap's x/y at the given distances (meters on the ref grid)."""
    d = delta_df["distance_m"].to_numpy(dtype=float)
    x = np.asarray(delta_df["x_cmp"], dtype=float)
    y = np.asarray(delta_df["y_cmp"], dtype=float)
    dd = np.asarray(distances, dtype=float)
    return np.interp(dd, d, x), np.interp(dd, d, y)


def _start_finish_trace(delta_df: pd.DataFrame) -> go.Scatter:
    d = delta_df["distance_m"].to_numpy(dtype=float)
    xs, ys = _interp_xy(delta_df, [float(d.min())])
    return go.Scatter(
        x=xs, y=ys, mode="markers+text", text=["S/F"], textposition="bottom center",
        marker=dict(size=12, symbol="square", color=_SF_LIGHT,
                    line=dict(width=1, color="#0B0E14")),
        textfont=dict(size=12, color=_SF_LIGHT), name="Start/Finish",
        hovertemplate="Start/Finish<extra></extra>",
    )


def _corner_trace(delta_df: pd.DataFrame, corners) -> go.Scatter:
    """corners: list of (label, distance_m)."""
    labels = [str(lbl) for lbl, _ in corners]
    dists = [float(dd) for _, dd in corners]
    xs, ys = _interp_xy(delta_df, dists)
    return go.Scatter(
        x=xs, y=ys, mode="markers+text", text=labels, textposition="top center",
        marker=dict(size=7, symbol="circle-open", color=_MUTED),
        textfont=dict(size=11, color=_MUTED), name="Corner",
        hovertemplate="Turn %{text}<extra></extra>",
    )


def track_map_figure(delta_df: pd.DataFrame, *, imperial: bool = False,
                     corners=None) -> go.Figure:
    """Track map (comparison-lap x/y) colored by cumulative delta; equal aspect.

    When ``corners`` (list of ``(label, distance_m)``) is given, official turn numbers and the
    start/finish line are overlaid.
    """
    fig = go.Figure()
    if not _has_position(delta_df):
        fig.update_layout(title="Track map unavailable (no position data)")
        return _apply_theme(fig)

    fig.add_trace(go.Scatter(
        x=np.asarray(delta_df["x_cmp"], dtype=float),
        y=np.asarray(delta_df["y_cmp"], dtype=float),
        mode="markers", name="delta",
        marker=dict(
            size=6,
            color=delta_df["delta_s"],
            colorscale="RdYlGn_r",  # green = comparison gaining, red = losing
            cmid=0,
            colorbar=dict(title=dict(text="Δ (s)", font=dict(color=_MUTED, size=12)),
                          tickfont=dict(color=_MUTED, size=10), outlinewidth=0,
                          thickness=12, len=0.8),
        ),
        hovertemplate="delta: %{marker.color:.3f} s<extra></extra>",
    ))
    if corners:
        fig.add_trace(_corner_trace(delta_df, corners))
    fig.add_trace(_start_finish_trace(delta_df))

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title="Where time is won / lost", height=560, showlegend=False,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return _apply_theme(fig)


def _gate_trace(delta_df: pd.DataFrame, boundary_dists) -> go.Scatter:
    """Short perpendicular 'gate' lines across the track at each boundary distance.

    These slice the dominance map into its sectors so the start/stop of each is visible.
    """
    d = delta_df["distance_m"].to_numpy(dtype=float)
    x = np.asarray(delta_df["x_cmp"], dtype=float)
    y = np.asarray(delta_df["y_cmp"], dtype=float)
    span = max(float(np.ptp(x)), float(np.ptp(y))) or 1.0
    half = 0.022 * span
    eps = max((d[-1] - d[0]) * 0.002, 1e-6)

    xs: list[float | None] = []
    ys: list[float | None] = []
    for bd in boundary_dists:
        x0, y0 = float(np.interp(bd, d, x)), float(np.interp(bd, d, y))
        x1, y1 = float(np.interp(bd + eps, d, x)), float(np.interp(bd + eps, d, y))
        dx, dy = x1 - x0, y1 - y0
        norm = (dx * dx + dy * dy) ** 0.5 or 1.0
        px, py = -dy / norm, dx / norm  # unit perpendicular to the local track direction
        xs += [x0 - px * half, x0 + px * half, None]
        ys += [y0 - py * half, y0 + py * half, None]
    return go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=_SF_LIGHT, width=1.2),
                      name="boundaries", hoverinfo="skip", showlegend=False)


def _dominance_map(
    delta_df: pd.DataFrame,
    seg_df: pd.DataFrame,
    ref_code: str,
    cmp_code: str,
    *,
    rc: str,
    cc: str,
    title: str,
    number_name: str,
) -> go.Figure:
    """Track map colored by which driver is faster in each segment, sliced at the boundaries.

    ``seg_df`` needs ``dist_start`` / ``dist_end`` / ``faster`` (``"REF"``/``"CMP"``) plus a
    ``label`` column used to number each segment at its midpoint. Shared by the mini-sector
    and official-sector dominance maps so they look identical.
    """
    fig = go.Figure()
    if not _has_position(delta_df):
        fig.update_layout(title=f"{title} — unavailable (no position data)")
        return _apply_theme(fig)

    dist = delta_df["distance_m"].to_numpy(dtype=float)
    x = np.asarray(delta_df["x_cmp"], dtype=float)
    y = np.asarray(delta_df["y_cmp"], dtype=float)

    # Assign each telemetry point to a segment, then split into faster-driver groups.
    edges = np.concatenate([seg_df["dist_start"].to_numpy(),
                            seg_df["dist_end"].to_numpy()[-1:]])
    idx = np.clip(np.searchsorted(edges, dist, side="right") - 1, 0, len(seg_df) - 1)
    faster = seg_df["faster"].to_numpy()[idx]

    for who, color, code in (("CMP", cc, cmp_code), ("REF", rc, ref_code)):
        mask = faster == who
        if mask.any():
            fig.add_trace(go.Scatter(
                x=x[mask], y=y[mask], mode="markers", name=f"{code} faster",
                marker=dict(size=6, color=color),
                hovertemplate=f"{code} faster<extra></extra>",
            ))

    fig.add_trace(_gate_trace(delta_df, edges[1:-1]))  # internal boundaries (S/F drawn below)

    mids = 0.5 * (seg_df["dist_start"].to_numpy() + seg_df["dist_end"].to_numpy())
    mx, my = _interp_xy(delta_df, mids)
    fig.add_trace(go.Scatter(
        x=mx, y=my, mode="text", text=[str(v) for v in seg_df["label"]],
        textposition="middle center", textfont=dict(size=11, color=_SF_LIGHT),
        name=number_name, hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(_start_finish_trace(delta_df))

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=title, height=560,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
    )
    return _apply_theme(fig)


def minisector_dominance_map(
    delta_df: pd.DataFrame,
    mini_df: pd.DataFrame,
    ref_code: str = "REF",
    cmp_code: str = "CMP",
    *,
    ref_color: str | None = None,
    cmp_color: str | None = None,
) -> go.Figure:
    """Dominance map over the ~5s mini-sectors (numbered, sliced at each boundary)."""
    rc, cc = _colors(ref_color, cmp_color)
    seg = mini_df.copy()
    seg["label"] = seg["mini"].astype(int).astype(str)
    return _dominance_map(delta_df, seg, ref_code, cmp_code, rc=rc, cc=cc,
                          title="Mini-sector dominance (who is faster where)",
                          number_name="mini #")


def sector_dominance_map(
    delta_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    ref_code: str = "REF",
    cmp_code: str = "CMP",
    *,
    ref_color: str | None = None,
    cmp_color: str | None = None,
) -> go.Figure:
    """Dominance map over the 3 official sectors — same format as the mini-sector map."""
    rc, cc = _colors(ref_color, cmp_color)
    seg = sector_df.copy()
    seg["label"] = ["S" + str(int(s)) for s in seg["sector"]]
    return _dominance_map(delta_df, seg, ref_code, cmp_code, rc=rc, cc=cc,
                          title="Sector dominance (who is faster where)",
                          number_name="sector #")
