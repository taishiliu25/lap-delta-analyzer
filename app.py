"""Lap Delta Analyzer — Streamlit dashboard.

Run locally with:  streamlit run app.py

Two pages that share **one** analysis dashboard — only the data *input* differs:

- **FastF1 Analysis** — cascading dropdowns (Year → Event → Session → driver+lap per side)
  select two real F1 laps.
- **iRacing (.ibt)** upload your own ``.ibt`` export; the stream is split into laps and two
  are selected.

Both feed the identical, source-agnostic :func:`render_comparison` — synced traces, gain/loss
delta, sector + mini-sector analysis, dominance + delta track maps, gear/RPM and corner
coaching — so the pages look and behave the same. Everything downstream sees only canonical
:class:`~lap_delta.schema.Lap` objects.
"""

from __future__ import annotations

import datetime as _dt
import pathlib
import sys

import streamlit as st

# Make `lap_delta` importable even without `pip install -e .`.
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from lap_delta import catalog, minisectors, sectors, units  # noqa: E402
from lap_delta.align import compute_delta, total_delta  # noqa: E402
from lap_delta.coaching import coaching_table  # noqa: E402
from lap_delta.corners import detect_corners  # noqa: E402
from lap_delta.schema import Lap, LapMeta, format_laptime  # noqa: E402
from lap_delta.viz import (  # noqa: E402
    delta_gain_loss_figure,
    gear_rpm_figure,
    minisector_dominance_map,
    sector_dominance_map,
    traces_figure,
    track_map_figure,
)

st.set_page_config(page_title="Lap Delta Analyzer", page_icon="🏁", layout="wide")

FIRST_TELEMETRY_YEAR = 2018
DEMO_DIR = pathlib.Path(__file__).parent / "data" / "demo"


# --- theme -------------------------------------------------------------------------

_THEME_CSS = """
<style>
:root { --pw-red:#E10600; --pw-bg:#0E1117; --pw-panel:#161A21; --pw-elev:#1A2029;
  --pw-line:#242C39; --pw-muted:#8A93A3; --pw-text:#D7DCE3; }
/* Hide the Deploy button + main menu (both inside stToolbarActions), but KEEP the
   toolbar — the collapsed-sidebar reopen button is a sibling inside it. */
[data-testid="stToolbarActions"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], #MainMenu, footer { display:none !important; }
header[data-testid="stHeader"] { background:transparent; }
/* Make the collapsed-sidebar reopen button clearly visible on the dark theme */
[data-testid="stExpandSidebarButton"] { opacity:1 !important; visibility:visible !important;
  background:var(--pw-elev) !important; border:1px solid var(--pw-line) !important;
  border-radius:8px !important; }
[data-testid="stExpandSidebarButton"] span,
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {
  color:var(--pw-text) !important; opacity:1 !important; }
.block-container { padding-top: 2.2rem; padding-bottom: 3.5rem; max-width: 1500px; }

/* Top bar */
.pw-top { display:flex; align-items:center; justify-content:space-between;
  border:1px solid var(--pw-line); border-left:3px solid var(--pw-red);
  background:linear-gradient(90deg, rgba(225,6,0,0.06), rgba(22,26,33,0.0) 45%);
  border-radius:10px; padding:.55rem .9rem; margin-bottom:1rem; }
.pw-brand { font-weight:700; letter-spacing:.14em; font-size:.92rem; }
.pw-brand .pw-red { color:var(--pw-red); }
.pw-brand .pw-sub { color:var(--pw-muted); font-weight:600; margin-left:.4rem; }
.pw-chip { font-size:.72rem; letter-spacing:.16em; font-weight:700; color:var(--pw-muted);
  border:1px solid var(--pw-line); border-radius:999px; padding:.2rem .7rem; }

/* Section headline */
.pw-h { display:flex; align-items:baseline; gap:.6rem; margin:.2rem 0 .1rem; }
.pw-h .t { font-size:1.15rem; font-weight:700; letter-spacing:.01em; }
.pw-h .s { color:var(--pw-muted); font-size:.9rem; }

/* KPI metric cards — elevated surface with a hairline top highlight */
[data-testid="stMetric"] { background:var(--pw-elev); border:1px solid var(--pw-line);
  border-radius:12px; padding:.8rem 1rem;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.03); }
[data-testid="stMetricLabel"] p { text-transform:uppercase; letter-spacing:.11em;
  font-size:.68rem; color:var(--pw-muted); }
[data-testid="stMetricValue"] { font-variant-numeric:tabular-nums; color:var(--pw-text);
  font-family:'JetBrains Mono','SF Mono',Consolas,monospace; font-size:1.55rem;
  letter-spacing:-0.01em; }
[data-testid="stMetricDelta"] { font-variant-numeric:tabular-nums; }

/* Monospace, tabular numbers in data tables */
[data-testid="stDataFrame"] { font-variant-numeric:tabular-nums; }
h4 { border-left:3px solid var(--pw-red); padding-left:.55rem; margin-top:1.1rem; }
section[data-testid="stSidebar"] { border-right:1px solid var(--pw-line); }
</style>
"""


def inject_theme() -> None:
    """Inject the shared pit-wall CSS. Called by both pages so they are pixel-identical."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


def _top_bar(source_label: str) -> None:
    st.markdown(
        f'<div class="pw-top">'
        f'<div class="pw-brand"><span class="pw-red">▮</span> LAP DELTA'
        f'<span class="pw-sub">TELEMETRY</span></div>'
        f'<div class="pw-chip">{source_label}</div></div>',
        unsafe_allow_html=True,
    )


def _headline(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="pw-h"><span class="t">{title}</span>'
                f'<span class="s">{subtitle}</span></div>', unsafe_allow_html=True)


# --- FastF1 cached data access -----------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_adapter():
    from lap_delta.adapters import FastF1Adapter

    return FastF1Adapter()


@st.cache_data(show_spinner="Loading season schedule…")
def load_schedule(year: int):
    """Return ``(event_options, sessions_by_round)`` for a year (real weekends only)."""
    import pandas as pd

    sched = get_adapter().schedule(year)
    events = catalog.event_options(sched)
    sessions_by_round: dict[int, list] = {}
    for _, row in sched.iterrows():
        rnd = row.get("RoundNumber")
        if not pd.isna(rnd):
            sessions_by_round[int(rnd)] = catalog.session_options(row)
    return events, sessions_by_round


@st.cache_data(show_spinner="Loading session entry list…")
def load_summary(year: int, event_name: str, session: str):
    """Return ``(driver_options, laps_df)`` from a cheap no-telemetry session load."""
    import pandas as pd

    results, laps = get_adapter().session_summary(year, event_name, session)
    drivers = catalog.driver_options(results)
    cols = [c for c in ("Driver", "LapNumber", "LapTime", "Deleted", "IsAccurate")
            if c in laps.columns]
    laps_df = pd.DataFrame(laps[cols]).reset_index(drop=True)
    return drivers, laps_df


@st.cache_data(show_spinner="Loading telemetry for the selected laps…")
def load_two_laps(year: int, event_name: str, session: str, selections: tuple):
    return get_adapter().load_selected(year, event_name, session, list(selections))


@st.cache_data(show_spinner=False)
def load_corners(year: int, event_name: str, session: str):
    """Official numbered corners for the track (cached; ``[]`` if unavailable)."""
    return get_adapter().circuit_corners(year, event_name, session)


# --- iRacing cached data access ----------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_iracing_adapter():
    from lap_delta.adapters import IracingIbtAdapter

    return IracingIbtAdapter()


@st.cache_data(show_spinner="Parsing .ibt telemetry…")
def load_ibt_bytes(data: bytes, name: str) -> list[Lap]:  # noqa: ARG001 - name aids cache key
    """Parse an uploaded ``.ibt`` (cache keyed on file content)."""
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".ibt", delete=False) as tf:
        tf.write(data)
        tmp = tf.name
    try:
        return get_iracing_adapter().load(tmp)
    finally:
        os.unlink(tmp)


# --- helpers -----------------------------------------------------------------------

def _sel_faster_to_code(df, ref_code: str, cmp_code: str):
    """Replace REF/CMP tokens in a sector table's ``faster`` column with driver codes."""
    out = df.copy()
    out["faster"] = out["faster"].map({"REF": ref_code, "CMP": cmp_code})
    return out


def _driver_lap_picker(side: str, drivers: list[dict], laps_df, key: str):
    """Driver dropdown + that driver's lap dropdown; returns (code, lap_number, color)."""
    codes = [d["code"] for d in drivers]
    labels = {d["code"]: f'{d["code"]} · {d["name"]}' for d in drivers}
    default = 0 if side == "Reference" else (1 if len(codes) > 1 else 0)
    code = st.selectbox(f"{side} driver", codes, index=default,
                        format_func=lambda c: labels.get(c, c), key=f"{key}_drv")
    color = next((d["color"] for d in drivers if d["code"] == code), None)

    opts = catalog.lap_options(laps_df, code)
    if not opts:
        st.warning(f"No timed laps for {code}.")
        return code, None, color
    lap = st.selectbox(f"{side} lap", opts, format_func=lambda o: o["label"], key=f"{key}_lap")
    return code, lap["lap_number"], color


# --- the shared dashboard (source-agnostic) ----------------------------------------

def render_comparison(
    reference: Lap,
    comparison: Lap,
    *,
    ref_code: str,
    cmp_code: str,
    ref_color: str | None,
    cmp_color: str | None,
    corner_markers: list[tuple[str, float]],
    imperial: bool,
    title: str,
    subtitle: str = "",
) -> None:
    """Draw the entire comparison dashboard for any two canonical laps.

    Both pages call exactly this, so the FastF1 and iRacing dashboards are identical; only the
    upstream selection of ``reference`` / ``comparison`` differs.
    """
    delta_df = compute_delta(comparison, reference)
    total = total_delta(delta_df)

    main_df = sectors.main_sectors(reference, comparison)
    edges_frac, _n, _cached = minisectors.get_or_compute(reference)
    mini_df = sectors.mini_sectors(delta_df, edges_frac)

    official = all(getattr(reference.meta, f"sector{i}_s") is not None for i in (1, 2, 3)) \
        and all(getattr(comparison.meta, f"sector{i}_s") is not None for i in (1, 2, 3))

    _headline(title, subtitle)

    mini_won = int((mini_df["faster"] == "CMP").sum())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"{ref_code} · reference", format_laptime(reference.meta.lap_time_s))
    k2.metric(f"{cmp_code} · comparison", format_laptime(comparison.meta.lap_time_s))
    k3.metric(f"{cmp_code} − {ref_code}", f"{total:+.3f}s",
              delta=f"{total:+.3f}s", delta_color="inverse")
    k4.metric(f"mini-sectors {cmp_code} faster", f"{mini_won}/{len(mini_df)}")

    verdict = (f"**{cmp_code} is {abs(total):.3f}s "
               f"{'behind' if total > 0 else 'ahead of'} {ref_code}** over the lap "
               f"(delta = {cmp_code} − {ref_code}).")
    st.markdown(verdict)

    ref_label, cmp_label = reference.meta.label, comparison.meta.label

    st.plotly_chart(
        traces_figure(delta_df, ref_label, cmp_label, imperial=imperial,
                      ref_color=ref_color, cmp_color=cmp_color),
        use_container_width=True,
    )
    st.plotly_chart(
        delta_gain_loss_figure(delta_df, ref_code, cmp_code, imperial=imperial),
        use_container_width=True,
    )

    # Official sectors: table + dominance map (boundaries at the real timing gates).
    st.markdown("#### Official sectors" if official
                else "#### Sectors (equal thirds, no timing gates in source)")
    s_left, s_right = st.columns([1, 1])
    with s_left:
        main_show = _sel_faster_to_code(main_df, ref_code, cmp_code).rename(columns={
            "sector": "Sector", "ref_s": f"{ref_code} (s)", "cmp_s": f"{cmp_code} (s)",
            "delta_s": "Δ (s)", "faster": "Faster",
        })
        st.dataframe(main_show, use_container_width=True, hide_index=True)
    with s_right:
        sec_edges = sectors.main_sector_edges(reference, delta_df)
        sector_dom_df = main_df.assign(dist_start=sec_edges[:-1], dist_end=sec_edges[1:])
        st.plotly_chart(
            sector_dominance_map(delta_df, sector_dom_df, ref_code, cmp_code,
                                 ref_color=ref_color, cmp_color=cmp_color),
            use_container_width=True,
        )

    # Mini-sectors: table + dominance map (sliced so each mini-sector is visible).
    st.markdown(f"#### Mini-sectors ({len(mini_df)} × ~5 s, fixed per track)")
    du = units.distance_unit(imperial)
    mini_show = _sel_faster_to_code(mini_df, ref_code, cmp_code)
    mini_show["dist_start"] = units.distance(mini_show["dist_start"].to_numpy(), imperial).round(0)
    mini_show["dist_end"] = units.distance(mini_show["dist_end"].to_numpy(), imperial).round(0)
    mini_show = mini_show.rename(columns={
        "mini": "Mini", "dist_start": f"start ({du})", "dist_end": f"end ({du})",
        "ref_s": f"{ref_code} (s)", "cmp_s": f"{cmp_code} (s)",
        "delta_s": "Δ (s)", "faster": "Faster",
    })
    m_left, m_right = st.columns([1, 1])
    with m_left:
        st.dataframe(mini_show, use_container_width=True, hide_index=True, height=460)
    with m_right:
        st.plotly_chart(
            minisector_dominance_map(delta_df, mini_df, ref_code, cmp_code,
                                     ref_color=ref_color, cmp_color=cmp_color),
            use_container_width=True,
        )

    # Delta-colored track map + gear/RPM.
    left2, right2 = st.columns([1, 1])
    with left2:
        st.plotly_chart(track_map_figure(delta_df, imperial=imperial, corners=corner_markers),
                        use_container_width=True)
    with right2:
        st.plotly_chart(
            gear_rpm_figure(delta_df, ref_label, cmp_label,
                            ref_color=ref_color, cmp_color=cmp_color, imperial=imperial),
            use_container_width=True,
        )

    st.markdown("#### Corner-by-corner (worst time loss first)")
    table = coaching_table(reference, comparison, delta_df, imperial=imperial,
                           ref_code=ref_code, cmp_code=cmp_code)
    if table.empty:
        st.info("No corners detected.")
    else:
        st.dataframe(table, use_container_width=True, hide_index=True)

    st.caption(f"Delta convention: {cmp_code} − {ref_code} (positive/red ⇒ {cmp_code} slower). "
               "Coaching hints are heuristic suggestions.")


# --- FastF1 page -------------------------------------------------------------------

def _fastf1_unavailable(what: str, exc: Exception) -> None:
    """Friendly message + Retry when a live FastF1 fetch fails (common on cold cloud starts).

    FastF1 downloads live F1 data on first use; on shared hosting with no warm cache that can
    be slow or rate-limited. We never surface the raw traceback — we explain it and offer a
    retry (a rerun re-attempts, since cached calls don't cache exceptions)."""
    st.warning(
        f"Couldn't load {what} from FastF1 just now. It fetches live F1 data on first use, "
        "which can be slow or rate-limited on shared hosting. Click **Retry** in a moment — the "
        "next attempt usually works — or open the **iRacing (.ibt)** page, which runs fully "
        "offline from an uploaded file."
    )
    st.button("Retry", type="primary", key=f"retry_{what}".replace(" ", "_"))
    with st.expander("Technical details"):
        st.code(str(exc))
    st.stop()


def fastf1_page():
    inject_theme()
    _top_bar("FASTF1 · FORMULA 1")

    years = list(range(_dt.date.today().year, FIRST_TELEMETRY_YEAR - 1, -1))
    with st.sidebar:
        st.header("Session")
        year = st.selectbox("Year", years, index=years.index(2023) if 2023 in years else 0)

        try:
            events, sessions_by_round = load_schedule(year)
        except Exception as exc:  # noqa: BLE001
            _fastf1_unavailable(f"the {year} schedule", exc)
            return
        if not events:
            st.warning("No events found for this year.")
            return

        default_ev = next((i for i, e in enumerate(events)
                           if "ital" in e["name"].lower() or "monza" in e["name"].lower()), 0)
        ev_i = st.selectbox("Event", range(len(events)), index=default_ev,
                            format_func=lambda i: events[i]["label"])
        event = events[ev_i]

        session_opts = sessions_by_round.get(event["round"], [("Qualifying", "Q")])
        s_labels = [lbl for lbl, _ in session_opts]
        default_s = next((i for i, (_, sid) in enumerate(session_opts) if sid == "Q"), 0)
        s_i = st.selectbox("Session", range(len(session_opts)), index=default_s,
                           format_func=lambda i: s_labels[i])
        session_id = session_opts[s_i][1]

        try:
            drivers, laps_df = load_summary(year, event["name"], session_id)
        except Exception as exc:  # noqa: BLE001
            _fastf1_unavailable(f"the entry list for {event['name']} {year}", exc)
            return
        if not drivers:
            st.warning("No drivers found for this session.")
            return

        st.header("Laps")
        ref_code, ref_lap, ref_color = _driver_lap_picker("Reference", drivers, laps_df, "ref")
        cmp_code, cmp_lap, cmp_color = _driver_lap_picker("Comparison", drivers, laps_df, "cmp")

        st.header("Display")
        imperial = st.toggle("Imperial units (mph / ft)", value=False)
        analyze = st.button("Analyze", type="primary", use_container_width=True)

    if ref_lap is None or cmp_lap is None:
        st.info("Pick a reference and comparison lap in the sidebar.")
        return
    if (ref_code, ref_lap) == (cmp_code, cmp_lap):
        st.warning("Reference and comparison are the same lap. Pick two different laps.")
        return

    sel = {
        "year": year, "event": event["name"], "session": session_id,
        "ref_code": ref_code, "ref_lap": ref_lap, "ref_color": ref_color,
        "cmp_code": cmp_code, "cmp_lap": cmp_lap, "cmp_color": cmp_color,
    }
    if analyze:
        st.session_state["committed"] = sel
    committed = st.session_state.get("committed")
    if not committed:
        st.info("Set your two laps, then click **Analyze**.")
        return

    _render_fastf1(committed, imperial)


def _render_fastf1(sel: dict, imperial: bool):
    """FastF1-specific: load the two selected laps, then hand off to the shared dashboard."""
    selections = ((sel["ref_code"], sel["ref_lap"]), (sel["cmp_code"], sel["cmp_lap"]))
    try:
        laps = load_two_laps(sel["year"], sel["event"], sel["session"], selections)
    except Exception as exc:  # noqa: BLE001
        _fastf1_unavailable("the telemetry for the selected laps", exc)
        return

    by_key = {(lp.meta.driver, lp.meta.lap_number): lp for lp in laps}
    reference = by_key.get((sel["ref_code"], sel["ref_lap"]))
    comparison = by_key.get((sel["cmp_code"], sel["cmp_lap"]))
    if reference is None or comparison is None:
        st.error("Selected laps could not be built from telemetry (try different laps).")
        return

    corner_markers = [(c["label"], c["distance_m"])
                      for c in load_corners(sel["year"], sel["event"], sel["session"])]
    if not corner_markers:
        corner_markers = [(str(c.number), c.apex_dist) for c in detect_corners(reference)]

    render_comparison(
        reference, comparison,
        ref_code=sel["ref_code"], cmp_code=sel["cmp_code"],
        ref_color=sel["ref_color"] or reference.meta.team_color,
        cmp_color=sel["cmp_color"] or comparison.meta.team_color,
        corner_markers=corner_markers, imperial=imperial,
        title=f"{sel['event']} {sel['year']} · {sel['session']}",
        subtitle=f"{sel['cmp_code']} vs {sel['ref_code']} (reference)",
    )


# --- iRacing page ------------------------------------------------------------------

def _ibt_lap_options(laps: list[Lap]) -> list[dict]:
    """Label each split segment; valid flying laps first, then by lap time."""
    opts = []
    for i, lp in enumerate(laps):
        m = lp.meta
        tag = "flying" if m.is_valid else ("pit" if m.is_pit else "partial")
        opts.append({
            "idx": i, "valid": m.is_valid, "lap_time_s": m.lap_time_s or 1e9,
            "label": f"L{m.lap_number} · {format_laptime(m.lap_time_s)} · {tag}",
        })
    opts.sort(key=lambda o: (not o["valid"], o["lap_time_s"]))
    return opts


def iracing_page():
    inject_theme()
    _top_bar("IRACING · .IBT")

    with st.sidebar:
        st.header("Telemetry file")
        up = st.file_uploader("iRacing .ibt file", type=["ibt"])

        laps: list[Lap] | None = None
        if up is not None:
            try:
                laps = load_ibt_bytes(up.getvalue(), up.name)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read .ibt: {exc}")
                return

        if laps is None:
            st.info("Upload an iRacing `.ibt` file to begin. Any practice/qualifying/race "
                    "export works; it's split into laps automatically.")
            st.stop()
        if not laps:
            st.warning("No laps could be extracted from this file.")
            return

        meta0 = laps[0].meta
        st.caption(f"**{meta0.track or 'Unknown track'}** · {meta0.compound or ''} · "
                   f"{len(laps)} lap(s) split")

        st.header("Laps")
        opts = _ibt_lap_options(laps)
        if len(opts) < 2:
            st.warning("Need at least two laps in the file to compare.")
            return
        labels = {o["idx"]: o["label"] for o in opts}
        order = [o["idx"] for o in opts]
        ref_idx = st.selectbox("Reference lap", order, index=0,
                               format_func=lambda i: labels[i], key="ibt_ref")
        cmp_default = 1 if len(order) > 1 else 0
        cmp_idx = st.selectbox("Comparison lap", order, index=cmp_default,
                               format_func=lambda i: labels[i], key="ibt_cmp")

        st.header("Display")
        imperial = st.toggle("Imperial units (mph / ft)", value=False)

    if ref_idx == cmp_idx:
        st.warning("Reference and comparison are the same lap. Pick two different laps.")
        return

    reference, comparison = laps[ref_idx], laps[cmp_idx]
    ref_code = f"L{reference.meta.lap_number}"
    cmp_code = f"L{comparison.meta.lap_number}"
    corner_markers = [(str(c.number), c.apex_dist) for c in detect_corners(reference)]

    render_comparison(
        reference, comparison,
        ref_code=ref_code, cmp_code=cmp_code,
        ref_color=None, cmp_color=None,  # viz falls back to blue/red defaults
        corner_markers=corner_markers, imperial=imperial,
        title=f"{reference.meta.track or 'iRacing'} · {reference.meta.compound or ''}".strip(" ·"),
        subtitle=f"{cmp_code} vs {ref_code} (reference) · iRacing",
    )


# --- Demo page (bundled, offline) --------------------------------------------------

@st.cache_data(show_spinner=False)
def load_demo():
    """Load the bundled Monza 2023 Q comparison (VER vs PER) from small CSVs.

    Pre-computed canonical laps, so this renders instantly with **no FastF1 download** —
    the reliable first impression on shared hosting where a live fetch may be slow.
    """
    import json

    import pandas as pd

    meta = json.loads((DEMO_DIR / "meta.json").read_text())
    out = {}
    for key in ("ver", "per"):
        df = pd.read_csv(DEMO_DIR / f"{key}_monza_2023q.csv")
        out[key] = Lap(df, LapMeta(**meta[key]))
    return out["ver"], out["per"]


def demo_page():
    inject_theme()
    _top_bar("DEMO · MONZA 2023 Q")
    st.caption("A pre-loaded example so you can see the analysis instantly, with no data "
               "download. Use **FastF1 Analysis** to pick your own laps, or **iRacing (.ibt)** "
               "to upload sim telemetry.")
    try:
        reference, comparison = load_demo()
    except Exception as exc:  # noqa: BLE001 - bundled data should always load
        st.error(f"Demo data unavailable: {exc}")
        return
    corner_markers = [(str(c.number), c.apex_dist) for c in detect_corners(reference)]
    render_comparison(
        reference, comparison,
        ref_code="VER", cmp_code="PER",
        ref_color=reference.meta.team_color, cmp_color=comparison.meta.team_color,
        corner_markers=corner_markers, imperial=False,
        title="Monza 2023 · Qualifying",
        subtitle="Verstappen vs Pérez (teammates) · PER vs VER (reference)",
    )


# --- entry -------------------------------------------------------------------------

def main():
    st.navigation(
        [
            st.Page(demo_page, title="Demo", icon="🏁", default=True),
            st.Page(fastf1_page, title="FastF1 Analysis", icon="📊"),
            st.Page(iracing_page, title="iRacing (.ibt)", icon="🏎️"),
        ],
        position="top",
    ).run()


# Streamlit executes the script with __name__ == "__main__"; guarding the entry lets tests
# import the page functions without triggering navigation as an import side effect.
if __name__ == "__main__":
    main()
