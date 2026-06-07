"""
LiDAR Survey Explorer
=====================
Password-protected Streamlit app for visualising WingtraRay LiDAR surveys.

Secrets required (.streamlit/secrets.toml):
    app_password = "..."        ← access gate

    [r2]
    account_id = "..."
    access_key = "..."          ← read-only token
    secret_key = "..."
    bucket     = "ptpn-bucket"

Run with:
    streamlit run lidar_app.py
"""

import os
import hmac
import base64
import numpy as np
import streamlit as st
import pydeck as pdk
import plotly.graph_objects as go

from lidar_r2 import list_surveys, load_bundle, refresh_surveys, presigned_url

APP_TITLE = "LiDAR Survey Explorer"
LOGO_PATH = "assets/wingtra_logo.png"

# --------------------------------------------------------------------------- #
#  Wingtra brand CSS
# --------------------------------------------------------------------------- #
MERCURY600 = "#1C2E36"
MERCURY500 = "#233A44"
URANUS300  = "#A3BABD"
URANUS0    = "#EDF2F2"
SUN_ORANGE = "#F46F29"

CSS = f"""
<style>
  .stApp {{ background-color: {URANUS0}; }}
  section[data-testid="stSidebar"] {{ background-color: {MERCURY500}; }}
  section[data-testid="stSidebar"] * {{ color: #FFFFFF; }}
  section[data-testid="stSidebar"] input,
  section[data-testid="stSidebar"] textarea {{
      background-color: #16242B; color: #FFFFFF; }}
  section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
      background-color: #16242B; color: #FFFFFF; border-color: #3A4F59; }}
  section[data-testid="stSidebar"] div[data-baseweb="select"] svg {{ fill: #FFFFFF; }}
  div[data-baseweb="popover"] ul, ul[role="listbox"] {{ background-color: #233A44; }}
  div[data-baseweb="popover"] li, ul[role="listbox"] li {{ color: #FFFFFF; }}
  ul[role="listbox"] li:hover {{ background-color: #16242B; }}

  .wingtra-header {{
      display: flex; align-items: center; gap: 18px;
      background-color: {MERCURY600};
      border-bottom: 3px solid {SUN_ORANGE};
      padding: 14px 22px; margin: -1.2rem -1.2rem 1.0rem -1.2rem;
  }}
  .wingtra-header img {{ height: 30px; }}
  .wingtra-wordmark {{ color: {SUN_ORANGE}; font-size: 28px; font-weight: 800; letter-spacing: .5px; }}
  .wingtra-title {{ color: #FFFFFF; font-size: 20px; font-weight: 800; line-height: 1.2; }}
  .wingtra-subtitle {{ color: {URANUS300}; font-size: 12.5px; font-weight: 500; }}

  div[data-testid="stMetric"] {{
      background: #FFFFFF; border: 1px solid #D7E0E2; border-radius: 12px;
      padding: 12px 16px; box-shadow: 0 1px 3px rgba(28,46,54,0.06); }}
  div[data-testid="stMetricLabel"] p {{ color: #5b6b72; font-weight: 600; }}

  .wingtra-card {{
      max-width: 680px; margin: 10px 0 0; padding: 22px 26px;
      background: #FFFFFF; border: 1px solid #D7E0E2; border-radius: 14px;
      box-shadow: 0 1px 4px rgba(28,46,54,0.08); }}
  .wingtra-card-title {{ color: {MERCURY600}; font-size: 18px; font-weight: 800; margin-bottom: 6px; }}
  .wingtra-card p, .wingtra-card li {{ color: #46555b; font-size: 14px; line-height: 1.6; }}
  .wingtra-card ul {{ margin: 8px 0 0; padding-left: 20px; }}
</style>
"""

# --------------------------------------------------------------------------- #
#  Branding
# --------------------------------------------------------------------------- #
def _logo_b64():
    if os.path.exists(LOGO_PATH):
        try:
            return base64.b64encode(open(LOGO_PATH, "rb").read()).decode()
        except Exception:
            return None
    return None


def render_header(subtitle="Point cloud · Digital terrain model"):
    st.markdown(CSS, unsafe_allow_html=True)
    b64   = _logo_b64()
    brand = (f'<img src="data:image/png;base64,{b64}" alt="Wingtra"/>' if b64
             else f'<span class="wingtra-wordmark">wingtra</span>')
    sub   = f'<div class="wingtra-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="wingtra-header">{brand}'
        f'<div><div class="wingtra-title">{APP_TITLE}</div>{sub}</div></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
#  Password gate  (same pattern as the multispectral app)
# --------------------------------------------------------------------------- #
def check_password():
    """Returns True if the user has entered the correct password."""
    if st.session_state.get("auth_ok"):
        return True

    render_header()
    st.caption("Enter the access password to continue.")
    pw = st.text_input("Password", type="password", key="pw_input")
    if pw:
        expected = st.secrets.get("app_password", "")
        if expected and hmac.compare_digest(pw, expected):
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _fmt_n(n):
    if n >= 1_000_000: return f"{n/1e6:.1f}M"
    if n >= 1_000:     return f"{n/1e3:.0f}K"
    return str(n)

def _layer_label(name):
    return {"ground":    "Ground (class 2)",
            "nonground": "Non-ground",
            "points":    "All points (unclassified)"}.get(name, name)


# --------------------------------------------------------------------------- #
#  Point cloud tab
# --------------------------------------------------------------------------- #
def render_point_cloud(bundle, visible_layers, point_size):
    meta = bundle["meta"]

    import pandas as pd
    frames = [df for name, df in bundle["layers"].items()
              if visible_layers.get(name, True)]

    if not frames:
        st.info("No layers selected — tick at least one in the sidebar.")
        return

    combined = pd.concat(frames, ignore_index=True)
    bounds   = meta["bounds_4326"]
    clat     = (bounds["lat_min"] + bounds["lat_max"]) / 2
    clon     = (bounds["lon_min"] + bounds["lon_max"]) / 2
    lon_span = bounds["lon_max"] - bounds["lon_min"]
    zoom     = max(10, min(17, round(np.log2(360 / lon_span) - 1)))

    layer = pdk.Layer(
        "PointCloudLayer",
        data=combined,
        get_position=["lon", "lat", "z"],
        get_color=["r", "g", "b"],
        point_size=point_size,
        pickable=True,
    )
    view = pdk.ViewState(latitude=clat, longitude=clon,
                         zoom=zoom, pitch=55, bearing=0)
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        map_provider=None,
        tooltip={"text": "Z: {z} m"},
    )
    st.pydeck_chart(deck, use_container_width=True)

    n_display = sum(len(bundle["layers"][n]) for n in visible_layers
                    if visible_layers[n] and n in bundle["layers"])
    cols = st.columns(4)
    cols[0].metric("Displayed",      _fmt_n(n_display))
    cols[1].metric("Total surveyed", _fmt_n(meta.get("n_total", 0)))
    cols[2].metric("Elevation min",  f"{meta.get('z_min', 0):.1f} m")
    cols[3].metric("Elevation max",  f"{meta.get('z_max', 0):.1f} m")

    st.caption(
        f"{'Classified' if meta['mode'] == 'classified' else 'Unclassified'} "
        f"point cloud · {_fmt_n(n_display)} display pts subsampled from "
        f"{_fmt_n(meta.get('n_total', 0))} raw · "
        f"Source CRS: {meta.get('crs_source', '—')} → WGS84"
    )


# --------------------------------------------------------------------------- #
#  DTM surface tab
# --------------------------------------------------------------------------- #
def render_dtm(bundle, vert_exag):
    dtm, dtm_meta = bundle["dtm"], bundle["dtm_meta"]
    if dtm is None:
        st.info("No DTM available for this survey.")
        return

    # ---- NaN / nodata handling ------------------------------------------ #
    # GeoTIFFs often have nodata regions converted to NaN during preprocessing.
    # Plotly Surface renders NaN cells as transparent holes (correct), but
    # passes them straight through to the hover template where they appear as
    # "NaN". Fix: build a separate hover array with NaN replaced by the minimum
    # finite value — those cells are invisible holes so the fill is never shown.
    finite_mask = np.isfinite(dtm)
    if not finite_mask.any():
        st.warning("DTM contains no valid elevation values — nodata may cover the entire grid.")
        return

    z_min_disp = float(np.min(dtm[finite_mask]))
    z_max_disp = float(np.max(dtm[finite_mask]))

    # Fill NaN for hover tooltip (invisible for hole cells)
    dtm_hover = np.where(finite_mask, dtm, z_min_disp)

    # ---- Coordinate conversion to local metres -------------------------- #
    bounds = dtm_meta["bounds_4326"]
    lat0   = (bounds["lat_min"] + bounds["lat_max"]) / 2
    lon0   = (bounds["lon_min"] + bounds["lon_max"]) / 2
    m_lat  = 111_320.0
    m_lon  = 111_320.0 * np.cos(np.radians(lat0))

    nrows, ncols = dtm.shape
    lons = np.linspace(bounds["lon_min"], bounds["lon_max"], ncols)
    lats = np.linspace(bounds["lat_min"], bounds["lat_max"], nrows)
    x_m  = (lons - lon0) * m_lon
    y_m  = (lats - lat0) * m_lat

    # z_pl retains NaN so Plotly renders nodata regions as holes
    z_pl = dtm * vert_exag

    # ---- Aspect ratio --------------------------------------------------- #
    span_x = float(x_m.max() - x_m.min())
    span_y = float(y_m.max() - y_m.min())
    span_z = (z_max_disp - z_min_disp) * vert_exag
    ms     = max(span_x, span_y, span_z, 1)

    # ---- Colorbar ticks from finite range ------------------------------- #
    tick_vals = np.linspace(z_min_disp, z_max_disp, 6)

    fig = go.Figure(data=[go.Surface(
        z=z_pl, x=x_m, y=y_m,
        colorscale="Earth",
        cmin=z_min_disp,
        cmax=z_max_disp,
        colorbar=dict(
            title="Elevation (m)",
            tickvals=tick_vals,
            ticktext=[f"{v:.0f}" for v in tick_vals],
            thickness=14, len=0.7,
        ),
        # customdata carries the true (non-exaggerated) elevation for the tooltip.
        # NaN replaced above so the hover never shows "NaN".
        customdata=dtm_hover,
        hovertemplate=(
            "E: %{x:.0f} m<br>"
            "N: %{y:.0f} m<br>"
            "Elevation: %{customdata:.1f} m"
            "<extra></extra>"
        ),
    )])

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Easting offset (m)"),
            yaxis=dict(title="Northing offset (m)"),
            zaxis=dict(title="Elevation (m)"),
            aspectmode="manual",
            aspectratio=dict(x=span_x/ms, y=span_y/ms, z=span_z/ms),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Stats — use finite-only range, not meta (which may be stale/NaN)
    res  = dtm_meta.get("src_resolution_m", "—")
    cols = st.columns(4)
    cols[0].metric("Min elevation", f"{z_min_disp:.1f} m")
    cols[1].metric("Max elevation", f"{z_max_disp:.1f} m")
    cols[2].metric("Relief",        f"{z_max_disp - z_min_disp:.1f} m")
    cols[3].metric("Source res.",   f"{res} m/px" if isinstance(res, (int, float)) else str(res))

    st.caption(
        f"DTM from LiDAR360 · {nrows}×{ncols} grid (decimated for display) · "
        f"vertical exaggeration {vert_exag:.1f}× · "
        f"horizontal axes in metres from survey centre"
    )


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    if not check_password():
        st.stop()

    render_header()

    # ---- Sidebar --------------------------------------------------------- #
    with st.sidebar:
        st.header("Survey")
        surveys = list_surveys()

        if not surveys:
            st.warning("No surveys found in R2.")
            if st.button("⟳ Refresh"):
                refresh_surveys(); st.rerun()
            return

        selected = st.selectbox("Dataset", surveys,
                                format_func=lambda s: s["name"])

        with st.spinner(f"Loading {selected['name']} …"):
            try:
                bundle = load_bundle(selected["slug"])
            except Exception as e:
                st.error(f"Could not load bundle: {e}")
                return

        meta = bundle["meta"]

        # Point cloud controls
        st.header("Point cloud")
        point_size = st.slider(
            "Point size", 1, 6, 2,
            help=(
                "Size of each rendered point in pixels. "
                "Increase if the cloud looks sparse or hard to read; "
                "decrease if it looks blocky or saturated. "
                "Higher values also slow down rendering for large point counts."
            ),
        )

        visible_layers = {}
        for name in meta.get("layers", []):
            visible_layers[name] = st.checkbox(_layer_label(name), value=True)

        # DTM controls (only when DTM is present)
        vert_exag = 1.0
        if bundle["dtm"] is not None:
            st.header("DTM surface")
            vert_exag = st.slider(
                "Vertical exaggeration", 0.5, 5.0, 1.0, step=0.5,
                help=(
                    "Multiplies elevation values to amplify terrain relief. "
                    "1.0 = true scale — horizontal and vertical axes match. "
                    "Useful for relatively flat terrain where subtle slopes are "
                    "hard to see at true scale. Try 2–3× for plantation surveys."
                ),
            )

        # Info
        st.header("Info")
        st.markdown(
            f"**Mode:** {meta['mode'].capitalize()}  \n"
            f"**Raw points:** {_fmt_n(meta.get('n_total', 0))}  \n"
            f"**Source CRS:** {meta.get('crs_source', '—')}  \n"
            f"**DTM:** {'Included' if meta.get('dtm_available') else 'Not available'}"
        )

        # Downloads (presigned URLs — browser ↔ R2 directly)
        st.header("Downloads")
        raw_las_key = meta.get("raw_las_key")
        raw_dtm_key = meta.get("raw_dtm_key")

        if raw_las_key:
            try:
                st.link_button("⬇  Raw LAS (full res)",
                               presigned_url(raw_las_key, ttl=900),
                               use_container_width=True, type="primary")
            except Exception:
                st.caption("LAS download unavailable")

        if raw_dtm_key:
            try:
                st.link_button("⬇  DTM GeoTIFF (LiDAR360)",
                               presigned_url(raw_dtm_key, ttl=900),
                               use_container_width=True)
            except Exception:
                st.caption("DTM download unavailable")

        if not raw_las_key and not raw_dtm_key:
            st.caption("Re-run the preprocessing script to add raw file downloads.")

        if st.button("⟳ Refresh surveys"):
            refresh_surveys(); st.rerun()

    # ---- Tabs ------------------------------------------------------------ #
    tab_labels = ["☁  Point Cloud"]
    if bundle["dtm"] is not None:
        tab_labels.append("⛰  DTM Surface")

    tabs = st.tabs(tab_labels)

    with tabs[0]:
        render_point_cloud(bundle, visible_layers, point_size)

    if len(tabs) > 1:
        with tabs[1]:
            render_dtm(bundle, vert_exag)


if __name__ == "__main__":
    main()
