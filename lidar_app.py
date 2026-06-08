"""
LIDAR Survey Explorer
=====================
Password-protected Streamlit app for visualising WingtraRay LIDAR surveys.

Secrets (.streamlit/secrets.toml):
    app_password   = "..."
    r2_public_url  = "https://pub-<hash>.r2.dev"

    [r2]
    account_id = "..."
    access_key = "..."     # read-only
    secret_key = "..."
    bucket     = "ptpn-bucket"
"""

import os
import hmac
import base64
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import pydeck as pdk
import plotly.graph_objects as go

from lidar_r2 import list_surveys, load_bundle, refresh_surveys, presigned_url

APP_TITLE   = "LIDAR Survey Explorer"
LOGO_PATH   = "assets/wingtra_logo.png"
PLACEHOLDER = "— Select a survey —"

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
  section[data-testid="stSidebar"] [data-testid="stTooltipHoverTarget"],
  section[data-testid="stSidebar"] [data-testid="stTooltipHoverTarget"] svg,
  section[data-testid="stSidebar"] [data-testid="stTooltipHoverTarget"] svg * {{
      color: {URANUS300} !important; fill: {URANUS300} !important;
      stroke: {URANUS300} !important; opacity: 1 !important; }}
  section[data-testid="stSidebar"] button[kind="secondary"],
  section[data-testid="stSidebar"] [data-testid="baseButton-secondary"] {{
      background-color: #16242B !important;
      border: 1px solid #3A4F59 !important; color: #FFFFFF !important; }}
  section[data-testid="stSidebar"] [data-testid="stLinkButtonContainer"] a,
  section[data-testid="stSidebar"] .stLinkButton a {{
      background-color: #16242B !important;
      border: 1px solid #3A4F59 !important; color: #FFFFFF !important; }}
  .wingtra-header {{
      display: flex; align-items: center; gap: 18px;
      background-color: {MERCURY600}; border-bottom: 3px solid {SUN_ORANGE};
      padding: 14px 22px; margin: -1.2rem -1.2rem 1.0rem -1.2rem; }}
  .wingtra-header img {{ height: 30px; }}
  .wingtra-wordmark {{ color: {SUN_ORANGE}; font-size: 28px; font-weight: 800; letter-spacing: .5px; }}
  .wingtra-title {{ color: #FFFFFF; font-size: 20px; font-weight: 800; line-height: 1.2; }}
  .wingtra-subtitle {{ color: {URANUS300}; font-size: 12.5px; font-weight: 500; }}
  div[data-testid="stMetric"] {{
      background: #FFFFFF; border: 1px solid #D7E0E2; border-radius: 12px;
      padding: 12px 16px; box-shadow: 0 1px 3px rgba(28,46,54,0.06); }}
  div[data-testid="stMetricLabel"] p {{ color: #5b6b72; font-weight: 600; }}
  .wingtra-card {{
      max-width: 680px; margin: 32px auto 0; padding: 28px 32px;
      background: #FFFFFF; border: 1px solid #D7E0E2; border-radius: 14px;
      box-shadow: 0 1px 4px rgba(28,46,54,0.08); }}
  .wingtra-card-title {{ color: {MERCURY600}; font-size: 20px; font-weight: 800; margin-bottom: 8px; }}
  .wingtra-card p, .wingtra-card li {{ color: #46555b; font-size: 14px; line-height: 1.6; }}
  .wingtra-card ul {{ margin: 10px 0 0; padding-left: 22px; }}
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
#  Password gate
# --------------------------------------------------------------------------- #
def check_password():
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

def _vivid_colors(z_vals):
    v = np.asarray(z_vals, dtype="float64")
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return (np.zeros_like(v, dtype="uint8"),) * 3
    z_lo = np.percentile(finite, 2)
    z_hi = np.percentile(finite, 98)
    t = np.clip((v - z_lo) / max(z_hi - z_lo, 1e-3), 0, 1)
    stops_t = [0.00, 0.15, 0.35, 0.55, 0.75, 0.90, 1.00]
    stops_r = [  20,    0,    0,   50,  230,  255,  180]
    stops_g = [  20,   80,  200,  220,  220,  120,    0]
    stops_b = [ 160,  220,  230,   80,    0,    0,    0]
    return (np.interp(t, stops_t, stops_r).astype("uint8"),
            np.interp(t, stops_t, stops_g).astype("uint8"),
            np.interp(t, stops_t, stops_b).astype("uint8"))


# --------------------------------------------------------------------------- #
#  Potree viewer
# --------------------------------------------------------------------------- #
_POTREE_COLOUR_JS = {
    "Elevation": """
        material.activeAttributeName = "elevation";
        material.gradient = Potree.Gradients.RAINBOW;
    """,
    "Intensity": """
        material.activeAttributeName = "intensity";
        material.gradient = Potree.Gradients.GRAYSCALE;
    """,
}

# CDN URLs — all dependencies that Potree 1.8.x requires before potree.js.
# From the official heidentor.html example:
# jquery, spectrum, jquery-ui, BinaryHeap, tween, d3, proj4,
# openlayers, i18next, jstree, laslaz — loaded before potree.js.
# THREE is imported as an ES module in the official examples; here we
# load it as a regular global script which potree.js also accepts.
_DEPS = [
    # THREE.js — potree.js needs window.THREE
    "https://cdnjs.cloudflare.com/ajax/libs/three.js/r124/three.min.js",
    # jQuery — UI and AJAX
    "https://cdnjs.cloudflare.com/ajax/libs/jquery/3.1.1/jquery.min.js",
    # jQuery UI — slider/accordion used by Potree sidebar
    "https://cdnjs.cloudflare.com/ajax/libs/jqueryui/1.12.1/jquery-ui.min.js",
    # TWEEN.js — camera animation, required by Viewer constructor
    "https://cdn.jsdelivr.net/gh/potree/potree@1.8.2/libs/tween/tween.min.js",
    # i18next — internationalisation used by Potree GUI
    "https://cdnjs.cloudflare.com/ajax/libs/i18next/21.6.14/i18next.min.js",
    # jstree — scene tree in Potree sidebar
    "https://cdnjs.cloudflare.com/ajax/libs/jstree/3.3.11/jstree.min.js",
    # proj4 — coordinate projections (used for georeferenced data)
    "https://cdnjs.cloudflare.com/ajax/libs/proj4js/2.8.0/proj4.js",
    # d3 — charts in Potree profile view
    "https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js",
    # spectrum — colour picker in Potree UI
    "https://cdnjs.cloudflare.com/ajax/libs/spectrum/1.8.0/spectrum.min.js",
]


def _potree_html(potree_url, assets_base, survey_name,
                 point_budget, edl_enabled, colour_mode):
    """
    Self-contained Potree 1.8.x viewer HTML.

    Dependency load order matches the official heidentor.html example exactly:
        THREE, jQuery, jQuery UI, TWEEN, i18next, jstree, proj4, d3, spectrum
        → potree.js
        → init script

    TWEEN.js is almost certainly the critical missing piece — the Viewer
    constructor sets up animation loops using TWEEN at construction time.

    The init script also logs Potree's exported keys to the console so the
    next error (if any) gives an exact diagnosis.
    """
    colour_js   = _POTREE_COLOUR_JS.get(colour_mode, _POTREE_COLOUR_JS["Elevation"])
    edl_js      = "true" if edl_enabled else "false"
    dep_scripts = "\n  ".join(f'<script src="{u}"></script>' for u in _DEPS)

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <link rel="stylesheet" href="{assets_base}/potree.css">
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    html, body {{ width: 100%; height: 100%; overflow: hidden; background: #0c0c12; }}
    #potree_render_area {{ position: absolute; width: 100%; height: 100%; }}
    #potree_sidebar_container {{ display: none !important; }}
    .potree_menu_toggle        {{ display: none !important; }}
    #loading_overlay {{
        position: absolute; top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        font-family: -apple-system, sans-serif; text-align: center;
        pointer-events: none; z-index: 10;
    }}
    #loading_overlay .msg {{ font-size: 14px; color: #A3BABD; margin-bottom: 6px; }}
    #loading_overlay .sub {{ font-size: 12px; color: #4a5a60; }}
    #error_overlay {{
        display: none; position: absolute; top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        font-family: -apple-system, sans-serif; text-align: center;
        max-width: 480px; padding: 0 20px; z-index: 10;
    }}
    #error_overlay .title  {{ font-size: 14px; font-weight: 700;
                               color: #F46F29; margin-bottom: 8px; }}
    #error_overlay .detail {{ font-size: 11px; color: #6a7880;
                               word-break: break-all; }}
  </style>
</head>
<body>
  <div id="potree_render_area"></div>

  <div id="loading_overlay">
    <div class="msg">Loading LIDAR point cloud…</div>
    <div class="sub">Streaming tiles from cloud storage</div>
  </div>

  <div id="error_overlay">
    <div class="title">⚠ Could not load point cloud</div>
    <div class="detail" id="error_detail">Check the browser console (F12) for details.</div>
  </div>

  <!-- All deps load before potree.js — matches heidentor.html order exactly -->
<!-- Worker monkey-patch: cross-origin Workers are silently blocked in srcdoc iframes.
       This intercepts Worker creation, fetches the script via XHR (CORS OK), rewrites
       relative importScripts to absolute URLs, and creates from a Blob URL instead. -->
  <script>
  (function() {{
      var _W = window.Worker;
      window.Worker = function(url, opts) {{
          try {{ return new _W(url, opts); }}
          catch(e) {{
              console.warn("[WorkerPatch] blob fallback for:", url);
              var base = url.substring(0, url.lastIndexOf('/') + 1);
              var xhr = new XMLHttpRequest();
              xhr.open('GET', url, false);
              xhr.send();
              if (xhr.status !== 200) throw new Error('Worker fetch failed: ' + xhr.status);
              var code = xhr.responseText.replace(
                  /importScripts\s*\(\s*['"]([^'"]+)['"]\s*\)/g,
                  function(m, p) {{
                      if (/^https?:\/\//.test(p)) return m;
                      try {{ return 'importScripts("' + new URL(p, base).href + '")'; }}
                      catch(err) {{ return m; }}
                  }}
              );
              var blob = new Blob([code], {{type: 'application/javascript'}});
              return new _W(URL.createObjectURL(blob), opts);
          }}
      }};
      window.Worker.prototype = _W.prototype;
  }})();
  </script>
  {dep_scripts}
  <script src="{assets_base}/libs/other/BinaryHeap.js"></script>
  <script src="{assets_base}/potree.js"></script>

  <script>
    function showError(msg) {{
        const lo = document.getElementById("loading_overlay");
        const eo = document.getElementById("error_overlay");
        const ed = document.getElementById("error_detail");
        if (lo) lo.style.display = "none";
        if (ed) ed.textContent   = String(msg);
        if (eo) eo.style.display = "block";
        console.error("[Potree]", msg);
    }}

    // Diagnostic dump — tells us exactly what potree.js exported
    console.log("[Potree] THREE:", typeof THREE, typeof THREE !== "undefined" ? "r"+THREE.REVISION : "MISSING");
    console.log("[Potree] TWEEN:", typeof TWEEN);
    console.log("[Potree] Potree object:", typeof Potree);
    if (typeof Potree === "object" || typeof Potree === "function") {{
        console.log("[Potree] Potree keys:", Object.keys(Potree).join(", "));
        console.log("[Potree] Potree.Viewer:", typeof Potree.Viewer);
    }}

    if (typeof Potree === "undefined") {{
        showError("Potree failed to load from R2.");
    }} else {{
        try {{
            Potree.scriptPath = "{assets_base}";

            var el = document.getElementById("potree_render_area");
                    function _awaitSize(cb) {{
                        if (el.clientWidth > 0 && el.clientHeight > 0) cb();
                        else requestAnimationFrame(function() {{ _awaitSize(cb); }});
                    }}
                    _awaitSize(function() {{
                    const viewer = new Potree.Viewer(el);
            viewer.setEDLEnabled({edl_js});
            viewer.setEDLRadius(1.4);
            viewer.setEDLStrength(0.4);
            viewer.setPointBudget({point_budget});
            viewer.setBackground("black");
            viewer.setFOV(60);

            const pcURL = "{potree_url}";
            console.log("[Potree] loading →", pcURL);

            viewer.loadGUI(() => {{
                const sb = document.getElementById("potree_sidebar_container");
                if (sb) sb.style.setProperty("display", "none", "important");
                const mt = document.querySelector(".potree_menu_toggle");
                if (mt) mt.style.setProperty("display", "none", "important");

                try {{
                    Potree.loadPointCloud(pcURL, "{survey_name}", e => {{
                        console.log("[Potree] point cloud loaded ✓");
                        const lo = document.getElementById("loading_overlay");
                        if (lo) lo.style.display = "none";

                        const pointcloud = e.pointcloud;
                        const material   = pointcloud.material;

                        {colour_js}

                        material.size          = 1;
                        material.pointSizeType = Potree.PointSizeType.ADAPTIVE;

                        viewer.scene.addPointCloud(pointcloud);
                        viewer.fitToScreen();
                    }});
                }} catch(e) {{
                    showError("loadPointCloud threw: " + e);
                }}
            }});

        }}); // end _awaitSize
        }} catch(e) {{
            showError("Viewer init failed: " + e);
        }}
    }}
  </script>
</body>
</html>"""


def render_potree_viewer(bundle, potree_settings):
    """Render the Potree point cloud viewer inside a Streamlit iframe."""
    meta       = bundle["meta"]
    r2_public  = st.secrets.get("r2_public_url", "").rstrip("/")

    if not r2_public:
        st.warning("**r2_public_url** not set in secrets — falling back to PyDeck view.")
        render_point_cloud(bundle, {}, 2)
        return

    potree_key  = meta.get("potree_key",
                           f"lidar/{meta['slug']}/potree/metadata.json")
    potree_url  = f"{r2_public}/{potree_key}"
    assets_base = f"{r2_public}/potree-assets"

    html = _potree_html(
        potree_url   = potree_url,
        assets_base  = assets_base,
        survey_name  = meta.get("name", meta.get("slug", "Survey")),
        point_budget = potree_settings["point_budget"],
        edl_enabled  = potree_settings["edl_enabled"],
        colour_mode  = potree_settings["colour_mode"],
    )

    components.html(html, height=650, scrolling=False)

    cols = st.columns(4)
    cols[0].metric("Total points",  _fmt_n(meta.get("n_total", 0)))
    cols[1].metric("Elevation min", f"{meta.get('z_min', 0):.1f} m")
    cols[2].metric("Elevation max", f"{meta.get('z_max', 0):.1f} m")
    cols[3].metric("Point budget",  _fmt_n(potree_settings["point_budget"]))

    edl_str = "EDL on" if potree_settings["edl_enabled"] else "EDL off"
    st.caption(
        f"Full {_fmt_n(meta.get('n_total', 0))}-point LIDAR dataset · "
        f"Potree progressive streaming · {edl_str} · "
        f"colour: {potree_settings['colour_mode'].lower()}"
    )


# --------------------------------------------------------------------------- #
#  PyDeck point cloud (fallback)
# --------------------------------------------------------------------------- #
_LIGHTING_EFFECT = {
    "@@type": "LightingEffect",
    "ambientLight": {"@@type": "AmbientLight", "color": [255,255,255], "intensity": 0.5},
    "_lights": [{"@@type": "DirectionalLight", "color": [255,255,255],
                 "intensity": 1.0, "direction": [-2, -4, -1]}],
}


def render_point_cloud(bundle, visible_layers, point_size):
    """PyDeck point cloud renderer (downsampled fallback)."""
    meta = bundle["meta"]

    import pandas as pd
    frames = [df for name, df in bundle["layers"].items()
              if visible_layers.get(name, True)]
    if not frames:
        st.info("No layers selected — tick at least one in the sidebar.")
        return

    with st.spinner("Preparing point cloud…"):
        combined = pd.concat(frames, ignore_index=True).copy()
        r, g, b  = _vivid_colors(combined["z"].values)
        combined["r"], combined["g"], combined["b"] = r, g, b
        combined["z"] = combined["z"].round(1)

        bounds   = meta["bounds_4326"]
        clat     = (bounds["lat_min"] + bounds["lat_max"]) / 2
        clon     = (bounds["lon_min"] + bounds["lon_max"]) / 2
        lon_span = bounds["lon_max"] - bounds["lon_min"]
        zoom     = max(10, min(17, round(np.log2(360 / lon_span) - 1)))

        layer = pdk.Layer(
            "PointCloudLayer", data=combined,
            get_position=["lon", "lat", "z"], get_color=["r", "g", "b"],
            point_size=point_size, pickable=True,
            material={"ambient":0.4,"diffuse":0.6,"shininess":32,
                      "specularColor":[60,60,60]},
        )
        view = pdk.ViewState(latitude=clat, longitude=clon,
                             zoom=zoom, pitch=55, bearing=0)
        deck = pdk.Deck(layers=[layer], initial_view_state=view,
                        map_provider=None, map_style="",
                        tooltip={"text": "Z: {z} m"},
                        effects=[_LIGHTING_EFFECT])

    st.pydeck_chart(deck, use_container_width=True)

    n_display = sum(len(bundle["layers"][n]) for n in visible_layers
                    if visible_layers[n] and n in bundle["layers"])
    cols = st.columns(4)
    cols[0].metric("Displayed",      _fmt_n(n_display))
    cols[1].metric("Total surveyed", _fmt_n(meta.get("n_total", 0)))
    cols[2].metric("Elevation min",  f"{meta.get('z_min', 0):.1f} m")
    cols[3].metric("Elevation max",  f"{meta.get('z_max', 0):.1f} m")

    st.caption(
        f"Downsampled LIDAR · {_fmt_n(n_display)} display pts from "
        f"{_fmt_n(meta.get('n_total', 0))} raw · colours = elevation"
    )


# --------------------------------------------------------------------------- #
#  DTM surface tab
# --------------------------------------------------------------------------- #
DTM_COLORSCALES = ["Turbo", "Viridis", "Plasma", "Inferno", "Earth", "RdYlBu"]


def render_dtm(bundle, vert_exag, colorscale):
    dtm, dtm_meta = bundle["dtm"], bundle["dtm_meta"]
    if dtm is None:
        st.info("No terrain model available for this survey.")
        return

    finite_mask = np.isfinite(dtm)
    if not finite_mask.any():
        st.warning("Terrain model contains no valid elevation values.")
        return

    z_min_disp = float(np.min(dtm[finite_mask]))
    z_max_disp = float(np.max(dtm[finite_mask]))

    bounds = dtm_meta["bounds_4326"]
    lat0   = (bounds["lat_min"] + bounds["lat_max"]) / 2
    lon0   = (bounds["lon_min"] + bounds["lon_max"]) / 2
    m_lat  = 111_320.0
    m_lon  = 111_320.0 * np.cos(np.radians(lat0))

    nrows, ncols = dtm.shape
    lons = np.linspace(bounds["lon_min"], bounds["lon_max"], ncols)
    lats = np.linspace(bounds["lat_max"], bounds["lat_min"], nrows)
    x_m  = (lons - lon0) * m_lon
    y_m  = (lats - lat0) * m_lat
    z_pl = dtm * vert_exag

    span_x    = float(x_m.max() - x_m.min())
    span_y    = float(y_m.max() - y_m.min())
    span_z    = (z_max_disp - z_min_disp) * vert_exag
    ms        = max(span_x, span_y, span_z, 1)
    tick_vals = np.linspace(z_min_disp, z_max_disp, 6)
    scene_bg  = "rgb(12, 12, 18)"

    fig = go.Figure(data=[go.Surface(
        z=z_pl, x=x_m, y=y_m,
        colorscale=colorscale, cmin=z_min_disp, cmax=z_max_disp,
        colorbar=dict(
            title=dict(text="Elevation (m)", font=dict(color="#cccccc")),
            tickvals=tick_vals, ticktext=[f"{v:.0f}" for v in tick_vals],
            tickfont=dict(color="#cccccc"), thickness=14, len=0.7,
        ),
        customdata=dtm,
        hovertemplate=(
            "E: %{x:.0f} m<br>N: %{y:.0f} m<br>"
            "Elevation: %{customdata:.1f} m<extra></extra>"
        ),
    )])

    ax = dict(tickfont=dict(color="#888888"), gridcolor="#2a2a3a",
              showbackground=False)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title=dict(text="Easting offset (m)",  font=dict(color="#aaaaaa")), **ax),
            yaxis=dict(title=dict(text="Northing offset (m)", font=dict(color="#aaaaaa")), **ax),
            zaxis=dict(title=dict(text="Elevation (m)",       font=dict(color="#aaaaaa")), **ax),
            aspectmode="manual",
            aspectratio=dict(x=span_x/ms, y=span_y/ms, z=span_z/ms),
            bgcolor=scene_bg,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=scene_bg,
    )
    st.plotly_chart(fig, use_container_width=True)

    res  = dtm_meta.get("src_resolution_m", "—")
    cols = st.columns(4)
    cols[0].metric("Min elevation", f"{z_min_disp:.1f} m")
    cols[1].metric("Max elevation", f"{z_max_disp:.1f} m")
    cols[2].metric("Relief",        f"{z_max_disp - z_min_disp:.1f} m")
    cols[3].metric("Source res.",   f"{res} m/px" if isinstance(res,(int,float)) else str(res))

    st.caption(
        f"Digital terrain model · {nrows}×{ncols} grid · "
        f"vertical exaggeration {vert_exag:.1f}× · colorscale: {colorscale}"
    )


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    if not check_password():
        st.stop()

    render_header()

    bundle          = None
    visible_layers  = {}
    point_size      = 2
    vert_exag       = 1.0
    colorscale      = "Turbo"
    picked          = PLACEHOLDER
    potree_settings = None

    with st.sidebar:
        st.header("Survey")
        surveys = list_surveys()

        if not surveys:
            st.warning("No surveys found in R2.")
            if st.button("⟳ Refresh"):
                refresh_surveys(); st.rerun()
        else:
            options = [PLACEHOLDER] + [s["name"] for s in surveys]
            picked  = st.selectbox("Dataset", options, index=0)

            if picked != PLACEHOLDER:
                survey = next(s for s in surveys if s["name"] == picked)

                with st.spinner(f"Loading {picked} …"):
                    try:
                        bundle = load_bundle(survey["slug"])
                    except Exception as e:
                        st.error(f"Could not load bundle: {e}")
                        bundle = None

                if bundle:
                    meta       = bundle["meta"]
                    r2_public  = st.secrets.get("r2_public_url", "").strip()
                    use_potree = meta.get("potree_available", False) and bool(r2_public)

                    st.header("Point cloud")

                    if use_potree:
                        point_budget = st.select_slider(
                            "Point budget",
                            options=[500_000, 1_000_000, 2_000_000,
                                     3_000_000, 5_000_000],
                            value=1_000_000,
                            format_func=_fmt_n,
                            help=(
                                "Max points rendered at once. Potree streams "
                                "progressively — higher = more detail but "
                                "slower first navigate."
                            ),
                        )
                        edl_enabled = st.checkbox(
                            "Eye-dome lighting (EDL)", value=True,
                            help=(
                                "Shading technique that gives strong depth "
                                "perception based on neighbouring points."
                            ),
                        )
                        colour_mode = st.selectbox(
                            "Colour by", ["Elevation", "Intensity"],
                            help=(
                                "Elevation = rainbow height map. "
                                "Intensity = laser return strength."
                            ),
                        )
                        potree_settings = {
                            "point_budget": point_budget,
                            "edl_enabled":  edl_enabled,
                            "colour_mode":  colour_mode,
                        }
                    else:
                        point_size = st.slider(
                            "Point size", 1, 6, 2,
                            help="Pixel size of each point.",
                        )
                        for name in meta.get("layers", []):
                            visible_layers[name] = st.checkbox(
                                _layer_label(name), value=True)
                        if not meta.get("potree_available"):
                            st.caption(
                                "💡 Run PotreeConverter + upload tiles to enable "
                                "the full-res viewer for this survey."
                            )

                    if bundle["dtm"] is not None:
                        st.header("Terrain model")
                        vert_exag = st.slider(
                            "Vertical exaggeration", 0.5, 5.0, 1.0, step=0.5,
                            help="1.0 = true scale. Try 2–3× for flat terrain.",
                        )
                        colorscale = st.selectbox(
                            "Colorscale", DTM_COLORSCALES, index=0,
                            help="Turbo = most vivid. Viridis = perceptually uniform.",
                        )

                    st.header("Info")
                    viewer_mode = "Potree (full res)" if use_potree else "PyDeck (downsampled)"
                    st.markdown(
                        f"**Viewer:** {viewer_mode}  \n"
                        f"**Mode:** {meta['mode'].capitalize()}  \n"
                        f"**Raw points:** {_fmt_n(meta.get('n_total', 0))}  \n"
                        f"**Source CRS:** {meta.get('crs_source', '—')}  \n"
                        f"**DTM:** "
                        f"{'Included' if meta.get('dtm_available') else 'Not available'}"
                    )

                    st.header("Downloads")
                    raw_las_key = meta.get("raw_las_key")
                    raw_dtm_key = meta.get("raw_dtm_key")

                    if raw_las_key:
                        try:
                            st.link_button(
                                "⬇  Raw LAS (full res)",
                                presigned_url(raw_las_key, ttl=900),
                                use_container_width=True, type="primary")
                        except Exception:
                            st.caption("LAS download unavailable")

                    if raw_dtm_key:
                        try:
                            st.link_button(
                                "⬇  Terrain model (GeoTIFF)",
                                presigned_url(raw_dtm_key, ttl=900),
                                use_container_width=True)
                        except Exception:
                            st.caption("DTM download unavailable")

                    if not raw_las_key and not raw_dtm_key:
                        st.caption("Re-run preprocessing to add downloads.")

            if st.button("⟳ Refresh surveys"):
                refresh_surveys(); st.rerun()

    if picked == PLACEHOLDER or bundle is None:
        st.markdown(
            """<div class="wingtra-card">
              <div class="wingtra-card-title">Select a survey to begin</div>
              <p>Pick a dataset from the <b>Dataset</b> dropdown in the sidebar.</p>
              <ul>
                <li>Full-resolution LIDAR via Potree — progressive streaming,
                    eye-dome lighting, up to 131M points</li>
                <li>Digital terrain model with adjustable vertical exaggeration
                    and colorscale</li>
                <li>Download the raw LAS and terrain model GeoTIFF</li>
              </ul>
            </div>""",
            unsafe_allow_html=True,
        )
        st.stop()

    tab_labels = ["☁  Point Cloud"]
    if bundle["dtm"] is not None:
        tab_labels.append("⛰  Terrain Model")

    tabs = st.tabs(tab_labels)

    with tabs[0]:
        if potree_settings is not None:
            render_potree_viewer(bundle, potree_settings)
        else:
            render_point_cloud(bundle, visible_layers, point_size)

    if len(tabs) > 1:
        with tabs[1]:
            render_dtm(bundle, vert_exag, colorscale)


if __name__ == "__main__":
    main()
