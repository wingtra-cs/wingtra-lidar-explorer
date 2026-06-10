"""
LIDAR Survey Explorer
=====================
Secrets (.streamlit/secrets.toml):
    app_password   = "..."
    r2_public_url  = "https://pub-<hash>.r2.dev"
    [r2]
    account_id = "..."
    access_key = "..."
    secret_key = "..."
    bucket     = "ptpn-bucket"
"""

import os, hmac, base64, json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import pydeck as pdk
import plotly.graph_objects as go
from lidar_r2 import list_surveys, load_bundle, refresh_surveys, presigned_url

APP_TITLE   = "LIDAR Survey Explorer"
LOGO_PATH   = "assets/wingtra_logo.png"
PLACEHOLDER = "— Select a survey —"
MERCURY600 = "#1C2E36"; MERCURY500 = "#233A44"
URANUS300  = "#A3BABD"; URANUS0    = "#EDF2F2"
SUN_ORANGE = "#F46F29"

CSS = f"""
<style>
  .stApp{{background-color:{URANUS0}}}
  section[data-testid="stSidebar"]{{background-color:{MERCURY500}}}
  section[data-testid="stSidebar"] *{{color:#FFF}}
  section[data-testid="stSidebar"] input,
  section[data-testid="stSidebar"] textarea{{background-color:#16242B;color:#FFF}}
  section[data-testid="stSidebar"] div[data-baseweb="select"]>div{{
      background-color:#16242B;color:#FFF;border-color:#3A4F59}}
  section[data-testid="stSidebar"] div[data-baseweb="select"] svg{{fill:#FFF}}
  div[data-baseweb="popover"] ul,ul[role="listbox"]{{background-color:#233A44}}
  div[data-baseweb="popover"] li,ul[role="listbox"] li{{color:#FFF}}
  ul[role="listbox"] li:hover{{background-color:#16242B}}
  section[data-testid="stSidebar"] [data-testid="stTooltipHoverTarget"],
  section[data-testid="stSidebar"] [data-testid="stTooltipHoverTarget"] svg,
  section[data-testid="stSidebar"] [data-testid="stTooltipHoverTarget"] svg *{{
      color:{URANUS300}!important;fill:{URANUS300}!important;
      stroke:{URANUS300}!important;opacity:1!important}}
  section[data-testid="stSidebar"] button[kind="secondary"],
  section[data-testid="stSidebar"] [data-testid="baseButton-secondary"]{{
      background-color:#16242B!important;border:1px solid #3A4F59!important;color:#FFF!important}}
  section[data-testid="stSidebar"] [data-testid="stLinkButtonContainer"] a,
  section[data-testid="stSidebar"] .stLinkButton a{{
      background-color:#16242B!important;border:1px solid #3A4F59!important;color:#FFF!important}}
  .wingtra-header{{display:flex;align-items:center;gap:18px;background-color:{MERCURY600};
      border-bottom:3px solid {SUN_ORANGE};padding:14px 22px;margin:-1.2rem -1.2rem 1rem -1.2rem}}
  .wingtra-header img{{height:30px}}
  .wingtra-wordmark{{color:{SUN_ORANGE};font-size:28px;font-weight:800;letter-spacing:.5px}}
  .wingtra-title{{color:#FFF;font-size:20px;font-weight:800;line-height:1.2}}
  .wingtra-subtitle{{color:{URANUS300};font-size:12.5px;font-weight:500}}
  div[data-testid="stMetric"]{{background:#FFF;border:1px solid #D7E0E2;border-radius:12px;
      padding:12px 16px;box-shadow:0 1px 3px rgba(28,46,54,.06)}}
  div[data-testid="stMetricLabel"] p{{color:#5b6b72;font-weight:600}}
  .wingtra-card{{max-width:680px;margin:32px auto 0;padding:28px 32px;background:#FFF;
      border:1px solid #D7E0E2;border-radius:14px;box-shadow:0 1px 4px rgba(28,46,54,.08)}}
  .wingtra-card-title{{color:{MERCURY600};font-size:20px;font-weight:800;margin-bottom:8px}}
  .wingtra-card p,.wingtra-card li{{color:#46555b;font-size:14px;line-height:1.6}}
  .wingtra-card ul{{margin:10px 0 0;padding-left:22px}}
</style>
"""


# --------------------------------------------------------------------------- #
#  Branding + auth
# --------------------------------------------------------------------------- #
def _logo_b64():
    if os.path.exists(LOGO_PATH):
        try:
            return base64.b64encode(open(LOGO_PATH, "rb").read()).decode()
        except Exception:
            pass
    return None


def render_header(subtitle="Point cloud · Digital terrain model"):
    st.markdown(CSS, unsafe_allow_html=True)
    b64 = _logo_b64()
    brand = (f'<img src="data:image/png;base64,{b64}" alt="Wingtra"/>'
             if b64 else f'<span class="wingtra-wordmark">wingtra</span>')
    sub = f'<div class="wingtra-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="wingtra-header">{brand}'
        f'<div><div class="wingtra-title">{APP_TITLE}</div>{sub}</div></div>',
        unsafe_allow_html=True)


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
    return {"ground": "Ground (class 2)", "nonground": "Non-ground",
            "points": "All points (unclassified)"}.get(name, name)


def _vivid_colors(z_vals):
    v = np.asarray(z_vals, dtype="float64")
    fin = v[np.isfinite(v)]
    if fin.size == 0:
        return (np.zeros_like(v, dtype="uint8"),) * 3
    lo, hi = np.percentile(fin, 2), np.percentile(fin, 98)
    t = np.clip((v - lo) / max(hi - lo, 1e-3), 0, 1)
    T = [0, .15, .35, .55, .75, .9, 1]
    return (np.interp(t, T, [20,0,0,50,230,255,180]).astype("uint8"),
            np.interp(t, T, [20,80,200,220,220,120,0]).astype("uint8"),
            np.interp(t, T, [160,220,230,80,0,0,0]).astype("uint8"))


# --------------------------------------------------------------------------- #
#  Contour helpers
# --------------------------------------------------------------------------- #
_ALL_INTERVALS = [1, 2, 5, 10, 20]


def _contour_options(relief):
    """Intervals that produce at least 3 contour lines."""
    return [i for i in _ALL_INTERVALS if relief / i >= 3] or [_ALL_INTERVALS[0]]


def _compute_contour_geojson(dtm, dtm_meta, interval):
    """Compute contour lines from DTM. Returns GeoJSON string (EPSG:4326).
    Requires matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fm = np.isfinite(dtm)
    zlo, zhi = float(np.min(dtm[fm])), float(np.max(dtm[fm]))

    bd = dtm_meta["bounds_4326"]
    nr, nc = dtm.shape
    lons = np.linspace(bd["lon_min"], bd["lon_max"], nc)
    lats = np.linspace(bd["lat_max"], bd["lat_min"], nr)
    LON, LAT = np.meshgrid(lons, lats)

    levels = np.arange(
        np.floor(zlo / interval) * interval,
        np.ceil(zhi / interval) * interval + interval * 0.5,
        interval)
    levels = levels[(levels >= zlo - interval) & (levels <= zhi + interval)]

    fig_c, ax_c = plt.subplots()
    cs = ax_c.contour(LON, LAT, dtm, levels=levels)
    plt.close(fig_c)

    features = []
    for i, level in enumerate(cs.levels):
        if i >= len(cs.collections):
            break
        for path in cs.collections[i].get_paths():
            verts = path.vertices
            if len(verts) < 2:
                continue
            coords = [[round(float(x), 7), round(float(y), 7)]
                      for x, y in verts]
            features.append({
                "type": "Feature",
                "properties": {"elevation_m": round(float(level), 1)},
                "geometry": {"type": "LineString", "coordinates": coords},
            })

    geojson = {
        "type": "FeatureCollection",
        "properties": {
            "interval_m": interval, "crs": "EPSG:4326",
            "source": "LIDAR Survey Explorer",
            "n_contours": len(features),
        },
        "features": features,
    }
    return json.dumps(geojson)


# =========================================================================== #
#  Potree viewer
# =========================================================================== #
_CDN_DEPS = [
    "https://cdnjs.cloudflare.com/ajax/libs/three.js/r124/three.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/jquery/3.1.1/jquery.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/jqueryui/1.12.1/jquery-ui.min.js",
    "https://cdn.jsdelivr.net/gh/potree/potree@1.8.2/libs/tween/tween.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/i18next/21.6.14/i18next.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/jstree/3.3.11/jstree.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/proj4js/2.8.0/proj4.js",
    "https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/spectrum/1.8.0/spectrum.min.js",
]


def _potree_html(potree_url, assets_base, survey_name, point_budget):
    dep_tags = "\n".join(f'<script src="{u}"></script>' for u in _CDN_DEPS)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="stylesheet" href="{assets_base}/potree.css">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
html,body{{width:100%;height:100%;overflow:hidden;background:#000}}
#potree_render_area{{position:absolute;width:100%;height:100%}}
#potree_sidebar_container{{display:none!important}}
.potree_menu_toggle{{display:none!important}}
#loading_overlay{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  font-family:-apple-system,sans-serif;text-align:center;pointer-events:none;z-index:10}}
#loading_overlay .msg{{font-size:14px;color:#A3BABD;margin-bottom:6px}}
#loading_overlay .sub{{font-size:12px;color:#4a5a60}}
#error_overlay{{display:none;position:absolute;top:50%;left:50%;
  transform:translate(-50%,-50%);font-family:-apple-system,sans-serif;
  text-align:center;max-width:480px;padding:0 20px;z-index:10}}
#error_overlay .title{{font-size:14px;font-weight:700;color:#F46F29;margin-bottom:8px}}
#error_overlay .detail{{font-size:11px;color:#6a7880;word-break:break-all}}
</style>
</head>
<body>
<div id="potree_render_area"></div>
<div id="loading_overlay">
  <div class="msg">Loading LIDAR point cloud…</div>
  <div class="sub">Streaming tiles from cloud storage</div>
</div>
<div id="error_overlay">
  <div class="title">Could not load point cloud</div>
  <div class="detail" id="error_detail">Check console (F12).</div>
</div>

<script>
(function(){{
  var _W=window.Worker;
  var _forceBlob=(window.location.origin==="null");
  window.Worker=function(url,opts){{
    if(!_forceBlob){{try{{return new _W(url,opts)}}catch(e){{}}}}
    var base=url.substring(0,url.lastIndexOf("/")+1);
    var xhr=new XMLHttpRequest();xhr.open("GET",url,false);xhr.send();
    if(xhr.status!==200)throw new Error("Worker fetch "+xhr.status);
    var code=xhr.responseText.replace(
      /importScripts\\s*\\(\\s*['"]([^'"]+)['"]\\s*\\)/g,
      function(m,p){{if(/^https?:\\/\\//.test(p))return m;
        try{{return 'importScripts("'+new URL(p,base).href+'")'}}
        catch(err){{return m}}}});
    return new _W(URL.createObjectURL(
      new Blob([code],{{type:"application/javascript"}})),opts);
  }};
  window.Worker.prototype=_W.prototype;
}})();
</script>

{dep_tags}
<script src="{assets_base}/libs/other/BinaryHeap.js"></script>
<script src="{assets_base}/potree.js"></script>

<script>
function showError(msg){{
  var lo=document.getElementById("loading_overlay");
  var eo=document.getElementById("error_overlay");
  var ed=document.getElementById("error_detail");
  if(lo)lo.style.display="none";
  if(ed)ed.textContent=String(msg);
  if(eo)eo.style.display="block";
}}

if(typeof Potree==="undefined"){{showError("Potree failed to load.");}}
else{{
  var el=document.getElementById("potree_render_area");

  function _awaitSize(){{
    if(el.clientWidth>0 && el.clientHeight>0) _initViewer();
    else requestAnimationFrame(_awaitSize);
  }}

  function _initViewer(){{
    try{{
      Potree.scriptPath="{assets_base}";
      var viewer=new Potree.Viewer(el);
      viewer.setEDLEnabled(false);
      viewer.setPointBudget({point_budget});
      viewer.setBackground("black");
      viewer.setFOV(60);

      Potree.loadPointCloud("{potree_url}","{survey_name}",function(e){{
        var lo=document.getElementById("loading_overlay");
        if(lo)lo.style.display="none";
        var pc=e.pointcloud, mat=pc.material;
        mat.activeAttributeName="elevation";
        mat.gradient=Potree.Gradients.RAINBOW;
        mat.size=1;
        mat.pointSizeType=Potree.PointSizeType.ADAPTIVE;
        viewer.scene.addPointCloud(pc);
        viewer.fitToScreen();
      }});
    }}catch(e){{showError("Init: "+e);}}
  }}

  _awaitSize();
}}
</script>
</body>
</html>"""


def render_potree_viewer(bundle, point_budget):
    meta = bundle["meta"]
    r2p = st.secrets.get("r2_public_url", "").rstrip("/")
    if not r2p:
        st.warning("**r2_public_url** not set — falling back to PyDeck.")
        render_point_cloud(bundle, {}, 2)
        return
    pk = meta.get("potree_key", f"lidar/{meta['slug']}/potree/metadata.json")
    html = _potree_html(
        f"{r2p}/{pk}", f"{r2p}/potree-assets",
        meta.get("name", meta.get("slug", "Survey")), point_budget)
    components.html(html, height=650, scrolling=False)
    co = st.columns(3)
    co[0].metric("Total points", _fmt_n(meta.get("n_total", 0)))
    co[1].metric("Elevation min", f"{meta.get('z_min', 0):.1f} m")
    co[2].metric("Elevation max", f"{meta.get('z_max', 0):.1f} m")
    st.caption(
        f"Full {_fmt_n(meta.get('n_total', 0))}-point LIDAR · Potree streaming · "
        f"elevation colour · budget: {_fmt_n(point_budget)}")


# =========================================================================== #
#  PyDeck fallback — black background, no basemap
# =========================================================================== #
_EMPTY_MAP_STYLE = (
    "data:application/json,"
    + json.dumps({"version": 8, "name": "empty", "sources": {}, "layers": []})
)

_LIGHTING = {
    "@@type": "LightingEffect",
    "ambientLight": {
        "@@type": "AmbientLight", "color": [255, 255, 255], "intensity": 0.5},
    "_lights": [{
        "@@type": "DirectionalLight", "color": [255, 255, 255],
        "intensity": 1.0, "direction": [-2, -4, -1]}],
}


def render_point_cloud(bundle, visible_layers, point_size):
    meta = bundle["meta"]
    import pandas as pd

    frames = [df for n, df in bundle["layers"].items()
              if visible_layers.get(n, True)]
    if not frames:
        st.info("No layers selected.")
        return

    with st.spinner("Preparing point cloud…"):
        c = pd.concat(frames, ignore_index=True).copy()
        r, g, b = _vivid_colors(c["z"].values)
        c["r"], c["g"], c["b"] = r, g, b
        c["z"] = c["z"].round(1)
        bd = meta["bounds_4326"]
        clat = (bd["lat_min"] + bd["lat_max"]) / 2
        clon = (bd["lon_min"] + bd["lon_max"]) / 2
        zoom = max(10, min(17,
                   round(np.log2(360 / (bd["lon_max"] - bd["lon_min"])) - 1)))
        layer = pdk.Layer(
            "PointCloudLayer", data=c,
            get_position=["lon", "lat", "z"],
            get_color=["r", "g", "b"],
            point_size=point_size, pickable=True,
            material={"ambient": .4, "diffuse": .6, "shininess": 32,
                      "specularColor": [60, 60, 60]})
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(
                latitude=clat, longitude=clon,
                zoom=zoom, pitch=55, bearing=0),
            map_style=_EMPTY_MAP_STYLE,
            tooltip={"text": "Z: {z} m"},
            effects=[_LIGHTING])
    st.pydeck_chart(deck, use_container_width=True)
    nd = sum(len(bundle["layers"][n]) for n in visible_layers
             if visible_layers[n] and n in bundle["layers"])
    co = st.columns(4)
    co[0].metric("Displayed", _fmt_n(nd))
    co[1].metric("Total", _fmt_n(meta.get("n_total", 0)))
    co[2].metric("Z min", f"{meta.get('z_min', 0):.1f} m")
    co[3].metric("Z max", f"{meta.get('z_max', 0):.1f} m")


# =========================================================================== #
#  DTM — hillshade + contours (visual + downloadable GeoJSON)
# =========================================================================== #
DTM_CS = ["Turbo", "Viridis", "Plasma", "Inferno", "Earth", "RdYlBu"]


def render_dtm(bundle, ve, cs):
    dtm, dm = bundle["dtm"], bundle["dtm_meta"]
    if dtm is None:
        st.info("No terrain model.")
        return
    fm = np.isfinite(dtm)
    if not fm.any():
        st.warning("No valid elevations.")
        return
    zlo, zhi = float(np.min(dtm[fm])), float(np.max(dtm[fm]))
    relief = zhi - zlo

    bd = dm["bounds_4326"]
    lat0 = (bd["lat_min"] + bd["lat_max"]) / 2
    lon0 = (bd["lon_min"] + bd["lon_max"]) / 2
    mo = 111320.0 * np.cos(np.radians(lat0))
    ml = 111320.0
    nr, nc = dtm.shape
    lons = np.linspace(bd["lon_min"], bd["lon_max"], nc)
    lats = np.linspace(bd["lat_max"], bd["lat_min"], nr)
    xm = (lons - lon0) * mo
    ym = (lats - lat0) * ml
    zp = dtm * ve
    sx = float(xm.max() - xm.min())
    sy = float(ym.max() - ym.min())
    sz = relief * ve
    ms = max(sx, sy, sz, 1)
    tv = np.linspace(zlo, zhi, 6)
    bg = "rgb(12,12,18)"

    # -- Contour interval (on DTM tab only, not sidebar) --------------------
    int_opts = _contour_options(relief)
    labels = ["Off"] + [f"{i}m" for i in int_opts]
    contour_choice = st.selectbox(
        "Contour interval", labels, index=0, key="dtm_contour_sel",
        help=f"Elevation contour lines. Survey relief: {relief:.0f}m.")
    contour_int = (int(contour_choice.replace("m", ""))
                   if contour_choice != "Off" else None)

    # -- Plotly contour config (visual on 3D surface) -----------------------
    contour_cfg = dict(z=dict(show=False))
    if contour_int is not None:
        c_start = np.floor(zlo / contour_int) * contour_int * ve
        c_end   = np.ceil(zhi / contour_int) * contour_int * ve
        contour_cfg = dict(z=dict(
            show=True, start=c_start, end=c_end,
            size=contour_int * ve,
            color="rgba(255,255,255,0.5)", width=1,
            usecolormap=False,
            highlightcolor="rgba(255,255,255,0.8)",
            project_z=False))

    # -- Surface plot -------------------------------------------------------
    fig = go.Figure(data=[go.Surface(
        z=zp, x=xm, y=ym,
        colorscale=cs, cmin=zlo * ve, cmax=zhi * ve,
        colorbar=dict(
            title=dict(text="Elevation (m)", font=dict(color="#ccc")),
            tickvals=tv * ve,
            ticktext=[f"{v:.0f}" for v in tv],
            tickfont=dict(color="#ccc"), thickness=14, len=.7),
        lighting=dict(ambient=0.35, diffuse=0.7, specular=0.15,
                      roughness=0.6, fresnel=0.05),
        lightposition=dict(x=-500, y=500, z=2000),
        contours=contour_cfg,
        hoverinfo="skip",
    )])

    ax = dict(tickfont=dict(color="#888"), gridcolor="#2a2a3a",
              showbackground=False)
    fig.update_layout(scene=dict(
        xaxis=dict(title=dict(text="Easting (m)",
                              font=dict(color="#aaa")), **ax),
        yaxis=dict(title=dict(text="Northing (m)",
                              font=dict(color="#aaa")), **ax),
        zaxis=dict(title=dict(text="Elevation (m)",
                              font=dict(color="#aaa")), **ax),
        aspectmode="manual",
        aspectratio=dict(x=sx/ms, y=sy/ms, z=sz/ms),
        bgcolor=bg),
        margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=bg)
    st.plotly_chart(fig, use_container_width=True)

    # -- Metrics ------------------------------------------------------------
    res = dm.get("src_resolution_m", "—")
    co = st.columns(4)
    co[0].metric("Min", f"{zlo:.1f} m")
    co[1].metric("Max", f"{zhi:.1f} m")
    co[2].metric("Relief", f"{relief:.1f} m")
    co[3].metric("Res.", f"{res} m/px" if isinstance(res, (int, float))
                 else str(res))
    cstr = f" · contours: {contour_int}m" if contour_int else ""
    st.caption(f"DTM · {nr}x{nc} · vert. exag. {ve:.1f}x · {cs}"
               f" · hillshade{cstr}")

    # -- Contour download: prepare + download -------------------------------
    if contour_int is not None:
        slug = dm.get("slug", bundle["meta"].get("slug", ""))

        if (st.session_state.get("prep_contour_int") != contour_int
                or st.session_state.get("prep_contour_slug") != slug):
            st.session_state.pop("prep_contour_json", None)
            st.session_state.pop("prep_contour_int", None)
            st.session_state.pop("prep_contour_slug", None)

        prepared = st.session_state.get("prep_contour_json")

        if prepared:
            n_feat = json.loads(prepared).get(
                "properties", {}).get("n_contours", "?")
            st.download_button(
                f"⬇  Download contours — {contour_int}m interval"
                f" ({n_feat} lines)",
                data=prepared,
                file_name=f"contours_{contour_int}m.geojson",
                mime="application/json",
                use_container_width=True)
            st.caption("Change the interval above to regenerate.")
        else:
            if st.button("Prepare contours for download",
                         use_container_width=True):
                with st.spinner(f"Computing {contour_int}m contours…"):
                    gj = _compute_contour_geojson(dtm, dm, contour_int)
                st.session_state["prep_contour_json"] = gj
                st.session_state["prep_contour_int"]  = contour_int
                st.session_state["prep_contour_slug"] = slug
                st.rerun()


# =========================================================================== #
#  Main
# =========================================================================== #
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    if not check_password():
        st.stop()
    render_header()

    bundle = None; vis = {}; ps = 2; ve = 1.0; cs = "Turbo"
    picked = PLACEHOLDER; pb = 1_000_000; use_pot = False

    with st.sidebar:
        st.header("Survey")
        surveys = list_surveys()
        if not surveys:
            st.warning("No surveys in R2.")
            if st.button("⟳ Refresh"):
                refresh_surveys(); st.rerun()
        else:
            opts = [PLACEHOLDER] + [s["name"] for s in surveys]
            picked = st.selectbox("Dataset", opts, index=0)

            if picked != PLACEHOLDER:
                survey = next(s for s in surveys if s["name"] == picked)
                with st.spinner(f"Loading {picked}…"):
                    try:
                        bundle = load_bundle(survey["slug"])
                    except Exception as e:
                        st.error(f"Load error: {e}")
                        bundle = None

                if bundle:
                    meta = bundle["meta"]
                    r2p = st.secrets.get("r2_public_url", "").strip()
                    use_pot = (meta.get("potree_available", False)
                               and bool(r2p))

                    st.header("Point cloud")
                    if use_pot:
                        pb = st.select_slider(
                            "Point budget",
                            options=[500_000, 1_000_000, 2_000_000,
                                     3_000_000, 5_000_000],
                            value=1_000_000, format_func=_fmt_n,
                            help="Max points rendered. "
                                 "Lower = smoother navigation.")
                    else:
                        ps = st.slider("Point size", 1, 6, 2,
                                       help="Pixel size per point.")
                        for n in meta.get("layers", []):
                            vis[n] = st.checkbox(_layer_label(n), value=True)
                        if not meta.get("potree_available"):
                            st.caption("Run PotreeConverter to enable "
                                       "full-res viewer.")

                    if bundle["dtm"] is not None:
                        st.header("Terrain model")
                        ve = st.slider("Vertical exag.", 0.5, 5.0, 1.0,
                                       step=0.5, help="1.0 = true scale.")
                        cs = st.selectbox("Colorscale", DTM_CS, index=0)

                    st.header("Info")
                    vm = ("Potree (full)" if use_pot
                          else "PyDeck (sampled)")
                    st.markdown(
                        f"**Viewer:** {vm}  \n"
                        f"**Mode:** {meta['mode'].capitalize()}  \n"
                        f"**Points:** {_fmt_n(meta.get('n_total', 0))}  \n"
                        f"**CRS:** {meta.get('crs_source', '—')}  \n"
                        f"**DTM:** "
                        f"{'Yes' if meta.get('dtm_available') else 'No'}")

                    st.header("Downloads")
                    rlk = meta.get("raw_las_key")
                    rdk = meta.get("raw_dtm_key")
                    if rlk:
                        try:
                            st.link_button(
                                "⬇ Raw LAS",
                                presigned_url(rlk, ttl=900),
                                use_container_width=True, type="primary")
                        except Exception:
                            st.caption("LAS unavailable")
                    if rdk:
                        try:
                            st.link_button(
                                "⬇ Terrain GeoTIFF",
                                presigned_url(rdk, ttl=900),
                                use_container_width=True)
                        except Exception:
                            st.caption("DTM unavailable")
                    if not rlk and not rdk:
                        st.caption("Re-run preprocessing for downloads.")

            if st.button("⟳ Refresh surveys"):
                refresh_surveys(); st.rerun()

    if picked == PLACEHOLDER or bundle is None:
        st.markdown(
            '<div class="wingtra-card">'
            '<div class="wingtra-card-title">Select a survey</div>'
            '<p>Pick a dataset from the sidebar.</p><ul>'
            '<li>Full LIDAR via Potree — streaming, 131M+ points</li>'
            '<li>Terrain model with hillshade, contours, '
            'and vertical exaggeration</li>'
            '<li>Downloadable contour GeoJSON at selectable intervals</li>'
            '<li>Download raw LAS and GeoTIFF</li></ul></div>',
            unsafe_allow_html=True)
        st.stop()

    tabs_l = ["☁ Point Cloud"]
    if bundle["dtm"] is not None:
        tabs_l.append("⛰ Terrain Model")
    tabs = st.tabs(tabs_l)

    with tabs[0]:
        if use_pot:
            render_potree_viewer(bundle, pb)
        else:
            render_point_cloud(bundle, vis, ps)

    if len(tabs) > 1:
        with tabs[1]:
            render_dtm(bundle, ve, cs)


if __name__ == "__main__":
    main()
